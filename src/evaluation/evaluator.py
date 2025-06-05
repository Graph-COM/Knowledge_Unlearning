import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import logging
from tqdm import tqdm
import torch
import re
from collections import deque
from src.data.knowledge_graph import KnowledgeGraph
from src.models.model_loader import load_model, generate_text, generate_text_batch
import networkx as nx
import transformers
import math
import numpy as np

def load_query_dataset(query_dataset_path):
    """
    Load the query dataset from a JSON file.
    
    Args:
        query_dataset_path (str): Path to the query dataset JSON file.
    
    Returns:
        dict: Dictionary with keys "q" (prompts) and "a" (ground-truth answers).
    """
    with open(query_dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def extract_relation_and_score_all(response, threshold=2):
    """
    Extracts relationships and confidence scores from the model's response.
    Filters out relations with confidence below the given threshold.

    Args:
        response (str): The raw text output from the model.
        threshold (int): Minimum confidence score required to keep a relation.

    Returns:
        str: A formatted string containing relations with confidence scores above the threshold.
             Format: "relation1 confidence1; relation2 confidence2; ..."
    """
    response = response.strip()  # Remove leading/trailing whitespace
    response = re.sub(r"\s+", " ", response)  # Normalize spaces

    # **Extract all relation-confidence pairs**
    matches = re.findall(r"RELATIONSHIP:\s*([\w_]+)\s*;\s*CONFIDENCE:\s*(\d+)", response, re.IGNORECASE)

    # Convert extracted confidence values to integers, ensuring they fall within {0,1,2,3}
    filtered_relations = []
    for rel, conf in matches:
        try:
            conf_int = int(conf)
            if 0 <= conf_int <= 3 and conf_int >= threshold:
                filtered_relations.append(f"{rel} {conf_int}")
        except ValueError:
            continue  # Ignore invalid confidence values

    # If no valid relations remain, return "UNKNOWN"
    return "; ".join(filtered_relations) if filtered_relations else "UNKNOWN"


def evaluate_model(model, tokenizer, query_data, multi_gpu=False, max_new_tokens=200, threshold=2):
    """
    Evaluate the model using the query dataset.

    Args:
        model: The HuggingFace model.
        tokenizer: The corresponding tokenizer.
        query_data (dict): A dict with keys "q" (list of prompts) and "a" (optional ground truth).
        multi_gpu (bool): If True, assumes device_map='auto' is used for multi-GPU setups.
        max_new_tokens (int): Maximum number of new tokens to generate per prompt.
        threshold (int): Minimum confidence score required to keep a relation.

    Returns:
        list: Filtered model responses for each prompt.
    """
    prompts = query_data["q"]
    results = []

    model.eval()

    # **Determine device: Multi-GPU uses automatic assignment, Single-GPU explicitly uses "cuda"**
    if not multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device  # Multi-GPU mode (auto-sharded)

    logging.info(f"Model is running on device: {device}")

    for prompt in tqdm(prompts, desc="Generating responses", unit="query"):
        # Tokenize input text and move to the correct device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate output using the model
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
                top_k=10,  # Ensures more deterministic generation
                top_p=0.9  # Balances diversity while keeping coherence
            )

        # Decode generated output and clean response
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Added lines below to remove prompt from the raw_response
        if raw_response.startswith(prompt):
            raw_response = raw_response[len(prompt):]
        cleaned_response = extract_relation_and_score_all(raw_response, threshold=threshold)
        results.append(cleaned_response)

    return results

def save_evaluation_results(results, output_file):
    """
    Save evaluation results to a file.
    
    Args:
        results (list): List of responses.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(res + "\n")
    logging.info(f"Saved evaluation results to {output_file}.")

def precompute_relation_options(kg):
    """
    Precompute unique relation options directly from the knowledge graph.

    Args:
        kg (KnowledgeGraph): The knowledge graph object.

    Returns:
        str: Formatted string of relations to be inserted in the prompt.
    """
    # Extract relations directly from the knowledge graph to ensure accuracy
    relation_options = sorted(set(kg.relations.values()))

    # Format the relation descriptions while preserving readability
    relation_descs = "\n".join([f"- {rel.replace('_', ' ').replace('/', ' / ')}" for rel in relation_options])

    return relation_descs

def process_custom_relations(relations_list):
    """
    Process a custom list of relations to format them in the same way as precompute_relation_options.
    
    Args:
        relations_list (list): List of relation strings from config
        
    Returns:
        str: Formatted string of relations to be inserted in the prompt
    """
    # Sort relations to maintain consistency
    relation_options = sorted(relations_list)
    
    # Format the relation descriptions while preserving readability
    relation_descs = "\n".join([f"- {rel.replace('_', ' ').replace('/', ' / ')}" for rel in relation_options])
    
    return relation_descs

def extract_relation_and_score(response, threshold=2):
    """
    Extracts relationships and confidence scores from the model's response.
    Filters out relations with confidence below the given threshold.

    Args:
        response (str): The raw text output from the model.
        threshold (int): Minimum confidence score required to keep a relation.

    Returns:
        list: A list of valid (relation, confidence) pairs above the threshold.
    """
    response = response.strip()
    response = re.sub(r"\s+", " ", response)  # Normalize spaces

    matches = re.findall(r"RELATIONSHIP:\s*([/\w_-]+)\s*;\s*CONFIDENCE:\s*([\w\s]+)", response, re.IGNORECASE)
    valid_relations = []
    for rel, conf in matches:
        valid_relations.append((rel, conf))
    return valid_relations

def query_llm_for_relations(model, tokenizer, entity1, entity2, relation_descs, multi_gpu=False, max_new_tokens=200, threshold=2):
    """
    Query the LLM to determine the relationships between two entities using full-text triples.

    Args:
        model: The HuggingFace model.
        tokenizer: The corresponding tokenizer.
        entity1 (str): First entity.
        entity2 (str): Second entity.
        relation_descs (str): Precomputed relation descriptions for query.
        multi_gpu (bool): If True, assumes multi-GPU execution.
        max_new_tokens (int): Maximum number of new tokens to generate.
        threshold (int): Confidence threshold for valid relations.

    Returns:
        list: A list of valid (relation, confidence) pairs.
    """
    system_message = (
        "You are an expert in knowledge graphs. Your task is to analyze potential relationships "
        "between two entities using a given known relationship. "
        "Your response should be structured, precise, and strictly follow the required format."
    )

    relation_descs = relation_descs.lstrip('- ').strip()

    user_message = f"""
    Please respond following this exact template:
    Please first provide a explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
    Your explanation should be limited to 2-3 sentences and be concise.
    **Form your explanation like:**
    "{entity1} is <description1>, and {entity2} is <description2>. <reasoning>. Therefore <conclusion>."
    Then output one line indicating your confidence score for the relationship.
    **Your response must be formatted exactly as follows:**
    Evaluation evidence: <evaluation explanation here>
    RELATIONSHIP: {relation_descs}; CONFIDENCE: <score>
    Provide answers for in a semicolon-separated format.

    ### Task:
    Determine the confidence for the potential relationship '{relation_descs}' between:
    - **Entity 1**: '{entity1}'
    - **Entity 2**: '{entity2}'

    ### Confidence score:
    - No meaningful
    - Plausible 
    - Definite
    - I don't know.
    """
    
    model.eval()
    
    if not multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device

    # Convert to chat format
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=terminators,  # Correct termination handling
            temperature=1.0,  # Lower randomness
            top_k=20,  # Reduce noise
            top_p=0.85  # Balance coherence and diversity
        )

    # Extract only the newly generated response
    response_tokens = outputs[0][input_ids.shape[-1]:]
    raw_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print(raw_response)
    # import pdb; pdb.set_trace()
    return extract_relation_and_score(raw_response, threshold=threshold)


def query_llm_for_relations_by_perplexity(model, tokenizer, entity1, entity2, relation_descs, multi_gpu=False, max_new_tokens=100, threshold=500):
    """
    Determine the relationship between two entities based on perplexity computed from multiple probing templates.
    
    For each candidate relation in 'relation_descs', this function constructs multiple prompt texts using different 
    probing templates that embed the two entities and the candidate relation. It then computes the perplexity for each 
    template and selects the minimum perplexity as the final score for that candidate. This approach aims to directly 
    reflect the LLM's memory regarding the triple, as a lower perplexity on at least one template indicates a better fit.
    
    Note:
    - In this implementation, the perplexity is computed over the entire text (including the prompt).
    - While an average perplexity could be used, taking the minimum perplexity is assumed to be more indicative of 
      the LLM's memory for the triple, given that template variations may yield large differences.
    
    Args:
        model: The HuggingFace model.
        tokenizer: The corresponding tokenizer.
        entity1 (str): The first entity.
        entity2 (str): The second entity.
        relation_descs (str): A string containing candidate relation descriptions, formatted with each candidate on a new line prefixed by "- ".
        multi_gpu (bool): If True, assumes multi-GPU execution.
        max_new_tokens (int): Kept for interface consistency but not used in this method.
        threshold (float): The perplexity threshold; only relations with a minimum perplexity lower than or equal to this value are returned.
        
    Returns:
        list: A list of valid (relation, min_perplexity) pairs where min_perplexity <= threshold.
    """
    model.eval()
    if not multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device

    templates = [
        "The head entity '{entity1}' is {relation} the tail entity '{entity2}'.",
        "From '{entity1}' to '{entity2}', the directed relationship is: {relation}.",
        "When considering '{entity1}' as the head and '{entity2}' as the tail, the relation is: {relation}.",
        "The directed connection from '{entity1}' (head) to '{entity2}' (tail) is: {relation}.",
        "If '{entity1}' is treated as the head and '{entity2}' as the tail, then they are {relation}.",
        "Given the directed pair — head: '{entity1}', tail: '{entity2}' — their relationship is: {relation}.",
        "In a directed sense, '{entity1}' leads to '{entity2}' with the relation: {relation}.",
        "The relation from head '{entity1}' to tail '{entity2}' is: {relation}.",
        "Considering the directionality, head entity '{entity1}' and tail entity '{entity2}' are {relation}.",
        "Directed relation: '{entity1}' (head) is {relation} to '{entity2}' (tail).",
        "For the ordered pair, where '{entity1}' is head and '{entity2}' is tail, the relation is: {relation}.",
        "Specifically, '{entity1}' as the head is {relation} to '{entity2}' as the tail.",
        "By direction, the relationship from '{entity1}' (head) to '{entity2}' (tail) is: {relation}.",
        "With '{entity1}' as the head and '{entity2}' as the tail, the relation can be expressed as: {relation}.",
        "If we establish a directed link where '{entity1}' is the head and '{entity2}' is the tail, then the connection is: {relation}."
    ]
    
    # Parse candidate relations; assume each candidate is on a new line prefixed with "- "
    candidates = [line.strip()[2:].strip() for line in relation_descs.splitlines() if line.strip().startswith("-")]
    valid_relations = []
    
    for candidate in candidates:
        relation = candidate.strip()
        if not relation:
            continue
        
        perplexities = []
        # Compute perplexity for each template
        for template in templates:
            prompt_text = template.format(entity1=entity1, entity2=entity2, relation=relation)
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss  # average cross-entropy loss per token
                ppl = math.exp(loss.item())  # perplexity = exp(loss)
                perplexities.append(ppl)
        
        # Use the minimum or average perplexity among all templates as the score
        min_perplexity = min(perplexities) if perplexities else float("inf")
        average_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("inf")
        
        # Filter: only keep relations with a minimum perplexity lower than or equal to the threshold
        if average_perplexity <= threshold:
            valid_relations.append((relation, average_perplexity))
    
    return valid_relations

def query_llm_for_relations_by_logprob(model, tokenizer, entity1, entity2, relation_descs, multi_gpu=False, max_new_tokens=100, threshold=-5.0):
    """
    Determine the relationship between two entities based on the average log probability computed from multiple chat-format prompts.
    
    For each candidate relation in 'relation_descs', this function constructs several chat-formatted prompts.
    Each prompt consists of a system message (with detailed instructions) and a user message that includes a task description
    and the candidate relation inserted after the "Answer:" marker. The model's loss (i.e. negative average log probability)
    is computed only on the answer portion. A higher average log probability (closer to zero) indicates higher confidence.
    
    Args:
        model: The HuggingFace model.
        tokenizer: The corresponding tokenizer, which supports chat formatting via apply_chat_template.
        entity1 (str): The head entity.
        entity2 (str): The tail entity.
        relation_descs (str): A string containing candidate relation descriptions, with each candidate on a new line prefixed by "- ".
        multi_gpu (bool): If True, assumes multi-GPU execution.
        max_new_tokens (int): Maximum number of new tokens to generate (not used in logprob computation).
        threshold (float): The log probability threshold; only candidate relations with an average log probability 
                           (computed on the answer tokens) greater than or equal to this value are returned.
    
    Returns:
        list: A list of valid (relation, avg_log_probability) pairs where avg_log_probability >= threshold.
    """
    import torch
    model.eval()
    if not multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device

    # Define a common system message with detailed instructions.
    system_message = (
        "You are an expert in knowledge graphs. Your task is to evaluate a candidate directed relation between two entities. "
        "First analyze the semantic properties of both entities and reason about potential connections before providing your final answer. "
        "Here are two examples of good reasoning:\n\n"
        "Example 1: For head entity 'Paris' and tail entity 'France', reasoning process: "
        "Paris is a city and France is a country. Geographically, cities are located within countries. "
        "Therefore, the relation is: locatedIn\n\n"
        "Example 2: For head entity 'Shakespeare' and tail entity 'Hamlet', reasoning process: "
        "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
        "The most specific relation between an author and their work is: wrote\n\n"
        # "Additional chain-of-thought guidelines:\n"
        "Be deliberate and analytical in your reasoning before providing your final answer. "
        "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    )

    # Define multiple user message templates to provide varied phrasing.
    user_templates = [
        "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', what is the directed relationship? Answer: {relation}",
        "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), determine the relation. Answer: {relation}",
        "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'. Answer: {relation}",
        "Task: Considering '{entity1}' as head and '{entity2}' as tail, what directed connection exists between them? Answer: {relation}",
        "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Answer: {relation}",
        "Task: What is the direct relationship from '{entity1}' to '{entity2}'? Answer: {relation}",
        "Task: Identify the directed relation from '{entity1}' to '{entity2}'. Answer: {relation}",
        "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner. Answer: {relation}",
        "Task: Specify the relationship that links '{entity1}' to '{entity2}'. Answer: {relation}",
        "Task: '{entity1}' is connected to '{entity2}' via what relation? Answer: {relation}",
        "Task: Find the specific connection from '{entity1}' to '{entity2}'. Answer: {relation}",
        "Task: In a knowledge graph, if '{entity1}' is the source node and '{entity2}' is the target node, what edge label connects them? Answer: {relation}",
        "Task: A knowledge graph contains entities '{entity1}' and '{entity2}'. What is the directed relationship from the first to the second? Answer: {relation}",
        "Task: When representing knowledge as a graph, how would you label the edge pointing from '{entity1}' to '{entity2}'? Answer: {relation}",
        "Task: In the direction from '{entity1}' towards '{entity2}', the relationship type is what? Answer: {relation}",
        "Task: Starting at '{entity1}' and following the graph edge to '{entity2}', how would you classify this connection? Answer: {relation}",
        "Task: The directional relationship '{entity1}' → '{entity2}' is best described as what? Answer: {relation}",
        "Task: As a knowledge graph expert, classify the edge type from node '{entity1}' to node '{entity2}'. Answer: {relation}",
        "Task: In semantic network terminology, the predicate linking subject '{entity1}' to object '{entity2}' is what? Answer: {relation}",
        "Task: Following RDF triple format, if '{entity1}' is the subject and '{entity2}' is the object, what is the predicate? Answer: {relation}",
        "Task: I have two entities: '{entity1}' and '{entity2}'. What directed relationship exists between them, with the first being the head? Answer: {relation}",
        "Task: Entity pair analysis: '{entity1}' → ? → '{entity2}'. Fill in the missing relation. Answer: {relation}",
        "Task: Complete the knowledge triple: ({entity1}, ?, {entity2}) where ? represents the relation type. Answer: {relation}",
        "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', first analyze their potential connection carefully. Consider their semantic categories, common contexts, and potential relationships. Then determine the most appropriate directed relationship. Answer: {relation}",
        "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), please think step by step. First, categorize both entities. Next, identify possible domains where they interact. Finally, determine the specific relation between them. Answer: {relation}",
        "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'. First consider the nature of '{entity1}' (e.g., person, place, concept), then the nature of '{entity2}', and analyze what logical connection could exist between such entities. Answer: {relation}",
        "Task: Considering '{entity1}' as head and '{entity2}' as tail, carefully analyze their semantic types. Then identify any factual or conceptual connections by considering encyclopedic knowledge. Based on your analysis, what directed connection exists between them? Answer: {relation}",
        "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). First, consider what type of entity each one is. Second, think about possible ways these entities could be connected in the real world. Finally, specify the most precise relationship. Answer: {relation}",
        "Task: What is the direct relationship from '{entity1}' to '{entity2}'? Before answering, analyze: (1) the category of each entity, (2) their possible interactions based on real-world knowledge, and (3) the most specific way to describe their connection. Answer: {relation}",
        "Task: Identify the directed relation from '{entity1}' to '{entity2}'. Consider the following: Are they in the same domain? What is the hierarchical relationship between them? How specifically are they connected in a knowledge graph context? Answer: {relation}",
        "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner. First, list potential categories for each entity. Then, evaluate possible relationships based on these categories. Finally, select the most appropriate relation. Answer: {relation}",
        "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Let's reason step by step:\n1) Entity types: What type of entity is '{entity1}'? What type of entity is '{entity2}'?\n2) Possible connections: Given their types, what are possible ways they could be related?\n3) Most specific relation: Among these possibilities, which one most precisely captures their relationship?\nAnswer: {relation}"
        #    "Task: Given head entity '{entity1}' (type: {type1}) and tail entity '{entity2}' (type: {type2}), determine their relationship. Answer: {relation}"
        "Task: Suppose there is an association from '{entity1}' to '{entity2}'. Describe that association. Answer: {relation}",
        "Task: We have two concepts: '{entity1}' and '{entity2}'. In the knowledge graph, they are connected by what property? Answer: {relation}",
        "Task: Considering '{entity1}' and '{entity2}', which property captures how the former is linked to the latter? Answer: {relation}",
        "Task: Based on your knowledge, how is '{entity1}' related to '{entity2}'? Answer: {relation}",
        "Task: If we were to label the relationship between '{entity1}' and '{entity2}' in a triple, what would it be? Answer: {relation}",
        "Task: Provide the term that logically connects '{entity1}' to '{entity2}' in a knowledge graph. Answer: {relation}",
        "Task: In an ontology, which predicate associates '{entity1}' with '{entity2}'? Answer: {relation}",
        "Task: Given '{entity1}' and '{entity2}', identify the relevant link that ties them together. Answer: {relation}",
        "Task: In a semantic web setting, how would you define the bond between '{entity1}' and '{entity2}'? Answer: {relation}",
        "Task: For '{entity1}' referring to '{entity2}', which label best expresses their connection? Answer: {relation}"
    ]
    
    # Parse candidate relations (assume each candidate is on a new line starting with "- ")
    candidates = [line.strip()[2:].strip() for line in relation_descs.splitlines() if line.strip().startswith("-")]
    valid_relations = []
    
    for candidate in candidates:
        # Skip empty candidates
        relation = candidate.strip()
        if not relation:
            continue

        log_probs = []
        # For each candidate, compute log probability over several chat-format templates
        for user_template in user_templates:
            # Construct chat messages in a list of dicts (chat format)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_template.format(entity1=entity1, entity2=entity2, relation=relation)}
            ]
            
            # Convert chat messages to a prompt using the tokenizer's chat template utility.
            # (Assumes the tokenizer supports apply_chat_template.)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            
            input_ids = inputs["input_ids"]
            # Decode the full prompt to locate the "Answer:" marker.
            full_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if "Answer:" not in full_prompt:
                continue
            # The answer portion starts immediately after "Answer:".
            question_part = full_prompt.split("Answer:", 1)[0] + "Answer:"
            # Tokenize the question part (without adding extra special tokens).
            question_tokens = tokenizer(question_part, add_special_tokens=False)["input_ids"]
            question_length = len(question_tokens)
            
            # Create labels: mask out the question part so that loss is computed only on the answer tokens.
            labels = input_ids.clone()
            labels[:, :question_length] = -100
            
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss  # loss computed only on the answer tokens
                avg_log_prob = -loss.item()  # convert negative loss to average log probability
                log_probs.append(avg_log_prob)
        
        # Average the log probability across all chat templates for the candidate relation.
        if log_probs:
            score = max(log_probs)
        else:
            score = -float("inf")
        
        # Filter: keep candidate relations with an average log probability >= threshold.
        if score >= threshold:
            valid_relations.append((relation, score))
    
    return valid_relations

def convert_score_to_confidence(relation_score_pairs, thresholds):
    """
    Convert log probability scores to confidence scores (0-3) based on thresholds.

    Args:
        relation_score_pairs (list): List of (relation, score) tuples.
        thresholds (list): List of three thresholds [t0, t1, t2] for determining confidence scores.

    Returns:
        list: List of (relation, confidence_score) tuples where confidence_score is 0, 1, 2, or 3.
    """
    result = []
    for relation, score in relation_score_pairs:
        if score < thresholds[0]:
            confidence_score = 0
        elif thresholds[0] <= score < thresholds[1]:
            confidence_score = 1
        elif thresholds[1] <= score < thresholds[2]:
            confidence_score = 2
        else:  # score >= thresholds[2]
            confidence_score = 3

        result.append((relation, confidence_score, score))

    return result

def filter_by_confidence_threshold(relation_confidence_pairs, confidence_threshold):
    """
    Filter relation-confidence pairs by a confidence score threshold.

    Args:
        relation_confidence_pairs (list): List of (relation, confidence_score) tuples.
        confidence_threshold (int): Minimum confidence score required (0-3).

    Returns:
        list: Filtered list of (relation, confidence_score) tuples where confidence_score >= confidence_threshold.
    """
    return [(relation, confidence_score, score) for relation, confidence_score, score in relation_confidence_pairs if confidence_score >= confidence_threshold]

def query_llm_for_relations_by_logprob_copy(model, tokenizer, entity1, entity2, relation_descs, multi_gpu=False, max_new_tokens=100, thresholds=[0.2, 0.5, 0.8], confidence_threshold=2):
    """
    Determine the relationship between two entities based on the log probability computed from multiple chat-format prompts.
    Convert log probabilities to confidence scores and filter by minimum confidence threshold.

    Args:
        model: The HuggingFace model.
        tokenizer: The corresponding tokenizer, which supports chat formatting via apply_chat_template.
        entity1 (str): The head entity.
        entity2 (str): The tail entity.
        relation_descs (str): A string containing candidate relation descriptions.
        multi_gpu (bool): If True, assumes multi-GPU execution.
        max_new_tokens (int): Maximum number of new tokens to generate (not used in logprob computation).
        thresholds (list): List of three threshold values [t0, t1, t2] for determining confidence scores.
        confidence_threshold (int): Minimum confidence score required (0-3).

    Returns:
        list: A list of valid (relation, confidence_score) pairs where confidence_score >= confidence_threshold.
    """
    import torch
    model.eval()
    if not multi_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = model.device

    # Define a common system message with detailed instructions.
    system_message = (
        "You are an expert in knowledge graphs. Your task is to evaluate a candidate directed relation between two entities. "
        "First analyze the semantic properties of both entities and reason about potential connections before providing your final answer. "
        "Here are two examples of good reasoning:\n\n"
        "Example 1: For head entity 'Paris' and tail entity 'France', reasoning process: "
        "Paris is a city and France is a country. Geographically, cities are located within countries. "
        "Therefore, the relation is: locatedIn\n\n"
        "Example 2: For head entity 'Shakespeare' and tail entity 'Hamlet', reasoning process: "
        "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
        "The most specific relation between an author and their work is: wrote\n\n"
        "Be deliberate and analytical in your reasoning before providing your final answer. "
        "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    )

    # Define multiple user message templates to provide varied phrasing.
    user_templates = [
        "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Answer: {relation}",
    ]

    # Parse candidate relations (assume each candidate is on a new line starting with "- ")
    candidates = [line.strip()[2:].strip() for line in relation_descs.splitlines() if line.strip().startswith("-")]
    logprob_relations = []

    for candidate in candidates:
        # Skip empty candidates
        relation = candidate.strip()
        if not relation:
            continue

        log_probs = []
        # For each candidate, compute log probability over several chat-format templates
        for user_template in user_templates:
            # Construct chat messages in a list of dicts (chat format)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_template.format(entity1=entity1, entity2=entity2, relation=relation)}
            ]

            # Convert chat messages to a prompt using the tokenizer's chat template utility.
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            input_ids = inputs
            # Decode the full prompt to locate the "Answer:" marker.
            full_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if "Answer:" not in full_prompt:
                continue
            # The answer portion starts immediately after "Answer:".
            question_part = full_prompt.split("Answer:", 1)[0] + "Answer:"
            # Tokenize the question part (without adding extra special tokens).
            question_tokens = tokenizer(question_part, add_special_tokens=False)["input_ids"]
            question_length = len(question_tokens)

            # Create labels: mask out the question part so that loss is computed only on the answer tokens.
            labels = input_ids.clone()
            labels[:, :question_length] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss  # loss computed only on the answer tokens
                avg_log_prob = -loss.item()  # convert negative loss to average log probability
                log_probs.append(avg_log_prob)

        # Get the maximum log probability across all templates for the candidate relation.
        if log_probs:
            score = max(log_probs)
            logprob_relations.append((relation, score))

    # Convert log probabilities to confidence scores
    confidence_relations = convert_score_to_confidence(logprob_relations, thresholds)

    # Filter by confidence threshold
    filtered_relations = filter_by_confidence_threshold(confidence_relations, confidence_threshold)

    return filtered_relations

def query_llm_for_relations_by_logprob_batch(model, tokenizer, entity_pairs, relation_candidates, thresholds=[0.2, 0.5, 0.8], confidence_threshold=2):
    """
    Optimized batch processing that groups prompts by token position to maximize parallel processing.
    
    This version groups prompts by token position (first token across all relations, second token, etc.)
    to maximize the benefits of batch processing while still computing logprobs for all tokens.
    
    Args:
        model: The vLLM model.
        tokenizer: The corresponding tokenizer.
        entity_pairs (list): List of (entity1, entity2) tuples.
        relation_candidates (list): List of relation strings to evaluate.
        thresholds (list): List of three threshold values [t0, t1, t2] for determining confidence scores.
        confidence_threshold (int): Minimum confidence score required (0-3).
        
    Returns:
        dict: A dictionary mapping (entity1, entity2) to a list of (relation, confidence_score) pairs.
    """
    # Define the system message
    system_message = (
        "You are an expert in knowledge graphs. Your task is to evaluate a candidate directed relation between two entities. "
        "First analyze the semantic properties of both entities and reason about potential connections before providing your final answer. "
        "Here are two examples of good reasoning:\n\n"
        "Example 1: For head entity 'Paris' and tail entity 'France', reasoning process: "
        "Paris is a city and France is a country. Geographically, cities are located within countries. "
        "Therefore, the relation is: locatedIn\n\n"
        "Example 2: For head entity 'Shakespeare' and tail entity 'Hamlet', reasoning process: "
        "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
        "The most specific relation between an author and their work is: wrote\n\n"
        "Be deliberate and analytical in your reasoning before providing your final answer. "
        "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    )

    # Define template for prompts
    user_template = "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Answer: {relation}"

    # Create prefixes for each entity pair (up to "Answer:")
    logging.info(f"Processing {len(entity_pairs)} entity pairs with {len(relation_candidates)} relation candidates")
    prefixes = {}
    for entity1, entity2 in entity_pairs:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_template.format(entity1=entity1, entity2=entity2, relation="")}
        ]
        
        # Convert to chat format up to "Answer:"
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        

        prefix = prompt.split("Answer:", 1)[0] + "Answer:"
        prefixes[(entity1, entity2)] = prefix
    
    # Prepare relation tokens information
    relation_tokens_info = {}
    max_tokens = 0
    for relation in relation_candidates:
        # Adding a space before the relation to match how it would appear after "Answer:"
        relation_with_space = " " + relation.strip()
        token_ids = tokenizer.encode(relation_with_space, add_special_tokens=False)
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        relation_tokens_info[relation] = {
            "token_ids": token_ids,
            "tokens": tokens
        }
        max_tokens = max(max_tokens, len(tokens))
    
    logging.info(f"Max tokens in any relation: {max_tokens}")
    
    # Initialize result storage
    all_logprobs = {}  # (entity1, entity2, relation) -> [token_logprobs]
    
    # Process token by token for all entity pairs and relations
    for token_idx in range(max_tokens):
        logging.info(f"Processing token position {token_idx+1}/{max_tokens}")
        batch_prompts = []
        batch_metadata = []  # (entity1, entity2, relation, token_id, token_text)
        
        for (entity1, entity2), prefix in prefixes.items():
            for relation, token_info in relation_tokens_info.items():
                # Skip if this relation doesn't have enough tokens
                if token_idx >= len(token_info["tokens"]):
                    continue
                
                # Get current state of prefix
                if token_idx == 0:
                    # First token - use original prefix
                    current_prefix = prefix
                else:
                    # Later tokens - use updated prefix with previous tokens
                    prev_tokens = token_info["tokens"][:token_idx]
                    current_prefix = prefix + "".join(prev_tokens)
                
                # Add to batch
                batch_prompts.append(current_prefix)
                batch_metadata.append((entity1, entity2, relation, 
                                      token_info["token_ids"][token_idx],
                                      token_info["tokens"][token_idx]))
        
        # Skip if batch is empty
        if not batch_prompts:
            continue
        
        # Process batch
        logging.info(f"Batch size for token position {token_idx+1}: {len(batch_prompts)}")
        batch_results = generate_text_batch(
            model,
            tokenizer,
            batch_prompts,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5
        )
        
        # Process results
        for i, ((entity1, entity2, relation, token_id, token_text), result) in enumerate(zip(batch_metadata, batch_results)):
            _, logprobs_data = result
            key = (entity1, entity2, relation)
            
            # Ensure the key exists in storage
            if key not in all_logprobs:
                all_logprobs[key] = []
            
            # Extract token logprob
            token_logprob = -200.0  # Default penalty
            
            if "logprobs" in logprobs_data and logprobs_data["logprobs"]:
                first_token_logprobs = logprobs_data["logprobs"][0]
                
                # Try to find by token ID
                if token_id in first_token_logprobs:
                    logprob_obj = first_token_logprobs[token_id]
                    if hasattr(logprob_obj, 'logprob'):
                        token_logprob = logprob_obj.logprob
                    elif isinstance(logprob_obj, (int, float)):
                        token_logprob = logprob_obj
                else:
                    # Try matching by decoded token
                    for id_key, logprob_obj in first_token_logprobs.items():
                        if hasattr(logprob_obj, 'decoded_token'):
                            decoded = logprob_obj.decoded_token
                            if decoded.replace('Ġ', ' ').strip() == token_text.strip():
                                token_logprob = logprob_obj.logprob
                                break
            
            # Store the logprob
            all_logprobs[key].append(token_logprob)
    
    # Convert logprobs to confidence scores
    pair_relation_logprobs = {}
    for (entity1, entity2, relation), token_logprobs in all_logprobs.items():
        # Skip if no logprobs
        if not token_logprobs:
            continue
        
        # Calculate the token-level log probability
        avg_logprob = sum(token_logprobs)
        
        # Group by entity pair
        pair_key = (entity1, entity2)
        if pair_key not in pair_relation_logprobs:
            pair_relation_logprobs[pair_key] = []
        
        pair_relation_logprobs[pair_key].append((relation, avg_logprob))

    # Convert logprobs to confidence scores and filter by threshold
    results = {}
    for pair_key, relation_score_pairs in pair_relation_logprobs.items():
        # Convert to confidence scores
        confidence_pairs = convert_score_to_confidence(relation_score_pairs, thresholds)
        
        # Filter by confidence threshold
        filtered_pairs = filter_by_confidence_threshold(confidence_pairs, confidence_threshold)
        
        # Store results if there are any valid pairs
        if filtered_pairs:
            results[pair_key] = filtered_pairs
    
    return results



def query_llm_for_relations_by_judgement_batch(model, model_name, tokenizer, entity_pairs, relation_candidates, times, thresholds=[0.2, 0.5, 0.8], confidence_threshold=2, batch_size=5000):
    """
    Optimized batch processing that gets Yes, No, and Unknown probabilities with batch-based processing.
    
    Args:
        model: The vLLM model.
        model_name: Name of the model being used.
        tokenizer: The corresponding tokenizer.
        entity_pairs (list): List of (entity1, entity2) tuples.
        relation_candidates (list): List of relation strings to evaluate.
        times (int): Number of times to repeat each prompt for robust estimation.
        thresholds (list): List of three threshold values [t0, t1, t2] for determining confidence scores.
        confidence_threshold (int): Minimum confidence score required (0-3).
        batch_size (int): Maximum number of prompts to process in a single batch.
        
    Returns:
        dict: A dictionary mapping (entity1, entity2) to a list of (relation, confidence_score) pairs.
    """
    import math
    import numpy as np
    from collections import defaultdict

    # Define the system message
    system_message = (
    "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown . "
    "First analyze the semantic properties of both entities, and then reason about whether the relation mentioned in the task is appropriate for these two entities.\n\n"
    "Here is an example of a correct relation:\n\n"
    "Example 1: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "'wrote' is one of the most specific relations between an author and their work, so the relation 'wrote' is correct.\n\n"
    "Here is an example of an incorrect relation:\n\n"
    "Example 2: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'locatedIn', reasoning process: "
    "Shakespeare is a person and Hamlet is a literary work. The relation 'locatedIn' typically describes spatial or geographic relationships, "
    "which does not apply to an author and their work. Therefore, the relation 'locatedIn' is incorrect.\n\n"
    "Here is an example of an unknown relation:\n\n"
    "Example 3: For head entity 'Hamlet', tail entity 'Existentialism', relation 'influencedBy', reasoning process: "
    "Hamlet is a literary work, while Existentialism is a philosophical movement. "
    "Although some scholars interpret Hamlet's introspective nature as proto-existentialist, there is no widely agreed-upon or factual relationship confirming that Hamlet was directly influenced by Existentialism. "
    "Therefore, the relation 'influencedBy' is unknown.\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'.")
    
    # Define template for prompts
    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",}

    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
    
    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0

    # Create combinations of entity pairs and relations
    all_combinations = []
    for h, t in entity_pairs: 
        for r in relation_candidates:
            all_combinations.append((h, r, t))
    
    # Get token IDs for "Yes", "No", and "Unknown"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()
    
    # Prepare all combination-prompt pairs (with duplication)
    all_prompts = []
    for h, r, t in all_combinations:
        if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
        prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
        
        # Add this prompt 'times' times for robust estimation
        for _ in range(times):
            all_prompts.append((prefix, (h, r, t)))
    
    # Initialize results container
    all_results = []  # List of (combo, result) tuples
    
    # Process in batches
    for i in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[i:i + batch_size]
        
        # Extract prefixes and corresponding combinations
        prefixes = [p[0] for p in batch_prompts]
        combo_indices = [p[1] for p in batch_prompts]
        
        # Make batch call to vLLM
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            prefixes,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5,
        )
        
        # Process outputs for this batch
        for j, output in enumerate(batch_outputs):
            combo = combo_indices[j]
            logprobs_data = output[1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            
            yes_found = no_found = unknown_found = False
            yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
            
            # Store top tokens with their probabilities for temperature scaling
            top_tokens = []
            
            # First collect all top tokens
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    logprob = logprob_obj.logprob
                    prob = math.exp(logprob)
                    top_tokens.append((decoded, logprob, prob))
            
            # Apply temperature scaling for Llama models
            if use_temperature_scaling and top_tokens:
                # Scale probabilities with temperature
                scaled_probs = [prob ** (1.0 / scaling_temperature) for _, _, prob in top_tokens]
                
                # Normalize probabilities to sum to 1
                total_scaled_prob = sum(scaled_probs)
                if total_scaled_prob > 0:
                    normalized_scaled_probs = [p / total_scaled_prob for p in scaled_probs]
                else:
                    normalized_scaled_probs = [1.0 / len(scaled_probs)] * len(scaled_probs)
                
                # Update top_tokens with scaled probabilities and corresponding logprobs
                for token_idx in range(len(top_tokens)):
                    token = top_tokens[token_idx][0]
                    scaled_prob = normalized_scaled_probs[token_idx]
                    scaled_logprob = math.log(max(scaled_prob, 1e-40))
                    top_tokens[token_idx] = (token, scaled_logprob, scaled_prob)
            
            # Find Yes, No, and Unknown in top tokens (after scaling if applicable)
            for token, logprob, prob in top_tokens:
                if token == yes_decoded and not yes_found:
                    yes_logprob = logprob
                    yes_prob = prob
                    yes_found = True
                
                if token == no_decoded and not no_found:
                    no_logprob = logprob
                    no_prob = prob
                    no_found = True

                if token == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob
                    unknown_prob = prob
                    unknown_found = True
                    
                # Break early if we found all three
                if yes_found and no_found and unknown_found:
                    break
            
            # Use very low probability if not found in top-k
            if not yes_found:
                yes_logprob = -100.0
                yes_prob = 1e-40
            
            if not no_found:
                no_logprob = -100.0
                no_prob = 1e-40

            if not unknown_found:
                unknown_logprob = -100.0
                unknown_prob = 1e-40
            
            # Store this result
            all_results.append((combo, {
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob)
            }))
    
    # Group results by combination
    results = {combo: [] for combo in all_combinations}
    for combo, result in all_results:
        results[combo].append(result)

    # Calculate statistics and confidence scores
    stats = {combo: {
        "Yes_prob_mean": 0.0,
        "Yes_prob_std": 0.0,
        "Yes_logprob_mean": 0.0,
        "Yes_logprob_std": 0.0,
        "No_prob_mean": 0.0,
        "No_prob_std": 0.0,
        "No_logprob_mean": 0.0,
        "No_logprob_std": 0.0,
        "Unknown_prob_mean": 0.0,  
        "Unknown_prob_std": 0.0,
        "Unknown_logprob_mean": 0.0,
        "Unknown_logprob_std": 0.0 
    } for combo in all_combinations}

    # Calculate mean and std for each metric
    pair_relation_scores = {}
    for combo in all_combinations:
        (entity1, relation, entity2) = combo
        
        # Extract values from results
        yes_probs = [run["Yes"][1] for run in results[combo]]
        yes_logprobs = [run["Yes"][0] for run in results[combo]]
        no_probs = [run["No"][1] for run in results[combo]]
        no_logprobs = [run["No"][0] for run in results[combo]]
        unknown_probs = [run["Unknown"][1] for run in results[combo]]  
        unknown_logprobs = [run["Unknown"][0] for run in results[combo]]

        # Calculate statistics
        stats[combo]["Yes_prob_mean"] = np.mean(yes_probs)
        stats[combo]["Yes_prob_std"] = np.std(yes_probs)
        stats[combo]["Yes_logprob_mean"] = np.mean(yes_logprobs)
        stats[combo]["Yes_logprob_std"] = np.std(yes_logprobs)
        stats[combo]["No_prob_mean"] = np.mean(no_probs)
        stats[combo]["No_prob_std"] = np.std(no_probs)
        stats[combo]["No_logprob_mean"] = np.mean(no_logprobs)
        stats[combo]["No_logprob_std"] = np.std(no_logprobs)
        stats[combo]["Unknown_prob_mean"] = np.mean(unknown_probs)  
        stats[combo]["Unknown_prob_std"] = np.std(unknown_probs)
        stats[combo]["Unknown_logprob_mean"] = np.mean(unknown_logprobs)
        stats[combo]["Unknown_logprob_std"] = np.std(unknown_logprobs)

        stats[combo]["score"] = stats[combo]["Yes_prob_mean"] * (1 - stats[combo]["Yes_prob_std"])

        pair_key = (entity1, entity2)
        if pair_key not in pair_relation_scores:
            pair_relation_scores[pair_key] = []
        pair_relation_scores[pair_key].append((relation, stats[combo]["score"]))

    # Post-processing
    final_results = {}
    for pair_key, relation_score_pairs in pair_relation_scores.items():
        # Convert to confidence scores
        confidence_pairs = convert_score_to_confidence(relation_score_pairs, thresholds)
        
        # Filter by confidence threshold
        filtered_pairs = filter_by_confidence_threshold(confidence_pairs, confidence_threshold)
        
        # Store results if there are any valid pairs
        if filtered_pairs:
            final_results[pair_key] = filtered_pairs
    
    return final_results


def query_llm_for_relations_by_judgement_entropy_batch(model, model_name, tokenizer, entity_pairs, relation_candidates, entropy_threshold, batch_size):
    """
    Optimized batch processing that gets Yes, No, and Unknown probabilities in batches.
    
    Args:
        model: The vLLM model.
        model_name: Name of the model being used.
        tokenizer: The corresponding tokenizer.
        entity_pairs (list): List of (entity1, entity2) tuples.
        relation_candidates (list): List of relation strings to evaluate.
        entropy_threshold (float): Maximum entropy threshold for filtering results.
        batch_size (int): Maximum number of prompts to process in a single batch.
        
    Returns:
        dict: A dictionary mapping (entity1, entity2) to a list of (relation, entropy, yes_prob, no_prob, unknown_prob) tuples.
    """
    import math
    
    # Define the system message
    system_message = (
    "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown . "
    "First analyze the semantic properties of both entities, and then reason about whether the relation mentioned in the task is appropriate for these two entities.\n\n"
    "Here is an example of a correct relation:\n\n"
    "Example 1: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "'wrote' is one of the most specific relations between an author and their work, so the relation 'wrote' is correct.\n\n"
    "Here is an example of an incorrect relation:\n\n"
    "Example 2: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'locatedIn', reasoning process: "
    "Shakespeare is a person and Hamlet is a literary work. The relation 'locatedIn' typically describes spatial or geographic relationships, "
    "which does not apply to an author and their work. Therefore, the relation 'locatedIn' is incorrect.\n\n"
    "Here is an example of an unknown relation:\n\n"
    "Example 3: For head entity 'Hamlet', tail entity 'Existentialism', relation 'influencedBy', reasoning process: "
    "Hamlet is a literary work, while Existentialism is a philosophical movement. "
    "Although some scholars interpret Hamlet's introspective nature as proto-existentialist, there is no widely agreed-upon or factual relationship confirming that Hamlet was directly influenced by Existentialism. "
    "Therefore, the relation 'influencedBy' is unknown.\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'.")
    
    # Define template for prompts
    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",}

    # Create combinations of entity pairs and relations
    all_combinations = []
    for h, t in entity_pairs: 
        for r in relation_candidates:
            all_combinations.append((h, r, t))
    
    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
    
    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0

    # Get token IDs for "Yes", "No", and "Unknown"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()
    
    # Initialize results dictionary
    all_results = []
    
    # Process combinations in batches
    for i in range(0, len(all_combinations), batch_size):
        batch_combinations = all_combinations[i:i + batch_size]
        
        # Create prefixes for this batch
        prefixes = []
        combo_indices = []
        
        for h, r, t in batch_combinations:
            if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
            prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
            
            prefixes.append(prefix)
            combo_indices.append((h, r, t))
        
        # Make batch call to vLLM for this batch of prompts
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            prefixes,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5,
        )
        
        # Process outputs for this batch
        for j, output in enumerate(batch_outputs):
            combo = combo_indices[j]
            logprobs_data = output[1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            
            yes_found = no_found = unknown_found = False
            yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
            
            # Store all top token probabilities
            top_tokens = []
            
            # Find Yes, No, and Unknown probabilities and collect all top tokens
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    logprob = logprob_obj.logprob
                    prob = math.exp(logprob)
                    
                    # Add to top tokens list
                    top_tokens.append((decoded, logprob, prob))
            
            # Apply temperature scaling for Llama models
            if use_temperature_scaling and top_tokens:
                # Scale probabilities with temperature
                scaled_probs = [prob ** (1.0 / scaling_temperature) for _, _, prob in top_tokens]
                
                # Normalize probabilities to sum to 1
                total_scaled_prob = sum(scaled_probs)
                if total_scaled_prob > 0:
                    normalized_scaled_probs = [p / total_scaled_prob for p in scaled_probs]
                else:
                    normalized_scaled_probs = [1.0 / len(scaled_probs)] * len(scaled_probs)
                
                # Update top_tokens with scaled probabilities and corresponding logprobs
                for token_idx in range(len(top_tokens)):
                    token = top_tokens[token_idx][0]
                    scaled_prob = normalized_scaled_probs[token_idx]
                    scaled_logprob = math.log(max(scaled_prob, 1e-40))
                    top_tokens[token_idx] = (token, scaled_logprob, scaled_prob)
            
            # Find Yes, No, and Unknown in top tokens (after scaling if applicable)
            for token, logprob, prob in top_tokens:
                if token == yes_decoded and not yes_found:
                    yes_logprob = logprob
                    yes_prob = prob
                    yes_found = True
                
                if token == no_decoded and not no_found:
                    no_logprob = logprob
                    no_prob = prob
                    no_found = True

                if token == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob
                    unknown_prob = prob
                    unknown_found = True
            
            # Sort top tokens by probability (descending)
            top_tokens.sort(key=lambda x: x[2], reverse=True)
            
            # Use very low probability if not found in top-k
            if not yes_found:
                yes_logprob = -100.0
                yes_prob = 1e-40
            
            if not no_found:
                no_logprob = -100.0
                no_prob = 1e-40

            if not unknown_found:
                unknown_logprob = -100.0
                unknown_prob = 1e-40
            
            # Store this result
            all_results.append((combo, {
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob),
                "top_tokens": top_tokens
            }))

    # Calculate entropy and organize results by entity pair
    pair_relation_scores = {}
    for combo, result in all_results:
        (entity1, relation, entity2) = combo
        
        yes_prob = result["Yes"][1]
        top_tokens = result.get("top_tokens", [])
        
        # Check if yes_prob is the highest among top tokens
        if top_tokens and top_tokens[0][2] == yes_prob:
            # Calculate entropy using all top tokens
            total_prob = sum(token[2] for token in top_tokens)
            
            if total_prob < 1e-10:  # Avoid division by zero
                entropy = 10000
            else:
                # Normalize probabilities
                normalized_probs = [token[2] / total_prob for token in top_tokens]
                
                # Calculate entropy
                entropy = 0
                for norm_prob in normalized_probs:
                    if norm_prob > 1e-10:
                        entropy -= norm_prob * math.log2(norm_prob)
        else:
            entropy = 10000

        no_prob = result["No"][1]
        unknown_prob = result["Unknown"][1]
        
        pair_key = (entity1, entity2)
        if pair_key not in pair_relation_scores:
            pair_relation_scores[pair_key] = []
        pair_relation_scores[pair_key].append((relation, entropy, yes_prob, no_prob, unknown_prob))

    # Post-processing
    results = {}
    for pair_key, relation_score_pairs in pair_relation_scores.items():
        results[pair_key] = [(relation, entropy, yes_prob, no_prob, unknown_prob) 
                            for relation, entropy, yes_prob, no_prob, unknown_prob in relation_score_pairs 
                            if entropy <= entropy_threshold]
    
    return results

def construct_subgraph(model, model_name, tokenizer, kg, target_triple, k, times, confidence_threshold, thresholds=[0.2, 0.5, 0.8], entropy_threshold=0.1, batch_size=2000, multi_gpu=False, relation_descs=None, use_vllm=False, use_entropy=False):
    """
    Constructs a k-hop subgraph from a given target triple (A, relation, B).
    Ensures only paths that contribute to the inference of (A, r, B) are retained.
    Supports accelerated batch processing with vLLM.

    Args:
        model: The HuggingFace or vLLM model.
        tokenizer: Tokenizer for the model.
        kg (KnowledgeGraph): The knowledge graph object.
        target_triple (tuple): (head, relation, tail) representing A, relation, B.
        k (int): Maximum hop distance for subgraph construction.
        confidence_threshold (int): Minimum confidence score required (0-3).
        thresholds (list): List of three threshold values [t0, t1, t2] for determining confidence scores.
        multi_gpu (bool): If True, enables multi-GPU execution.
        relation_descs (str): String containing relation descriptions.
        use_vllm (bool): If True, use vLLM batch processing for acceleration.

    Returns:
        list: List of triples (head, relation, tail, confidence) forming the high-confidence subgraph.
    """
    A, relation, B = target_triple
    logging.info(f"Building {k}-hop subgraph from '{A}' to '{B}' with confidence threshold {confidence_threshold}...")
    logging.info(f"Using vLLM batch acceleration: {use_vllm}")

    # Generate relation descriptions if not provided
    if relation_descs is None:
        relation_descs = precompute_relation_options(kg)

    # Parse relation candidates from relation_descs
    relation_candidates = [line.strip()[2:].strip() for line in relation_descs.splitlines() if line.strip().startswith("-")]

    # Initialize necessary sets and graph
    potential_neighbors = kg.get_k_hop_neighbors(A, k=k, sample=False) - {A, B}
    N_star_A = set()
    subgraph_triples = set()
    graph = nx.DiGraph()

    # ***PHASE 1: Process A's neighbors***
    logging.info(f"PHASE 1: Processing {len(potential_neighbors)} potential neighbors of {A}")

    if use_vllm and potential_neighbors and not use_entropy:
        # Batch process all potential neighbors with vLLM
        entity_pairs = [(A, neighbor) for neighbor in potential_neighbors]
        logging.info(f"Batch processing {len(entity_pairs)} entity pairs with vLLM")
        batch_results = query_llm_for_relations_by_judgement_batch(
            model, model_name, tokenizer, entity_pairs, relation_candidates, times=times,
            thresholds=thresholds, confidence_threshold=confidence_threshold, batch_size=batch_size
        )

        # Process batch results
        for (head, tail), relations in batch_results.items():
            for rel, conf, score in relations:
                N_star_A.add(tail)
                subgraph_triples.add((head, rel, tail, conf, score))
                graph.add_edge(head, tail)

        logging.info(f"Added {len(N_star_A)} high-confidence neighbors of {A} to the subgraph")
    elif use_vllm and potential_neighbors and use_entropy:
        # Batch process all potential neighbors with vLLM
        entity_pairs = [(A, neighbor) for neighbor in potential_neighbors]
        logging.info(f"Batch processing {len(entity_pairs)} entity pairs with vLLM")
        batch_results = query_llm_for_relations_by_judgement_entropy_batch(
            model, model_name, tokenizer, entity_pairs, relation_candidates, entropy_threshold=entropy_threshold, batch_size=batch_size
        )

        # Process batch results
        for (head, tail), relations in batch_results.items():
            for relation, entropy, yes_prob, no_prob, unknown_prob in relations:
                N_star_A.add(tail)
                subgraph_triples.add((head, relation, tail, entropy, yes_prob, no_prob, unknown_prob))
                graph.add_edge(head, tail)

        logging.info(f"Added {len(N_star_A)} high-confidence neighbors of {A} to the subgraph") 
    else:
        # Original sequential processing
        for neighbor in tqdm(potential_neighbors, desc="Querying LLM for 1-hop relations"):
            predicted_relations = query_llm_for_relations_by_logprob_copy(
                model, tokenizer, A, neighbor, relation_descs, 
                multi_gpu=multi_gpu, 
                thresholds=thresholds, 
                confidence_threshold=confidence_threshold
            )

            for rel, conf, score in predicted_relations:
                N_star_A.add(neighbor)
                subgraph_triples.add((A, rel, neighbor, conf, score))
                graph.add_edge(A, neighbor)

    # ***PHASE 2: Expand from high-confidence neighbors***
    expanded_neighbors = set()
    logging.info(f"PHASE 2: Expanding from {len(N_star_A)} high-confidence neighbors")

    if use_vllm and N_star_A and not use_entropy:
        # Collect all entity pairs for batch processing
        all_second_hop_pairs = []

        for node in N_star_A:
            second_hop_neighbors = kg.get_k_hop_neighbors(node, k=k, sample=False) - {A, B, node} - N_star_A
            for second_neighbor in second_hop_neighbors:
                all_second_hop_pairs.append((node, second_neighbor))

        logging.info(f"Found {len(all_second_hop_pairs)} second-hop entity pairs to process")

        if all_second_hop_pairs:
            # Batch process all second hop neighbors
            batch_results = query_llm_for_relations_by_judgement_batch(
                model, model_name, tokenizer, all_second_hop_pairs, relation_candidates, times=times,
                thresholds=thresholds, confidence_threshold=confidence_threshold, batch_size=batch_size
            )

            # Process batch results
            for (head, tail), relations in batch_results.items():
                for rel, conf, score in relations:
                    expanded_neighbors.add(tail)
                    subgraph_triples.add((head, rel, tail, conf, score))
                    graph.add_edge(head, tail)

            logging.info(f"Added {len(expanded_neighbors)} expanded neighbors to the subgraph")
            
    elif use_vllm and N_star_A and use_entropy:
        # Collect all entity pairs for batch processing
        all_second_hop_pairs = []

        for node in N_star_A:
            second_hop_neighbors = kg.get_k_hop_neighbors(node, k=k, sample=False) - {A, B, node} - N_star_A
            for second_neighbor in second_hop_neighbors:
                all_second_hop_pairs.append((node, second_neighbor))

        logging.info(f"Found {len(all_second_hop_pairs)} second-hop entity pairs to process")

        if all_second_hop_pairs:
            # Batch process all second hop neighbors
            batch_results = query_llm_for_relations_by_judgement_entropy_batch(
                model, model_name, tokenizer, all_second_hop_pairs, relation_candidates, entropy_threshold=entropy_threshold, batch_size=batch_size
            )

            # Process batch results
            for (head, tail), relations in batch_results.items():
                for relation, entropy, yes_prob, no_prob, unknown_prob in relations:
                    expanded_neighbors.add(tail)
                    subgraph_triples.add((head, relation, tail, entropy, yes_prob, no_prob, unknown_prob))
                    graph.add_edge(head, tail)

            logging.info(f"Added {len(expanded_neighbors)} expanded neighbors to the subgraph")
    else:
        # Original sequential processing code remains unchanged
        for node in tqdm(N_star_A, desc="Expanding subgraph to 2-hop neighbors"):
            second_hop_neighbors = kg.get_k_hop_neighbors(node, k=k, sample=False) - {A, B}

            for second_neighbor in tqdm(second_hop_neighbors, desc=f"Querying LLM for {node}'s neighbors"):
                if second_neighbor in N_star_A or second_neighbor == A:
                    continue  

                predicted_relations = query_llm_for_relations_by_logprob_copy(
                    model, tokenizer, node, second_neighbor, relation_descs, 
                    multi_gpu=multi_gpu, 
                    thresholds=thresholds, 
                    confidence_threshold=confidence_threshold
                )

                for rel, conf, score in predicted_relations:
                    expanded_neighbors.add(second_neighbor)
                    subgraph_triples.add((node, rel, second_neighbor, conf, score))
                    graph.add_edge(node, second_neighbor)

    # ***PHASE 3: Verify all nodes against B***
    verified_connections = set()
    valid_nodes_to_B = set()
    all_nodes_to_check = N_star_A.union(expanded_neighbors, {A}) - {B}

    logging.info(f"PHASE 3: Verifying {len(all_nodes_to_check)} nodes against {B}")

    if use_vllm and all_nodes_to_check and not use_entropy:
        # Batch process connections to B
        entity_pairs = [(node, B) for node in all_nodes_to_check]
        batch_results = query_llm_for_relations_by_judgement_batch(
            model, model_name, tokenizer, entity_pairs, relation_candidates, times=times,
            thresholds=thresholds, confidence_threshold=confidence_threshold, batch_size=batch_size
        )

        # Process batch results
        for (head, tail), relations in batch_results.items():
            for rel, conf, score in relations:
                verified_connections.add((head, rel, tail, conf, score))
                graph.add_edge(head, tail)
                valid_nodes_to_B.add(head)

        logging.info(f"Found {len(valid_nodes_to_B)} nodes with valid connections to {B}")

    elif use_vllm and all_nodes_to_check and use_entropy:
        # Batch process connections to B
        entity_pairs = [(node, B) for node in all_nodes_to_check]
        batch_results = query_llm_for_relations_by_judgement_entropy_batch(
            model, model_name, tokenizer, entity_pairs, relation_candidates, entropy_threshold=entropy_threshold, batch_size=batch_size
        )

        # Process batch results
        for (head, tail), relations in batch_results.items():
            for relation, entropy, yes_prob, no_prob, unknown_prob in relations:
                verified_connections.add((head, relation, tail, entropy, yes_prob, no_prob, unknown_prob))
                graph.add_edge(head, tail)
                valid_nodes_to_B.add(head)

        logging.info(f"Found {len(valid_nodes_to_B)} nodes with valid connections to {B}")

    else:
        # Original sequential processing
        for node in tqdm(all_nodes_to_check, desc="Querying LLM for final connections to B"):
            predicted_relations = query_llm_for_relations_by_logprob_copy(
                model, tokenizer, node, B, relation_descs, 
                multi_gpu=multi_gpu, 
                thresholds=thresholds, 
                confidence_threshold=confidence_threshold
            )

            for rel, conf, score in predicted_relations:
                verified_connections.add((node, rel, B, conf, score))
                graph.add_edge(node, B)
                valid_nodes_to_B.add(node)

    # ***PHASE 4: Remove Disconnected Paths*** (unchanged)
    logging.info("PHASE 4: Pruning disconnected paths")
    final_subgraph = set()

    # Reverse-check all paths (same for both vLLM and non-vLLM)
    if not use_entropy:
        for (head, rel, tail, conf, score) in subgraph_triples.union(verified_connections):
            if tail in valid_nodes_to_B or tail == B:  # Ensure the tail is also connected to B
                final_subgraph.add((head, rel, tail, conf, score))
                valid_nodes_to_B.add(head)  # Propagate in reverse to ensure the head is also retained
    else:
        for (head, rel, tail, entropy, yes_prob, no_prob, unknown_prob) in subgraph_triples.union(verified_connections):
            if tail in valid_nodes_to_B or tail == B:  # Ensure the tail is also connected to B
                final_subgraph.add((head, rel, tail, entropy, yes_prob, no_prob, unknown_prob))
                valid_nodes_to_B.add(head)  # Propagate in reverse to ensure the head is also retained

    logging.info(f"Subgraph construction completed. {len(final_subgraph)} triples retained forming paths from '{A}' to '{B}'.")

    return list(final_subgraph)



def batch_evaluate_triples(model, model_name, tokenizer, triples, times=5, batch_size=2000, thresholds=[0.2, 0.5, 0.8]):
    """
    Batch evaluate triples to get confidence scores in parallel.
    
    Args:
        model: The LLM model.
        tokenizer: The tokenizer for the model.
        triples: List of (head, relation, tail) tuples.
        times: Number of times to repeat for robust estimation.
        batch_size: Maximum number of prompts to process at once.
        use_vllm: Whether to use vLLM.
        logprobs_top_k: Top-k logprobs to retrieve per token.
        thresholds: Thresholds for determining confidence levels.
        
    Returns:
        List[int]: Confidence levels for each triple (0-3).
    """
    
    # Define the system message
    system_message = (
    "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown . "
    "First analyze the semantic properties of both entities, and then reason about whether the relation mentioned in the task is appropriate for these two entities.\n\n"
    "Here is an example of a correct relation:\n\n"
    "Example 1: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "'wrote' is one of the most specific relations between an author and their work, so the relation 'wrote' is correct.\n\n"
    "Here is an example of an incorrect relation:\n\n"
    "Example 2: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'locatedIn', reasoning process: "
    "Shakespeare is a person and Hamlet is a literary work. The relation 'locatedIn' typically describes spatial or geographic relationships, "
    "which does not apply to an author and their work. Therefore, the relation 'locatedIn' is incorrect.\n\n"
    "Here is an example of an unknown relation:\n\n"
    "Example 3: For head entity 'Hamlet', tail entity 'Existentialism', relation 'influencedBy', reasoning process: "
    "Hamlet is a literary work, while Existentialism is a philosophical movement. "
    "Although some scholars interpret Hamlet's introspective nature as proto-existentialist, there is no widely agreed-upon or factual relationship confirming that Hamlet was directly influenced by Existentialism. "
    "Therefore, the relation 'influencedBy' is unknown.\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'.")
    
    # Define template for prompts
    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",}
    
    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
    
    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0

    # Get token IDs for "Yes" and "No"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]    # Just get the first token
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()

    from collections import defaultdict
    # Initialize results container
    results = defaultdict(list)
    
    # Process triples in batches to avoid VRAM overflow
    for i in range(0, len(triples), batch_size // times):
        batch_triples = triples[i:i + batch_size // times]
        
        # Create prefixes for this batch of triples
        prefixes = []
        combo_indices = []
        
        for h, r, t in batch_triples:
            if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
            prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
            
            # Add this prompt 'times' times
            for _ in range(times):
                prefixes.append(prefix)
                combo_indices.append((h, r, t))
        
        # Make batch call to vLLM with all duplicated prompts
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            prefixes,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5,
        )
        
        # Process outputs
        for j, output in enumerate(batch_outputs):
            combo = combo_indices[j]
            logprobs_data = output[1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            
            yes_found = no_found = unknown_found = False
            yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
            
            # Store all top token probabilities
            top_tokens = []
            
            # First collect all top tokens
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    logprob = logprob_obj.logprob
                    prob = math.exp(logprob)
                    top_tokens.append((decoded, logprob, prob))
            
            # Apply temperature scaling for Llama models
            if use_temperature_scaling and top_tokens:
                # Scale probabilities with temperature
                scaled_probs = [prob ** (1.0 / scaling_temperature) for _, _, prob in top_tokens]
                
                # Normalize probabilities to sum to 1
                total_scaled_prob = sum(scaled_probs)
                if total_scaled_prob > 0:
                    normalized_scaled_probs = [p / total_scaled_prob for p in scaled_probs]
                else:
                    normalized_scaled_probs = [1.0 / len(scaled_probs)] * len(scaled_probs)
                
                # Update top_tokens with scaled probabilities and corresponding logprobs
                for token_idx in range(len(top_tokens)):
                    token = top_tokens[token_idx][0]
                    scaled_prob = normalized_scaled_probs[token_idx]
                    scaled_logprob = math.log(max(scaled_prob, 1e-40))
                    top_tokens[token_idx] = (token, scaled_logprob, scaled_prob)
            
            # Find Yes, No, and Unknown in top tokens (after scaling if applicable)
            for token, logprob, prob in top_tokens:
                if token == yes_decoded and not yes_found:
                    yes_logprob = logprob
                    yes_prob = prob
                    yes_found = True
                
                if token == no_decoded and not no_found:
                    no_logprob = logprob
                    no_prob = prob
                    no_found = True
                
                if token == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob
                    unknown_prob = prob
                    unknown_found = True
                
                # Break early if we found all three
                if yes_found and no_found and unknown_found:
                    break
            
            # Use very low probability if not found
            if not yes_found:
                yes_logprob = -100.0
                yes_prob = 1e-40
            
            if not no_found:
                no_logprob = -100.0
                no_prob = 1e-40
            
            if not unknown_found:
                unknown_logprob = -100.0
                unknown_prob = 1e-40
            
            # Store result
            results[combo].append({
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob)
            })
    
    # Calculate confidence for each triple
    confidence_levels = []
    for triple in triples:
        triple = tuple(triple)
        if triple not in results:
            # If we don't have results for this triple, give it a low confidence
            confidence_levels.append(0)
            continue
            
        triple_results = results[triple]
        
        # Calculate Yes probability statistics
        yes_probs = [run["Yes"][1] for run in triple_results]
        yes_prob_mean = np.mean(yes_probs)
        yes_prob_std = np.std(yes_probs)
        
        # Calculate raw score
        raw_score = yes_prob_mean * (1 - yes_prob_std)
        
        # Convert to confidence level using thresholds
        confidence = 0
        for k, threshold in enumerate(thresholds):
            if raw_score > threshold:
                confidence = k + 1
        
        confidence_levels.append(confidence)
    
    return confidence_levels

def batch_evaluate_triples_entropy(model, model_name, tokenizer, triples, batch_size=2000, entropy_threshold=0.1):
    """
    Batch evaluate triples to get entropy scores in parallel.
    
    Args:
        model: The LLM model.
        model_name: Name of the model being used.
        tokenizer: The tokenizer for the model.
        triples: List of (head, relation, tail) tuples.
        batch_size: Maximum number of prompts to process at once.
        entropy_threshold: Maximum entropy threshold for filtering results.
        
    Returns:
        List[float]: Entropy levels for each triple.
    """
    import math
    
    # Define the system message
    system_message = (
    "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown . "
    "First analyze the semantic properties of both entities, and then reason about whether the relation mentioned in the task is appropriate for these two entities.\n\n"
    "Here is an example of a correct relation:\n\n"
    "Example 1: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "'wrote' is one of the most specific relations between an author and their work, so the relation 'wrote' is correct.\n\n"
    "Here is an example of an incorrect relation:\n\n"
    "Example 2: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'locatedIn', reasoning process: "
    "Shakespeare is a person and Hamlet is a literary work. The relation 'locatedIn' typically describes spatial or geographic relationships, "
    "which does not apply to an author and their work. Therefore, the relation 'locatedIn' is incorrect.\n\n"
    "Here is an example of an unknown relation:\n\n"
    "Example 3: For head entity 'Hamlet', tail entity 'Existentialism', relation 'influencedBy', reasoning process: "
    "Hamlet is a literary work, while Existentialism is a philosophical movement. "
    "Although some scholars interpret Hamlet's introspective nature as proto-existentialist, there is no widely agreed-upon or factual relationship confirming that Hamlet was directly influenced by Existentialism. "
    "Therefore, the relation 'influencedBy' is unknown.\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'.")
    
    # Define template for prompts
    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",}

    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False

    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0
    
    # Get token IDs for "Yes", "No", and "Unknown"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()

    from collections import defaultdict
    # Initialize results container
    results = defaultdict(list)
    
    # Process triples in batches to avoid VRAM overflow
    for i in range(0, len(triples), batch_size):
        batch_triples = triples[i:i + batch_size]
     
        # Create all prompts and track their combinations
        prefixes = []  # All prompts to submit
        combo_indices = []  # Track which combination each prompt belongs to
        
        for h, r, t in batch_triples:
            if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
            prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
            
            prefixes.append(prefix)
            combo_indices.append((h, r, t))

        # Make a single batch call to vLLM with all prompts
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            prefixes,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5,
        )
        
        for j, output in enumerate(batch_outputs):
            combo = combo_indices[j]
            logprobs_data = output[1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            
            yes_found = no_found = unknown_found = False
            yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
            
            # Store all top token probabilities
            top_tokens = []
            
            # Find Yes, No, and Unknown probabilities and collect all top tokens
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    logprob = logprob_obj.logprob
                    prob = math.exp(logprob)
                    
                    # Add to top tokens list
                    top_tokens.append((decoded, logprob, prob))
            
            # Apply temperature scaling for Llama models
            if use_temperature_scaling and top_tokens:
                # Scale probabilities with temperature
                scaled_probs = [prob ** (1.0 / scaling_temperature) for _, _, prob in top_tokens]
                
                # Normalize probabilities to sum to 1
                total_scaled_prob = sum(scaled_probs)
                if total_scaled_prob > 0:
                    normalized_scaled_probs = [p / total_scaled_prob for p in scaled_probs]
                else:
                    normalized_scaled_probs = [1.0 / len(scaled_probs)] * len(scaled_probs)
                
                # Update top_tokens with scaled probabilities and corresponding logprobs
                for token_idx in range(len(top_tokens)):
                    token = top_tokens[token_idx][0]
                    scaled_prob = normalized_scaled_probs[token_idx]
                    scaled_logprob = math.log(max(scaled_prob, 1e-40))
                    top_tokens[token_idx] = (token, scaled_logprob, scaled_prob)
            
            # Find Yes, No, and Unknown in top tokens (after scaling if applicable)
            for token, logprob, prob in top_tokens:
                if token == yes_decoded and not yes_found:
                    yes_logprob = logprob
                    yes_prob = prob
                    yes_found = True
                
                if token == no_decoded and not no_found:
                    no_logprob = logprob
                    no_prob = prob
                    no_found = True

                if token == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob
                    unknown_prob = prob
                    unknown_found = True
            
            # Sort top tokens by probability (descending)
            top_tokens.sort(key=lambda x: x[2], reverse=True)
            
            # Use very low probability if not found in top-k
            if not yes_found:
                yes_logprob = -100.0
                yes_prob = 1e-40
            
            if not no_found:
                no_logprob = -100.0
                no_prob = 1e-40

            if not unknown_found:
                unknown_logprob = -100.0
                unknown_prob = 1e-40

            # Store result
            results[combo] = {
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob),
                "top_tokens": top_tokens
            }

    # Calculate entropy for each triple
    entropy_levels = []
    for triple in triples:
        triple = tuple(triple)
        if triple not in results:
            # If we don't have results for this triple, give it a very large entropy
            entropy_levels.append(10000)
            continue
            
        triple_result = results[triple]
        yes_prob = triple_result["Yes"][1]
        top_tokens = triple_result.get("top_tokens", [])
        
        # Check if yes_prob is the highest among top tokens
        if top_tokens and top_tokens[0][2] == yes_prob:
            # Calculate entropy using all top tokens
            total_prob = sum(token[2] for token in top_tokens)
            
            if total_prob < 1e-10:  # Avoid division by zero
                entropy = 10000
            else:
                # Normalize probabilities
                normalized_probs = [token[2] / total_prob for token in top_tokens]
                
                # Calculate entropy
                entropy = 0
                for norm_prob in normalized_probs:
                    if norm_prob > 1e-10:
                        entropy -= norm_prob * math.log2(norm_prob)
        else:
            entropy = 10000
            
        entropy_levels.append(entropy)
    
    return entropy_levels


def batch_score_triples_entropy(model, model_name, tokenizer, triples, batch_size=2000, entropy_threshold=0.1):
    """
    Batch score triples to get entropy scores in parallel.
    
    Args:
        model: The LLM model.
        model_name: Name of the model being used.
        tokenizer: The tokenizer for the model.
        triples: List of (head, relation, tail) tuples.
        batch_size: Maximum number of prompts to process at once.
        entropy_threshold: Maximum entropy threshold for filtering results.
        
    Returns:
        dict: Dictionary mapping each triple to its results, including probabilities and entropy.
    """
    import math
    
    # Define the system message
    system_message = (
    "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown . "
    "First analyze the semantic properties of both entities, and then reason about whether the relation mentioned in the task is appropriate for these two entities.\n\n"
    "Here is an example of a correct relation:\n\n"
    "Example 1: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "'wrote' is one of the most specific relations between an author and their work, so the relation 'wrote' is correct.\n\n"
    "Here is an example of an incorrect relation:\n\n"
    "Example 2: For head entity 'Shakespeare', tail entity 'Hamlet', relation 'locatedIn', reasoning process: "
    "Shakespeare is a person and Hamlet is a literary work. The relation 'locatedIn' typically describes spatial or geographic relationships, "
    "which does not apply to an author and their work. Therefore, the relation 'locatedIn' is incorrect.\n\n"
    "Here is an example of an unknown relation:\n\n"
    "Example 3: For head entity 'Hamlet', tail entity 'Existentialism', relation 'influencedBy', reasoning process: "
    "Hamlet is a literary work, while Existentialism is a philosophical movement. "
    "Although some scholars interpret Hamlet's introspective nature as proto-existentialist, there is no widely agreed-upon or factual relationship confirming that Hamlet was directly influenced by Existentialism. "
    "Therefore, the relation 'influencedBy' is unknown.\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'.")
    
    # Define template for prompts
    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",}

    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
    
    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0

    # Get token IDs for "Yes", "No", and "Unknown"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()

    from collections import defaultdict
    # Initialize results container
    results = defaultdict(dict)
    
    # Process triples in batches to avoid VRAM overflow
    for i in range(0, len(triples), batch_size):
        batch_triples = triples[i:i + batch_size]
     
        # Create all prompts and track their combinations
        prefixes = []  # All prompts to submit
        combo_indices = []  # Track which combination each prompt belongs to
        
        for h, r, t in batch_triples:
            if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
            prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
            
            prefixes.append(prefix)
            combo_indices.append((h, r, t))

        # Make a single batch call to vLLM with all prompts
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            prefixes,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5,
        )
        
        for j, output in enumerate(batch_outputs):
            combo = combo_indices[j]
            logprobs_data = output[1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            
            yes_found = no_found = unknown_found = False
            yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
            
            # Store all top token probabilities
            top_tokens = []
            
            # Find Yes, No, and Unknown probabilities and collect all top tokens
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    logprob = logprob_obj.logprob
                    prob = math.exp(logprob)
                    
                    # Add to top tokens list
                    top_tokens.append((decoded, logprob, prob))
            
            # Apply temperature scaling for Llama models
            if use_temperature_scaling and top_tokens:
                # Scale probabilities with temperature
                scaled_probs = [prob ** (1.0 / scaling_temperature) for _, _, prob in top_tokens]
                
                # Normalize probabilities to sum to 1
                total_scaled_prob = sum(scaled_probs)
                if total_scaled_prob > 0:
                    normalized_scaled_probs = [p / total_scaled_prob for p in scaled_probs]
                else:
                    normalized_scaled_probs = [1.0 / len(scaled_probs)] * len(scaled_probs)
                
                # Update top_tokens with scaled probabilities and corresponding logprobs
                for token_idx in range(len(top_tokens)):
                    token = top_tokens[token_idx][0]
                    scaled_prob = normalized_scaled_probs[token_idx]
                    scaled_logprob = math.log(max(scaled_prob, 1e-40))
                    top_tokens[token_idx] = (token, scaled_logprob, scaled_prob)
            
            # Find Yes, No, and Unknown in top tokens (after scaling if applicable)
            for token, logprob, prob in top_tokens:
                if token == yes_decoded and not yes_found:
                    yes_logprob = logprob
                    yes_prob = prob
                    yes_found = True
                
                if token == no_decoded and not no_found:
                    no_logprob = logprob
                    no_prob = prob
                    no_found = True

                if token == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob
                    unknown_prob = prob
                    unknown_found = True
                    
            # Sort top tokens by probability (descending)
            top_tokens.sort(key=lambda x: x[2], reverse=True)
            
            # Use very low probability if not found in top-k
            if not yes_found:
                yes_logprob = -100.0
                yes_prob = 1e-40
            
            if not no_found:
                no_logprob = -100.0
                no_prob = 1e-40

            if not unknown_found:
                unknown_logprob = -100.0
                unknown_prob = 1e-40

            # Store result
            results[combo] = {
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob),
                "top_tokens": top_tokens
            }

    # Calculate entropy for each triple and add to results
    for triple in triples:
        triple = tuple(triple)
        if triple not in results:
            # If we don't have results for this triple, use a placeholder
            results[triple] = {
                "Yes": (-100.0, 1e-40),
                "No": (-100.0, 1e-40),
                "Unknown": (-100.0, 1e-40),
                "Entropy": 10000
            }
            continue
            
        # Get probabilities
        yes_prob = results[triple]["Yes"][1]
        top_tokens = results[triple].get("top_tokens", [])
        
        # Check if yes_prob is the highest among top tokens
        if top_tokens and top_tokens[0][2] == yes_prob:
            # Calculate entropy using all top tokens
            total_prob = sum(token[2] for token in top_tokens)
            
            if total_prob < 1e-10:  # Avoid division by zero
                entropy = 10000
            else:
                # Normalize probabilities
                normalized_probs = [token[2] / total_prob for token in top_tokens]
                
                # Calculate entropy
                entropy = 0
                for norm_prob in normalized_probs:
                    if norm_prob > 1e-10:
                        entropy -= norm_prob * math.log2(norm_prob)
        else:
            entropy = 10000
        
        # Add entropy to results
        results[triple]["Entropy"] = entropy
    
    return results


def save_entropy_results_to_json(results, output_path):
    """
    Save triple entropy results to a JSON file.
    
    Args:
        results: Dictionary mapping triples to their evaluation results.
        output_path: Path to save the JSON file.
    """
    # Format results for JSON output
    formatted_results = {}
    for triple, result in results.items():
        # Format triple as string for JSON key
        triple_str = str(triple)
        
        # Extract values
        entropy = result["Entropy"]
        yes_prob = result["Yes"][1]
        no_prob = result["No"][1]
        unknown_prob = result["Unknown"][1]
        
        # Store as array following the requested format
        formatted_results[triple_str] = [entropy, yes_prob, no_prob, unknown_prob]
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(formatted_results, f, indent=2)


def save_subgraph(subgraph_triples, output_file):
    """
    Save the extracted subgraph to a file, including confidence scores.

    Args:
        subgraph_triples (list): List of triples (head, relation, tail, confidence).
        output_file (str): File path to save the subgraph.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for head, relation, tail, conf in subgraph_triples:
            f.write(f"{head},{relation},{tail},{conf}\n")  # Store confidence score

    logging.info(f"Saved subgraph to {output_file}.")


def estimate_llm_query_complexity(kg, target_triple, k, relation_num=None):
    """
    Estimate the number of LLM queries and the overall time complexity when constructing a subgraph.

    Args:
        kg (KnowledgeGraph): The loaded knowledge graph object.
        target_triple (tuple): The target triple (A, relation, B).
        k (int): The maximum hop distance for subgraph expansion (k-hop).

    Returns:
        tuple: (estimated_query_count, estimated_time_complexity)
            - estimated_query_count: Estimated total number of LLM queries in the worst case.
            - estimated_time_complexity: Estimated total time complexity = query count × number of relations.
    """
    A, _, B = target_triple

    # Step 1: Get the k-hop neighbors of A (excluding A and B)
    potential_neighbors = kg.get_k_hop_neighbors(A, k=k, sample=False) - {A, B}
    count1 = len(potential_neighbors)

    # Step 2: For each first-hop neighbor, get their k-hop neighbors (excluding A and B)
    count2 = 0
    for neighbor in potential_neighbors:
        second_hop_neighbors = kg.get_k_hop_neighbors(neighbor, k=k, sample=False) - {A, B}
        count2 += len(second_hop_neighbors)

    # Step 3: Estimate LLM query count for all candidates:
    # (N_star_A ∪ expanded_neighbors ∪ {A}) queried against B
    # Worst-case query count = count1 + count2 + 1 (A itself)
    estimated_query_count = 2 * (count1 + count2) + 1

    # Estimate time complexity based on number of relations
    if relation_num is None:
        relation_size = len(kg.relations)
    else:
        relation_size = relation_num
    estimated_time_complexity = estimated_query_count * relation_size

    return estimated_query_count, estimated_time_complexity

# Save the valid subgraph to a file
def save_valid_subgraph(subgraph_triples, target_triple, output_dir, index):
    """
    Save a valid subgraph with its target triple to a file.
    
    Args:
        subgraph_triples: List of tuples (subject, relation, object, confidence)
        target_triple: The target triple (subject, relation, object)
        output_dir: Directory to save the subgraph
        index: Index of the subgraph for naming
    
    Returns:
        str: Path to the saved subgraph file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Format filename
    filename = f"subgraph_{index}_{target_triple[0]}_{target_triple[1]}_{target_triple[2]}.txt"
    output_file = os.path.join(output_dir, filename)
    
    # Write subgraph to file
    with open(output_file, "w", encoding="utf-8") as f:
        # First line contains the target triple
        f.write(f"TARGET_TRIPLE:{target_triple[0]},{target_triple[1]},{target_triple[2]}\n")
        
        # Rest of the file contains the subgraph triples
        for s, r, o, c in subgraph_triples:
            f.write(f"{s},{r},{o},{c}\n")
    
    logging.info(f"Saved valid subgraph to {output_file}")
    return output_file

def is_valid_subgraph(subgraph, min_size=10):
    """
    Check if a subgraph is valid based on minimum size.
    
    Args:
        subgraph: List of tuples (subject, relation, object, confidence)
        min_size: Minimum number of triples required
        
    Returns:
        bool: True if valid, False otherwise
    """
    return len(subgraph) >= min_size

def load_existing_subgraphs(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_subgraph_to_json(json_path, triple_key, subgraph_triples):
    data = load_existing_subgraphs(json_path)
    data[str(triple_key)] = subgraph_triples
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def is_triple_already_processed(json_path, triple_key):
    if not os.path.exists(json_path):
        return False
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return str(triple_key) in data
