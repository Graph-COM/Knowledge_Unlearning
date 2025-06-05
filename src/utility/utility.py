import os
import json
import logging
import random
from tqdm import tqdm
from collections import defaultdict
import math
from src.models.model_loader import generate_text_batch

def sample_nontarget_triples(kg, unlearn_triples, sample_size, loc, loc_num):
    """
    Efficiently sample non-target triples from the KG with locality constraints.

    Args:
        kg: Loaded KnowledgeGraph object
        unlearn_triples: List of triples to be unlearned
        sample_size: Number of triples to sample
        loc: Whether to sample within local neighborhood
        loc_num: hop distance constraint

    Returns:
        List of sampled triples
    """
    logging.info(f"Sampling nontarget triples with loc={loc}, loc_num={loc_num}, size={sample_size}")

    # 1. Compute unlearn entity set
    unlearn_entities = set()
    for h, _, t in unlearn_triples:
        unlearn_entities.add(h)
        unlearn_entities.add(t)

    # 2. Get remaining triples (exclude unlearn ones)
    all_triples = set(kg.triples['total'])
    unlearn_set = set(unlearn_triples)
    remain_triples = list(all_triples - unlearn_set)
    random.shuffle(remain_triples)

    # 3. Build adj list if needed and compute k-hop neighborhood of unlearn_entities
    if kg.adj_list is None:
        kg.build_adjacency_list()

    all_khop_neighbors = set()
    for ent in unlearn_entities:
        all_khop_neighbors |= kg.get_k_hop_neighbors(ent, k=loc_num)

    # 4. Sample triples according to loc flag
    sampled = []
    for triple in tqdm(remain_triples, desc="Sampling utility triples"):
        h, _, t = triple
        in_local = (h in all_khop_neighbors) or (t in all_khop_neighbors)
        if (loc and in_local) or (not loc and not in_local):
            sampled.append(triple)
        if len(sampled) >= sample_size:
            break

    logging.info(f"Sampled {len(sampled)} triples for utility evaluation.")
    return sampled

def save_sampled_triples(triples, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([list(t) for t in triples], f, indent=2)
    logging.info(f"Saved sampled triples to {path}")

def load_sampled_triples(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [tuple(t) for t in data]

def batch_score_triples(model, model_name, tokenizer, triples, batch_size=2000):
    """
    Score triples using the model to answer Yes/No/Unknown in batch via vLLM.
    Returns logprobs and entropy scores.
    """
    logging.info("Scoring triples with vLLM batch inference...")

    system_message = (
        "You are an expert in knowledge graphs. Your task is to determine whether a given relation between two entities is correct, incorrect, or unknown. "
        "Your answer should be 'Answer: Yes', 'Answer: No', or 'Answer: Unknown'."
    )

    user_template = {
        "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
        "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown"
    }

    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
    
    # Set temperature for Llama models (used for temperature scaling)
    # Qwen models use default temperature (1.0)
    use_temperature_scaling = True if model_name in ["meta-llama/Llama-3.1-8B-Instruct"] else False
    scaling_temperature = 0.2452 if use_temperature_scaling else 1.0

    yes_token_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    unknown_token_id = tokenizer.encode(" Unknown", add_special_tokens=False)[0]

    yes_decoded = tokenizer.decode([yes_token_id]).strip()
    no_decoded = tokenizer.decode([no_token_id]).strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).strip()

    results = defaultdict(dict)

    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        prompts, indices = [], []

        for h, r, t in batch:
            if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)
            prefix = prompt.rsplit("Answer:", 1)[0] + "Answer:"
            prompts.append(prefix)
            indices.append((h, r, t))

        outputs = generate_text_batch(
            model,
            tokenizer,
            prompts,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=5
        )

        for j, output in enumerate(outputs):
            triple = indices[j]
            logprob_dict = output[1].get("logprobs", [{}])[0]

            top_tokens = []
            yes_found = no_found = unknown_found = False
            yes_prob = no_prob = unknown_prob = 1e-40
            yes_logprob = no_logprob = unknown_logprob = -100.0

            # First collect all top tokens
            for tok, logprob_obj in logprob_dict.items():
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

            total = yes_prob + no_prob + unknown_prob
            norm = lambda x: x / total if total > 0 else 1e-10
            entropy = -sum(norm(p) * math.log2(norm(p)) for p in [yes_prob, no_prob, unknown_prob])

            results[triple] = {
                "Yes": (yes_logprob, yes_prob),
                "No": (no_logprob, no_prob),
                "Unknown": (unknown_logprob, unknown_prob),
                "Entropy": entropy
            }

    return results

def save_scored_results_to_json(results, path):
    formatted = {}
    for triple, val in results.items():
        triple_str = str(triple)
        formatted[triple_str] = [val["Entropy"], val["Yes"][1], val["No"][1], val["Unknown"][1]]
    with open(path, 'w') as f:
        json.dump(formatted, f, indent=2)


def batch_score_utility_triples_entropy(model, model_name, tokenizer, triples, batch_size=2000, entropy_threshold=0.1):
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
        dict: Dictionary mapping each triple to its results, including top token probabilities and entropy.
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
            
            # Store result with top tokens (maximum 5)
            results[combo] = {
                "top_tokens": top_tokens[:5] if len(top_tokens) >= 5 else top_tokens + [("", -100.0, 1e-40)] * (5 - len(top_tokens))
            }

    # Calculate entropy for each triple and add to results
    for triple in triples:
        triple = tuple(triple)
        if triple not in results:
            # If we don't have results for this triple, use a placeholder
            empty_token = ("", -100.0, 1e-40)
            results[triple] = {
                "top_tokens": [empty_token] * 5,
                "Entropy": 10000
            }
            continue
            
        # Get top tokens
        top_tokens = results[triple]["top_tokens"]
        
        # Calculate entropy using all top tokens
        if top_tokens:
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


def save_utility_triples_results_to_json(results, output_path):
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
        top_tokens = result["top_tokens"]
        
        # Create entry with entropy and top 5 tokens and their probabilities
        entry = [entropy]
        for token, _, prob in top_tokens:
            entry.append(token)
            entry.append(prob)
        
        # Store the formatted result
        formatted_results[triple_str] = entry
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(formatted_results, f, indent=2)