"""
Threshold selection for Query LLM Knowledge and Construct Subgraph.

example:
python src/evaluation/threshold_selection.py --config config/config_threshold.yml
"""

import sys
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(24)
import re
import difflib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import yaml
import argparse
import json
from huggingface_hub import snapshot_download

# Append parent directories if necessary
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import KnowledgeGraph and the pre-implemented load_model with vLLM support
from src.data.knowledge_graph import KnowledgeGraph
from src.models.model_loader import load_model, generate_text, generate_text_batch
from src.utils.utils import plot_probability_sum_distribution

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Threshold selection for KG relation extraction")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--finetune", action="store_true", help="Whether to use instruct fine-tuning")
    return parser.parse_args()


def load_evaluation_samples(kg, num_samples=100, hard_examples=False):
    """
    Load evaluation samples from the KnowledgeGraph's threshold_selection_triples.

    Args:
        kg (KnowledgeGraph): The knowledge graph instance with precomputed threshold_selection_triples.
        num_samples (int): Number of samples per type.
        hard_examples (bool): Whether to use hard negative samples by replacing relations with similar ones.

    Returns:
        tuple: (positive_samples, negative_samples) where each sample is (entity1, relation, entity2).
    """
    logging.info("Loading evaluation samples from KnowledgeGraph threshold_selection_triples...")
    positive_samples = kg.threshold_selection_triples['positive'][:num_samples]

    if hard_examples:
        logging.info("Generating hard negative samples using relation_close_mapping...")
        import random
        negative_samples = []

        for h, r, t in positive_samples:
            if hasattr(kg, 'relation_close_mapping') and r in kg.relation_close_mapping and kg.relation_close_mapping[r]:
                similar_relations = kg.relation_close_mapping[r]
                chosen_r = random.choice(similar_relations)
                negative_samples.append((h, chosen_r, t))
            else:
                logging.warning(f"No similar relation found for: {r}, using fallback.")
                negative_samples.append((h, r, t))  # fallback: not a true negative

        # Clip to num_samples if too many
        negative_samples = negative_samples[:num_samples]
    else:
        negative_samples = kg.threshold_selection_triples['negative'][:num_samples]

    logging.info(f"Loaded {len(positive_samples)} positive samples and {len(negative_samples)} negative samples.")
    return positive_samples, negative_samples

class TemperatureScale(nn.Module):
    def __init__(self, temperature=None):
        """
        How to use the TemperatureScale class:
        temp_model = TemperatureScale()
        temp_model = temp_model.to(device)
        probs = LLM_func()
        # assume a three class prob tensor [[0(no), 1(yes), 2(unknown)]]
        scaled_probs = temp_model(probs)
        """
        super().__init__()
        if not temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            self.temperature = nn.Parameter(torch.tensor(temperature))


    def forward(self, probs):
        scaled_probs = probs.clamp(min=1e-12) 
        scaled_probs = scaled_probs ** (1.0 / self.temperature) 
        scaled_probs = scaled_probs / scaled_probs.sum(dim=1, keepdim=True)
        return scaled_probs


def compute_ece(probs, y_true, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) in PyTorch.

    Args:
        y_true (torch.Tensor): Ground truth labels. Shape (N,)
        probs (torch.Tensor): Model output probabilities. Shape (N, num_classes)
        n_bins (int): Number of bins for confidence partitioning.

    Returns:
        float: Expected Calibration Error (ECE).
    """
    # Predicted class and confidence
    print("ece check", probs.size())
    confidences, y_pred = torch.max(probs[:,:3], dim=1)

    # Re-check in case all were filtered
    if y_true.numel() == 0:
        return torch.tensor(float('nan'), device=probs.device)

    # Bin edges
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)

    ece = torch.zeros(1, device=probs.device)
    total_samples = y_true.size(0)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Find samples in bin
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = in_bin.sum().item()

        if bin_size > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = (y_pred[in_bin] == y_true[in_bin]).float().mean()
            ece += (bin_size / total_samples) * torch.abs(bin_accuracy - bin_confidence)

    return ece

def temp_scaling(model, tokenizer, positive_prob, negative_prob, times, device, system_message, user_template, use_chat_template, lower=0.2, upper=5.0, eps=1e-4):
    """
    Perform a grid search to calibrate the model
    """
    temp_model = TemperatureScale()
    temp_model = temp_model.to(device)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.1, max_iter=100)
    nll_criterion = nn.NLLLoss()

    probs = torch.cat([positive_prob, negative_prob])
    # log_probs = torch.log(all_probs.clamp(min=1e-12))
    labels = torch.cat([
        torch.ones(positive_prob.size(0), device=probs.device),
        torch.zeros(negative_prob.size(0), device=probs.device)
    ]).long()

    # Compute predicted labels
    probs = probs[:,:3]
    # print("check size", probs.size())
    preds = torch.argmax(probs, dim=1)

    def compute_nll():
        optimizer.zero_grad()
        scaled_probs = temp_model(probs)
        log_probs = scaled_probs.clamp(min=1e-12).log()
        loss = nll_criterion(log_probs, labels)
        ece = compute_ece(scaled_probs.clamp(min=1e-12), labels)
        print(f"Loss: {loss.item()}, ECE: {ece.item()}")
        loss.backward()
        return loss

    optimizer.step(compute_nll)
    logging.info(f"best temperature: {temp_model.temperature}")
    return temp_model


# Global templates for perplexity-based scoring
PERPLEXITY_TEMPLATES = [
    "The head entity '{entity1}' is {relation} the tail entity '{entity2}'.",
    "From '{entity1}' to '{entity2}', the directed relationship is: {relation}.",
    "Considering '{entity1}' as the head and '{entity2}' as the tail, their relationship is: {relation}.",
    "'{entity1}' (head) connects to '{entity2}' (tail) via {relation}.",
    "If '{entity1}' is the head and '{entity2}' is the tail, then the relationship is {relation}.",
    "In a directed sense, '{entity1}' is linked to '{entity2}' by {relation}.",
    "The head entity '{entity1}' and tail entity '{entity2}' share a {relation} relationship.",
    "Considering directionality, '{entity1}' (head) is {relation} to '{entity2}' (tail).",
    "Directed relation: '{entity1}' (head) is {relation} to '{entity2}' (tail).",
    "For the ordered pair with head '{entity1}' and tail '{entity2}', the relation is: {relation}."
]

# Global messages for log probability–based scoring.
LOGPROB_SYSTEM_MESSAGE = (
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

LOGPROB_USER_TEMPLATES = [
    # "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', what is the directed relationship? Answer: {relation}",
    # "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), determine the relation. Answer: {relation}",
    # "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'. Answer: {relation}",
    # "Task: Considering '{entity1}' as head and '{entity2}' as tail, what directed connection exists between them? Answer: {relation}",
    # "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Answer: {relation}",
    # "Task: What is the direct relationship from '{entity1}' to '{entity2}'? Answer: {relation}",
    # "Task: Identify the directed relation from '{entity1}' to '{entity2}'. Answer: {relation}",
    # "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner. Answer: {relation}",
    # "Task: Specify the relationship that links '{entity1}' to '{entity2}'. Answer: {relation}",
    # "Task: '{entity1}' is connected to '{entity2}' via what relation? Answer: {relation}",
    # "Task: Find the specific connection from '{entity1}' to '{entity2}'. Answer: {relation}",
    # "Task: In a knowledge graph, if '{entity1}' is the source node and '{entity2}' is the target node, what edge label connects them? Answer: {relation}",
    # "Task: A knowledge graph contains entities '{entity1}' and '{entity2}'. What is the directed relationship from the first to the second? Answer: {relation}",
    # "Task: When representing knowledge as a graph, how would you label the edge pointing from '{entity1}' to '{entity2}'? Answer: {relation}",
    # "Task: In the direction from '{entity1}' towards '{entity2}', the relationship type is what? Answer: {relation}",
    # "Task: Starting at '{entity1}' and following the graph edge to '{entity2}', how would you classify this connection? Answer: {relation}",
    # "Task: The directional relationship '{entity1}' → '{entity2}' is best described as what? Answer: {relation}",
    # "Task: As a knowledge graph expert, classify the edge type from node '{entity1}' to node '{entity2}'. Answer: {relation}",
    # "Task: In semantic network terminology, the predicate linking subject '{entity1}' to object '{entity2}' is what? Answer: {relation}",
    # "Task: Following RDF triple format, if '{entity1}' is the subject and '{entity2}' is the object, what is the predicate? Answer: {relation}",
    # "Task: I have two entities: '{entity1}' and '{entity2}'. What directed relationship exists between them, with the first being the head? Answer: {relation}",
    # "Task: Entity pair analysis: '{entity1}' → ? → '{entity2}'. Fill in the missing relation. Answer: {relation}",
    # "Task: Complete the knowledge triple: ({entity1}, ?, {entity2}) where ? represents the relation type. Answer: {relation}",
    # "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', first analyze their potential connection carefully. Consider their semantic categories, common contexts, and potential relationships. Then determine the most appropriate directed relationship. Answer: {relation}",
    # "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), please think step by step. First, categorize both entities. Next, identify possible domains where they interact. Finally, determine the specific relation between them. Answer: {relation}",
    # "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'. First consider the nature of '{entity1}' (e.g., person, place, concept), then the nature of '{entity2}', and analyze what logical connection could exist between such entities. Answer: {relation}",
    # "Task: Considering '{entity1}' as head and '{entity2}' as tail, carefully analyze their semantic types. Then identify any factual or conceptual connections by considering encyclopedic knowledge. Based on your analysis, what directed connection exists between them? Answer: {relation}",
    "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). First, consider what type of entity each one is. Second, think about possible ways these entities could be connected in the real world. Finally, specify the most precise relationship. Answer: {relation}",
    # "Task: What is the direct relationship from '{entity1}' to '{entity2}'? Before answering, analyze: (1) the category of each entity, (2) their possible interactions based on real-world knowledge, and (3) the most specific way to describe their connection. Answer: {relation}",
    # "Task: Identify the directed relation from '{entity1}' to '{entity2}'. Consider the following: Are they in the same domain? What is the hierarchical relationship between them? How specifically are they connected in a knowledge graph context? Answer: {relation}",
    # "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner. First, list potential categories for each entity. Then, evaluate possible relationships based on these categories. Finally, select the most appropriate relation. Answer: {relation}",
    # "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Let's reason step by step:\n1) Entity types: What type of entity is '{entity1}'? What type of entity is '{entity2}'?\n2) Possible connections: Given their types, what are possible ways they could be related?\n3) Most specific relation: Among these possibilities, which one most precisely captures their relationship?\nAnswer: {relation}"
    # #    "Task: Given head entity '{entity1}' (type: {type1}) and tail entity '{entity2}' (type: {type2}), determine their relationship. Answer: {relation}"
    # "Task: Suppose there is an association from '{entity1}' to '{entity2}'. Describe that association. Answer: {relation}",
    # "Task: We have two concepts: '{entity1}' and '{entity2}'. In the knowledge graph, they are connected by what property? Answer: {relation}",
    # "Task: Considering '{entity1}' and '{entity2}', which property captures how the former is linked to the latter? Answer: {relation}",
    # "Task: Based on your knowledge, how is '{entity1}' related to '{entity2}'? Answer: {relation}",
    # "Task: If we were to label the relationship between '{entity1}' and '{entity2}' in a triple, what would it be? Answer: {relation}",
    # "Task: Provide the term that logically connects '{entity1}' to '{entity2}' in a knowledge graph. Answer: {relation}",
    # "Task: In an ontology, which predicate associates '{entity1}' with '{entity2}'? Answer: {relation}",
    # "Task: Given '{entity1}' and '{entity2}', identify the relevant link that ties them together. Answer: {relation}",
    # "Task: In a semantic web setting, how would you define the bond between '{entity1}' and '{entity2}'? Answer: {relation}",
    # "Task: For '{entity1}' referring to '{entity2}', which label best expresses their connection? Answer: {relation}"
]

# Global messages for log probability–based scoring.
LOGPROB_SYSTEM_MESSAGE_EXPLICIT = (
    "You are an expert in knowledge graphs. Your task is to evaluate a candidate directed relation between two entities. "
    "First analyze the semantic properties of both entities and reason about potential connections before providing your final answer. "
    "Here are two examples of good reasoning:\n\n"
    "Example 1: For head entity 'Paris' and tail entity 'France', reasoning process: "
    "Paris is a city and France is a country. Geographically, cities are located within countries. "
    "Therefore, the relation is: locatedIn\n\n"
    "Example 2: For head entity 'Shakespeare' and tail entity 'Hamlet', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "The most specific relation between an author and their work is: wrote\n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. Note that reasoning part do not generate to many, only two or three sentences are enough."
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "Your reply must follow a required format: at the end of your response, you must highlight the answer in the form 'Answer: relation', where 'relation' is the correct answer you believe. No further output is allowed after 'Answer: relation'. A standard reply is as follows:"
    "Paris is a city and France is a country. Geographically, cities are located within countries."  
    "Therefore, the relation is: locatedIn. Answer: locatedIn"  
)

LOGPROB_USER_TEMPLATES_EXPLICIT = [
    # "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', what is the directed relationship?",
    # "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), determine the relation.",
    # "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'.",
    # "Task: Considering '{entity1}' as head and '{entity2}' as tail, what directed connection exists between them?",
    # "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail).",
    # "Task: What is the direct relationship from '{entity1}' to '{entity2}'?",
    # "Task: Identify the directed relation from '{entity1}' to '{entity2}'.",
    # "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner.",
    # "Task: Specify the relationship that links '{entity1}' to '{entity2}'.",
    # "Task: '{entity1}' is connected to '{entity2}' via what relation?",
    # "Task: Find the specific connection from '{entity1}' to '{entity2}'.",
    # "Task: In a knowledge graph, if '{entity1}' is the source node and '{entity2}' is the target node, what edge label connects them?",
    # "Task: A knowledge graph contains entities '{entity1}' and '{entity2}'. What is the directed relationship from the first to the second?",
    # "Task: When representing knowledge as a graph, how would you label the edge pointing from '{entity1}' to '{entity2}'?",
    # "Task: In the direction from '{entity1}' towards '{entity2}', the relationship type is what?",
    # "Task: Starting at '{entity1}' and following the graph edge to '{entity2}', how would you classify this connection?",
    # "Task: The directional relationship '{entity1}' → '{entity2}' is best described as what?",
    # "Task: As a knowledge graph expert, classify the edge type from node '{entity1}' to node '{entity2}'.",
    # "Task: In semantic network terminology, the predicate linking subject '{entity1}' to object '{entity2}' is what?",
    # "Task: Following RDF triple format, if '{entity1}' is the subject and '{entity2}' is the object, what is the predicate?",
    # "Task: I have two entities: '{entity1}' and '{entity2}'. What directed relationship exists between them, with the first being the head?",
    # "Task: Entity pair analysis: '{entity1}' → ? → '{entity2}'. Fill in the missing relation.",
    # "Task: Complete the knowledge triple: ({entity1}, ?, {entity2}) where ? represents the relation type.",
    # "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', first analyze their potential connection carefully. Consider their semantic categories, common contexts, and potential relationships. Then determine the most appropriate directed relationship.",
    # "Task: For the directed pair (head: '{entity1}', tail: '{entity2}'), please think step by step. First, categorize both entities. Next, identify possible domains where they interact. Finally, determine the specific relation between them.",
    # "Task: Identify the relationship from head entity '{entity1}' to tail entity '{entity2}'. First consider the nature of '{entity1}' (e.g., person, place, concept), then the nature of '{entity2}', and analyze what logical connection could exist between such entities.",
    # "Task: Considering '{entity1}' as head and '{entity2}' as tail, carefully analyze their semantic types. Then identify any factual or conceptual connections by considering encyclopedic knowledge. Based on your analysis, what directed connection exists between them?",
    "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). First, consider what type of entity each one is. Second, think about possible ways these entities could be connected in the real world. Finally, specify the most precise relationship.",
    "Task: What is the direct relationship from '{entity1}' to '{entity2}'? Before answering, analyze: (1) the category of each entity, (2) their possible interactions based on real-world knowledge, and (3) the most specific way to describe their connection.",
    # "Task: Identify the directed relation from '{entity1}' to '{entity2}'. Consider the following: Are they in the same domain? What is the hierarchical relationship between them? How specifically are they connected in a knowledge graph context?",
    # "Task: Determine how '{entity1}' relates to '{entity2}' in a directed manner. First, list potential categories for each entity. Then, evaluate possible relationships based on these categories. Finally, select the most appropriate relation.",
    # "Task: Determine the directed relation between '{entity1}' (head) and '{entity2}' (tail). Let's reason step by step:\n1) Entity types: What type of entity is '{entity1}'? What type of entity is '{entity2}'?\n2) Possible connections: Given their types, what are possible ways they could be related?\n3) Most specific relation: Among these possibilities, which one most precisely captures their relationship?"
    # #    "Task: Given head entity '{entity1}' (type: {type1}) and tail entity '{entity2}' (type: {type2}), determine their relationship."
    # "Task: Suppose there is an association from '{entity1}' to '{entity2}'. Describe that association.",
    # "Task: We have two concepts: '{entity1}' and '{entity2}'. In the knowledge graph, they are connected by what property?",
    # "Task: Considering '{entity1}' and '{entity2}', which property captures how the former is linked to the latter?",
    # "Task: Based on your knowledge, how is '{entity1}' related to '{entity2}'?",
    # "Task: If we were to label the relationship between '{entity1}' and '{entity2}' in a triple, what would it be?",
    # "Task: Provide the term that logically connects '{entity1}' to '{entity2}' in a knowledge graph.",
    # "Task: In an ontology, which predicate associates '{entity1}' with '{entity2}'?",
    # "Task: Given '{entity1}' and '{entity2}', identify the relevant link that ties them together.",
    # "Task: In a semantic web setting, how would you define the bond between '{entity1}' and '{entity2}'?",
    # "Task: For '{entity1}' referring to '{entity2}', which label best expresses their connection?"
]

LLM_SCORE_SYSTEM_MESSAGE = (
    "You are an expert in knowledge graphs. Your task is to analyze potential relationships "
    "between two entities based on a given known relationship. "
    "The relation defines a directional link from entity1 to entity2. "
    "You should evaluate whether this relationship is logically sound and factually accurate based on your knowledge. "
    "Do not overinterpret the meaning of the relation. "
    "Please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. "
    "Your response should be structured, precise, limited to 2-3 sentences, and strictly follow the required format."
)

LLM_SCORE_USER_TEMPLATES = (
    "Your response must be formatted exactly as follows:\n"
    "Evaluation evidence: \"{entity1} is <description1>, and {entity2} is <description2>. <reasoning>. Therefore, <conclusion>.\"\n"
    "RELATIONSHIP: {relation}; CONFIDENCE: <score>\n"
    "### Task:\n"
    "Determine the confidence for the potential relationship '{relation}' between:\n"
    "- Entity 1: '{entity1}'\n"
    "- Entity 2: '{entity2}'\n\n"
    "### Confidence score:\n"
    "- Not meaningful\n"
    "- Plausible\n"
    "- Definite\n"
    "- I don't know\n"
)

LLM_SCORE_RAW_RATING = ["Not meaningful", "I don't know", "Plausible", "Definite"]


JUDGE_SYSTEM_MESSAGE = (
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
    "According to the user's task, you should provide your final answer in the format 'Answer: Yes' or 'Answer: No' or 'Answer: Unknown'. "
)


JUDGE_USER_TEMPLATES = {
    #"Task: Determine whether the directed relation {relation} between {entity1} (head entity) and {entity2} (tail entity) is correct. First, consider the type of each entity. Second, think about how these entities might be connected in the real world. Finally, state whether {relation} is the correct directed relation between the two entities. Answer: Yes/No",
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.2-3B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.2-1B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.1-8B": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",
    "meta-llama/Llama-3.2-3B": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer: Yes/No/Unknown",
}


TF_LOGPROB_SYSTEM_MESSAGE = (
    "You are an expert in knowledge graphs. Your task is to evaluate a candidate directed relation between two entities. "
    "First analyze the semantic properties of both entities and reason about potential connections before providing your final answer. "
    "Here are two examples of good reasoning:\n\n"
    "Example 1: For head entity 'Paris' and tail entity 'France', relation: 'locatedIn', reasoning process: "
    "Paris is a city and France is a country. Geographically, cities are located within countries. "
    "Therefore, 'Paris' 'locatedIn' 'France'. "
    "Answer: True \n\n"
    "Example 2: For head entity 'Shakespeare' and tail entity 'Hamlet', relation: 'wrote', reasoning process: "
    "Shakespeare is a person, specifically an author, while Hamlet is a literary work. "
    "Therefore, 'Shakespeare' 'wrote' 'Hamlet'. "
    "Answer: True \n\n"
    # "You should consider the direction of the relation."
    # "Example 2: For head entity 'France' and tail entity 'Paris', relation: 'locatedIn', reasoning process:  "
    # "Paris is a city and France is a country. Logically, countries are not located within cities. "
    # "Therefore, 'France' 'locatedIn' 'Paris'. "
    # "Answer: False \n\n"
    "Be deliberate and analytical in your reasoning before providing your final answer. "
    "Your answer (which is provided) should be taken as-is; the goal is to compute its log probability given the context."
    "Your reply must follow a required format: at the end of your response, you must highlight the answer in the form 'Answer: True' or 'Answer: False'."
)

# TF_LOGPROB_USER_TEMPLATES = (
#     "Task: Determine whether the directed relation '{relation}' holds from '{entity1}' (head) to '{entity2}' (tail). First, consider what type of entity each one is. Second, think about possible ways these entities could be connected in the real world. Finally, decide whether the relation '{relation}' truly exists from '{entity1}' to '{entity2}'. Answer: Yes or No."
# )

TF_LOGPROB_USER_TEMPLATES = (
    "Task: Determine whether the directed relation '{relation}' holds from '{entity1}' (head) to '{entity2}' (tail). First, consider what '{entity1}' and '{entity2}' are. Reasoning in at most two sentences."
)

def check_scores_cache(cache_path, method, kg_name, model_name, num_samples):
    """
    Check if scores cache exists at the specified path.

    Args:
        cache_path (str): Path to the cache directory.
        method (str): "perplexity" or "logprob".
        kg_name (str): Name of the knowledge graph.
        model_name (str): Name of the model.
        num_samples (int): Number of samples used.

    Returns:
        tuple or None: (positive_scores, negative_scores, positive_samples, negative_samples) if cache exists, 
                      None otherwise.
    """
    if not cache_path:
        return None

    # Create a filename that includes method, kg, model, and sample size to ensure cache validity
    filename = f"{method}_{num_samples}_scores.json"
    cache_path = os.path.join(cache_path, f"{kg_name}_{model_name}")
    cache_file = os.path.join(cache_path, filename)

    if os.path.exists(cache_file):
        logging.info(f"Found cached scores at {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Ensure the cache contains all required data
            required_keys = ["positive_scores", "negative_scores", "positive_samples", "negative_samples"]
            if all(key in cache_data for key in required_keys):
                logging.info("Successfully loaded scores from cache")
                return (
                    cache_data["positive_scores"],
                    cache_data["negative_scores"],
                    cache_data["positive_samples"],
                    cache_data["negative_samples"]
                )
            else:
                logging.warning("Cache file is missing required data, recalculating scores")
                return None
        except Exception as e:
            logging.warning(f"Error loading cache: {e}, recalculating scores")
            return None
    else:
        logging.info(f"No cache found at {cache_file}, will compute scores")
        return None


def save_scores_cache(cache_path, method, kg_name, model_name, num_samples, 
                     positive_scores, negative_scores, positive_samples, negative_samples):
    """
    Save computed scores to cache.

    Args:
        cache_path (str): Path to the cache directory.
        method (str): "perplexity" or "logprob".
        kg_name (str): Name of the knowledge graph.
        model_name (str): Name of the model.
        num_samples (int): Number of samples used.
        positive_scores (list): List of scores for positive samples.
        negative_scores (list): List of scores for negative samples.
        positive_samples (list): List of positive triple samples.
        negative_samples (list): List of negative triple samples.
    """
    if not cache_path:
        return

    # Ensure cache directory exists
    cache_path = os.path.join(cache_path, f"{kg_name}_{model_name}")
    os.makedirs(cache_path, exist_ok=True)

    # Create a filename that includes method, kg, model, and sample size
    filename = f"{method}_{num_samples}_scores.json"
    cache_file = os.path.join(cache_path, filename)

    # Convert potential tuples in samples to lists for JSON serialization
    pos_samples_json = [list(triple) for triple in positive_samples]
    neg_samples_json = [list(triple) for triple in negative_samples]

    cache_data = {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "positive_samples": pos_samples_json,
        "negative_samples": neg_samples_json,
        "method": method,
        "kg_name": kg_name,
        "model_name": model_name,
        "num_samples": num_samples
    }

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved scores to cache at {cache_file}")
    except Exception as e:
        logging.warning(f"Error saving cache: {e}")

def save_raw_rating_cache(cache_path, method, kg_name, model_name, num_samples, 
                    positive_raw_rating, negative_raw_rating, positive_samples, negative_samples):
    """
    Save the LLM rating to a csv file.

    Args:
        cache_path (str): Path to the cache directory.
        method (str): "perplexity" or "logprob".
        kg_name (str): Name of the knowledge graph.
        model_name (str): Name of the model.
        num_samples (int): Number of samples used.
        positive_raw_rating (list): List of LLM raw rating for positive samples.
        negative_raw_rating (list): List of LLM raw rating for negative samples.
        positive_samples (list): List of positive triple samples.
        negative_samples (list): List of negative triple samples.
    """
    if not cache_path:
        return

    # Ensure cache directory exists
    cache_path = os.path.join(cache_path, f"{kg_name}_{model_name}")
    os.makedirs(cache_path, exist_ok=True)

    # Create a filename that includes method, kg, model, and sample size
    filename = f"{method}_{num_samples}_raw_rating.csv"
    cache_file = os.path.join(cache_path, filename)

    # Save the triples and confidence scores to the output file
    with open(cache_file, "w", encoding="utf-8") as f:
        for (head, relation, tail), conf in zip(positive_samples+negative_samples, positive_raw_rating+negative_raw_rating):
            f.write(f"{head},{relation},{tail},{conf}\n")  # Store confidence score
    logging.info(f"Saved llm raw rating at {cache_file}.")

def compute_perplexity_metric(model, tokenizer, triple, device, use_vllm=False):
    """
    Compute the minimum perplexity score for a given triple (entity1, relation, entity2)
    using multiple probing templates.
    
    Args:
        model: HuggingFace model or vLLM LLM instance.
        tokenizer: Corresponding tokenizer.
        triple (tuple): A triple (entity1, relation, entity2).
        device: Torch device.
        use_vllm (bool): Whether to use vLLM for computations.
        
    Returns:
        float: The minimum perplexity score among all templates.
    """
    entity1, relation, entity2 = triple
    perplexities = []
    
    # Use same templates for both vLLM and non-vLLM to ensure consistency
    for template in PERPLEXITY_TEMPLATES:
        prompt_text = template.format(entity1=entity1, entity2=entity2, relation=relation)
        
        if use_vllm:
            # For vLLM, use generate_text with return_logprobs=True
            try:
                # Generate with logprobs to compute perplexity
                _, logprobs_data = generate_text(
                    model, 
                    tokenizer, 
                    prompt_text, 
                    max_tokens=0,  # Don't generate new tokens, just get logprobs for the prompt
                    temperature=1.0,  # Use deterministic sampling for perplexity calculation
                    top_p=1.0,  # No nucleus sampling for perplexity calculation
                    use_vllm=True,
                    return_logprobs=True,
                    logprobs_top_k=5  # vLLM maximum allowed value
                )
                
                # Extract logprobs if available
                valid_logprobs = []
                
                # Check the correct key in logprobs_data
                if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
                    # Iterate through the logprobs entries (each corresponds to a token)
                    for logprobs_entry in logprobs_data['logprobs']:
                        # Check if we have any entries in this dictionary
                        if logprobs_entry:
                            # Get the highest-ranked logprob (the generated token)
                            best_logprob = None
                            
                            # Find the logprob with rank=1 (highest probability)
                            for token_id, logprob_obj in logprobs_entry.items():
                                if hasattr(logprob_obj, 'rank') and logprob_obj.rank == 1:
                                    if hasattr(logprob_obj, 'logprob'):
                                        best_logprob = logprob_obj.logprob
                                        break
                                # If it's a simple value (not an object)
                                elif isinstance(logprob_obj, (int, float)):
                                    best_logprob = logprob_obj
                                    break
                            
                            if best_logprob is not None:
                                valid_logprobs.append(best_logprob)
                
                # If we have valid logprobs, calculate perplexity
                if valid_logprobs:
                    avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
                    ppl = math.exp(-avg_logprob)
                    perplexities.append(ppl)
                    logging.info(f"Template perplexity: {ppl}")
                
            except Exception as e:
                logging.warning(f"Error computing vLLM perplexity: {e}")
                continue
        else:
            # Original perplexity computation for regular models
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = model(inputs, labels=inputs["input_ids"])
                loss = outputs.loss  # average cross-entropy loss per token
                ppl = math.exp(loss.item())  # perplexity = exp(loss)
                perplexities.append(ppl)
    
    # Use the minimum perplexity among all templates
    min_perplexity = min(perplexities) if perplexities else float("inf")
    return min_perplexity

def compute_logprob_metric(model, tokenizer, triple, device, use_vllm=False, explicit=False):
    """
    Compute the maximum log probability score for a given triple using multiple chat-format prompts.
    Supports both traditional models and vLLM with optional explicit reasoning mode.

    Args:
        model: HuggingFace model or vLLM LLM instance.
        tokenizer: Corresponding tokenizer.
        triple (tuple): A triple (entity1, relation, entity2).
        device: Torch device.
        use_vllm (bool): Whether to use vLLM for computation.
        explicit (bool): When True and use_vllm=True, makes the model generate reasoning 
                         before providing the answer.

    Returns:
        float: The maximum log probability across all templates.
    """
    entity1, relation, entity2 = triple
    log_probs = []

    # For non-vLLM or vLLM without explicit reasoning, use standard templates
    if not use_vllm or (use_vllm and not explicit):
        system_message = LOGPROB_SYSTEM_MESSAGE
        user_templates = LOGPROB_USER_TEMPLATES
    else:
        # For vLLM with explicit reasoning, use the explicit templates
        system_message = LOGPROB_SYSTEM_MESSAGE_EXPLICIT
        user_templates = LOGPROB_USER_TEMPLATES_EXPLICIT

    # Use the templates based on the mode
    for user_template in user_templates:
        if use_vllm and explicit:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_template.format(entity1=entity1, entity2=entity2)}
            ]
        else:
            messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_template.format(entity1=entity1, entity2=entity2, relation=relation)}
        ]

        if not use_vllm:
            # Original logprob computation for regular models - KEEP UNCHANGED
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)

            # Decode the full prompt to locate the "Answer:" marker.
            full_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if "Answer:" not in full_prompt:
                continue

            # The answer portion starts immediately after "Answer:".
            question_part = full_prompt.split("Answer:", 1)[0] + "Answer:"
            question_tokens = tokenizer(question_part, add_special_tokens=False)["input_ids"]
            question_length = len(question_tokens)

            # Create labels where tokens corresponding to the question part are masked out.
            labels = input_ids.clone()
            labels[:, :question_length] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss  # loss computed only on the answer tokens
                log_prob = -loss.item()  # convert negative loss to log probability
                log_probs.append(log_prob)
        elif use_vllm and not explicit:
            # Token-by-token computation for vLLM - follow similar workflow as non-vLLM
            try:
                # First create the token sequence just like in non-vLLM mode
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)

                # Decode the full prompt to locate the "Answer:" marker - same as non-vLLM workflow
                full_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)

                if "Answer:" not in full_prompt:
                    continue

                # The answer portion starts immediately after "Answer:"
                question_part = full_prompt.split("Answer:", 1)[0] + "Answer:"
                answer_part = full_prompt.split("Answer:", 1)[1].strip()

                # Verify if our relation appears in the answer part
                if relation not in answer_part:
                    continue

                # Token-by-token calculation of log probabilities
                token_log_probs = []

                # Start with the prefix up to "Answer:"
                current_prefix = question_part

                # Tokenize the relation part correctly (with space as it would appear after "Answer:")
                relation_with_space = " " + relation.strip()
                relation_token_ids = tokenizer.encode(relation_with_space, add_special_tokens=False)
                relation_tokens = [tokenizer.decode([token_id]) for token_id in relation_token_ids]

                logging.info(f"Computing logprobs for relation '{relation}' with {len(relation_token_ids)} tokens")

                # Calculate log probabilities for each token of the relation
                for i, token_id in enumerate(relation_token_ids):
                    token_text = relation_tokens[i]

                    # Generate the next token with logprobs
                    try:
                        _, logprobs_data = generate_text(
                            model,
                            tokenizer,
                            current_prefix,
                            max_tokens=1,
                            temperature=1.0,  # Use deterministic generation
                            top_p=1.0,
                            use_vllm=True,
                            return_logprobs=True,
                            logprobs_top_k=100  # vLLM maximum allowed value
                        )

                        # Find our token by comparing token IDs in the logprobs data
                        token_logprob = None

                        # Check if logprobs is available
                        if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
                            logprobs_dict = logprobs_data['logprobs'][0]  # First token's logprobs

                            # Check if our token_id is directly in the dictionary keys
                            if token_id in logprobs_dict:
                                # Extract the logprob value
                                logprob_obj = logprobs_dict[token_id]
                                if hasattr(logprob_obj, 'logprob'):
                                    token_logprob = logprob_obj.logprob
                                elif isinstance(logprob_obj, (int, float)):
                                    token_logprob = logprob_obj
                            else:
                                # Try matching by decoded token
                                for id_key, logprob_obj in logprobs_dict.items():
                                    if hasattr(logprob_obj, 'decoded_token'):
                                        # Compare normalized versions to handle whitespace differences
                                        decoded = logprob_obj.decoded_token
                                        if decoded.replace('Ġ', ' ').strip() == token_text.strip():
                                            token_logprob = logprob_obj.logprob
                                            break

                        if token_logprob is not None:
                            token_log_probs.append(token_logprob)
                            logging.info(f"Token '{token_text}' (ID: {token_id}) logprob: {token_logprob}")
                        else:
                            # If token not found, try to get the highest probability token as fallback
                            if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
                                logprobs_dict = logprobs_data['logprobs'][0]
                                highest_logprob = None

                                # Find the highest ranked logprob
                                for id_key, logprob_obj in logprobs_dict.items():
                                    if hasattr(logprob_obj, 'rank') and logprob_obj.rank == 1:
                                        highest_logprob = logprob_obj.logprob
                                        break

                                if highest_logprob is not None:
                                    # Apply a penalty factor to the highest logprob
                                    adjusted_logprob = highest_logprob - 100.0  # Significant penalty
                                    token_log_probs.append(adjusted_logprob)
                                    logging.warning(f"Token '{token_text}' not found, using penalized best logprob: {adjusted_logprob}")
                                else:
                                    # No highest logprob found, use fixed penalty
                                    penalty = -200.0
                                    token_log_probs.append(penalty)
                                    logging.warning(f"Token '{token_text}' not found, no best logprob, using penalty: {penalty}")
                            else:
                                # No logprobs available, use fixed penalty
                                penalty = -200.0
                                token_log_probs.append(penalty)
                                logging.warning(f"No logprobs data for token '{token_text}', using penalty: {penalty}")

                        # Update the prefix for the next token
                        current_prefix += token_text

                    except Exception as e:
                        logging.warning(f"Error generating with vLLM for token '{token_text}': {str(e)}")
                        # Use a penalty value for this token
                        penalty = -200.0
                        token_log_probs.append(penalty)
                        # Still update the prefix to continue with next token
                        current_prefix += token_text

                # Compute the average log probability
                if token_log_probs:
                    avg_log_prob = sum(token_log_probs)
                    log_probs.append(avg_log_prob)
                    logging.info(f"Template average log probability: {avg_log_prob}")

            except Exception as e:
                logging.warning(f"Error computing vLLM logprob: {str(e)}")
                continue
        else:
            # vLLM with explicit reasoning mode
            try:
                # Create chat prompt as string
                chat_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )#.to(device)
                full_prompt = tokenizer.decode(chat_ids[0], skip_special_tokens=False) 
                # First, generate the full assistant response with reasoning
                # Temperature higher to allow for natural reasoning
                full_response = generate_text(
                    model,
                    tokenizer,
                    full_prompt,
                    max_tokens=512,  # Allow enough tokens for reasoning
                    temperature=1.0,  # Allow some creativity in reasoning
                    top_p=1.0,
                    use_vllm=True
                )

                logging.info(f"Generated explicit response: {full_response}")

                # Check if the response contains "Answer:" format
                if "Answer:" not in full_response:
                    logging.warning(f"Generated response does not contain 'Answer:' marker: {full_response}")
                    continue

                # Create the full prompt including generated reasoning up to "Answer:"
                full_prompt_with_reasoning = full_prompt + full_response.split("Answer:", 1)[0] + "Answer:"

                # Now compute token-by-token logprobs for the relation
                token_log_probs = []

                # Tokenize the relation part correctly
                relation_with_space = " " + relation.strip()
                relation_token_ids = tokenizer.encode(relation_with_space, add_special_tokens=False)
                relation_tokens = [tokenizer.decode([token_id]) for token_id in relation_token_ids]
       
                logging.info(f"Computing explicit logprobs for relation '{relation}' with {len(relation_token_ids)} tokens")

                # Use the full prompt with reasoning up to "Answer:" as prefix
                current_prefix = full_prompt_with_reasoning

                # Calculate log probabilities for each token of the relation
                for i, token_id in enumerate(relation_token_ids):
                    token_text = relation_tokens[i]

                    # Generate the next token with logprobs
                    try:
                        _, logprobs_data = generate_text(
                            model,
                            tokenizer,
                            current_prefix,
                            max_tokens=1,
                            temperature=1.0,  # Use deterministic generation for logprobs
                            top_p=1.0,
                            use_vllm=True,
                            return_logprobs=True,
                            logprobs_top_k=50  # vLLM maximum allowed value
                        )

                        # Find our token by comparing token IDs in the logprobs data
                        token_logprob = None

                        # Check if logprobs is available
                        if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
                            logprobs_dict = logprobs_data['logprobs'][0]  # First token's logprobs

                            # Check if our token_id is directly in the dictionary keys
                            if token_id in logprobs_dict:
                                # Extract the logprob value
                                logprob_obj = logprobs_dict[token_id]
                                if hasattr(logprob_obj, 'logprob'):
                                    token_logprob = logprob_obj.logprob
                                elif isinstance(logprob_obj, (int, float)):
                                    token_logprob = logprob_obj
                            else:
                                # Try matching by decoded token
                                for id_key, logprob_obj in logprobs_dict.items():
                                    if hasattr(logprob_obj, 'decoded_token'):
                                        # Compare normalized versions to handle whitespace differences
                                        decoded = logprob_obj.decoded_token
                                        if decoded.replace('Ġ', ' ').strip() == token_text.strip():
                                            token_logprob = logprob_obj.logprob
                                            break

                        if token_logprob is not None:
                            token_log_probs.append(token_logprob)
                            logging.info(f"Explicit token '{token_text}' (ID: {token_id}) logprob: {token_logprob}")
                        else:
                            # If token not found, try to get the highest probability token as fallback
                            if 'logprobs' in logprobs_data and logprobs_data['logprobs']:
                                logprobs_dict = logprobs_data['logprobs'][0]
                                highest_logprob = None

                                # Find the highest ranked logprob
                                for id_key, logprob_obj in logprobs_dict.items():
                                    if hasattr(logprob_obj, 'rank') and logprob_obj.rank == 1:
                                        highest_logprob = logprob_obj.logprob
                                        break

                                if highest_logprob is not None:
                                    # Apply a penalty factor to the highest logprob
                                    adjusted_logprob = highest_logprob - 100.0  # Significant penalty
                                    token_log_probs.append(adjusted_logprob)
                                    logging.warning(f"Explicit token '{token_text}' not found, using penalized best logprob: {adjusted_logprob}")
                                else:
                                    # No highest logprob found, use fixed penalty
                                    penalty = -200.0
                                    token_log_probs.append(penalty)
                                    logging.warning(f"Explicit token '{token_text}' not found, no best logprob, using penalty: {penalty}")
                            else:
                                # No logprobs available, use fixed penalty
                                penalty = -200.0
                                token_log_probs.append(penalty)
                                logging.warning(f"No logprobs data for explicit token '{token_text}', using penalty: {penalty}")

                        # Update the prefix for the next token
                        current_prefix += token_text

                    except Exception as e:
                        logging.warning(f"Error generating with vLLM for explicit token '{token_text}': {str(e)}")
                        # Use a penalty value for this token
                        penalty = -200.0
                        token_log_probs.append(penalty)
                        # Still update the prefix to continue with next token
                        current_prefix += token_text

                # Compute the average log probability
                if token_log_probs:
                    avg_log_prob = sum(token_log_probs)
                    log_probs.append(avg_log_prob)
                    logging.info(f"Explicit template average log probability: {avg_log_prob}")

            except Exception as e:
                logging.warning(f"Error computing explicit vLLM logprob: {str(e)}")
                continue

    # Use the sum log probability among all templates
    sum_log_prob = sum(log_probs) if log_probs else -float("inf")
    return sum_log_prob

def compute_logprob_metric_batch(model, tokenizer, triples, system_message, user_template, use_vllm=True, logprobs_top_k=5):
    """
    Efficient batch computation of average log probabilities over relation tokens for a list of (head, relation, tail) triples.

    Args:
        model: vLLM engine
        tokenizer: Hugging Face tokenizer
        triples (list): List of (head, relation, tail) tuples
        system_message (str): System message for the chat template
        user_template (str): User message template with {entity1}, {entity2}, {relation}
        use_vllm (bool): Must be True for vLLM batch mode
        logprobs_top_k (int): Top-k logprobs to retrieve per token

    Returns:
        List[float]: Average log probability of relation tokens for each triple
    """
    from collections import defaultdict

    # Step 1: Build prompts up to "Answer:" and relation token info
    prefixes = []  # Prompt prefix for each triple
    relation_tokens_batch = []  # Decoded relation tokens per triple
    relation_token_ids_batch = []  # Token ids of relation

    for h, r, t in triples:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_template.format(entity1=h, entity2=t, relation=r)},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prefix = full_prompt.split("Answer:")[0] + "Answer:"
        prefixes.append(prefix)

        relation_with_space = " " + r.strip()
        token_ids = tokenizer.encode(relation_with_space, add_special_tokens=False)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        relation_token_ids_batch.append(token_ids)
        relation_tokens_batch.append(tokens)

    # Step 2: Find max token length to iterate by token position
    max_len = max(len(ids) for ids in relation_token_ids_batch)
    num_triples = len(triples)

    # Step 3: Init output containers
    logprobs_result = [[] for _ in range(num_triples)]

    # Step 4: Iterate over token positions (batch by token index)
    for step in range(max_len):
        batch_prompts = []
        batch_indices = []  # triple index
        expected_token_ids = []
        expected_token_strs = []

        for i in range(num_triples):
            if step >= len(relation_tokens_batch[i]):
                continue

            prev_tokens = "".join(relation_tokens_batch[i][:step])
            full_prompt = prefixes[i] + prev_tokens

            batch_prompts.append(full_prompt)
            batch_indices.append(i)
            expected_token_ids.append(relation_token_ids_batch[i][step])
            expected_token_strs.append(relation_tokens_batch[i][step])

        if not batch_prompts:
            continue

        # Step 5: Batched generation with logprobs
        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            batch_prompts,
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=logprobs_top_k,
        )

        # Step 6: Match predicted token with expected token
        for j, (idx, token_id, token_str) in enumerate(zip(batch_indices, expected_token_ids, expected_token_strs)):
            logprobs_data = batch_outputs[j][1]
            logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
            logprob = None

            # Match by decoded token string
            for tok_key, logprob_obj in logprobs_dict.items():
                if hasattr(logprob_obj, 'decoded_token'):
                    decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                    if decoded == token_str.strip():
                        logprob = logprob_obj.logprob
                        break
                elif isinstance(tok_key, str):
                    if tok_key.replace("Ġ", " ").strip() == token_str.strip():
                        if isinstance(logprob_obj, (int, float)):
                            logprob = logprob_obj
                        elif hasattr(logprob_obj, "logprob"):
                            logprob = logprob_obj.logprob
                        break

            if logprob is not None:
                logprobs_result[idx].append(logprob)
            else:
                logprobs_result[idx].append(-200.0)  # Fallback penalty

    # Step 7: Return average logprob per triple
    return [
        sum(logprobs) if logprobs else -float("inf")
        for logprobs in logprobs_result
    ]


def compute_judgement_logprob_metric_batch(model, tokenizer, triples, times, device, system_message=None, user_template=None, use_chat_template=True, use_vllm=True, temperature=1.0, logprobs_top_k=5):
    """
    Efficient batch computation of Yes/No judgement log probabilities for all combinations of input triples and relations.
    
    Args:
        model: vLLM engine
        tokenizer: Hugging Face tokenizer
        triples (list): List of (head, _, tail) tuples - the relation in the triple is ignored
        relation_set (list): List of relations to evaluate with each triple
        times (int): Number of times to repeat the computation for each prompt
        system_message (str): System message for the chat template
        user_template (str): User message template
        use_chat_template (bool): True for Qwen and False for Llama
        use_vllm (bool): Must be True for vLLM batch mode
        logprobs_top_k (int): Top-k logprobs to retrieve per token
        
    Returns:
        Dict: Mapping of (head, relation, tail) -> list of times results, each with {'Yes': (logprob, prob), 'No': (logprob, prob)}
    """
    import math
    
    # Step 1: Generate all combinations of triples with relations from relation_set
    """ all_combinations = []
    for h, _, t in triples:  # Ignore the existing relation in the triple
        for r in relation_set:
            all_combinations.append((h, r, t)) """
    all_combinations = triples
    
    # Step 2: Build prompts up to "Answer:"
    prefixes = []  # Prompt prefix for each combination
    combo_indices = []  # Keep track of which combination each prefix belongs to
    
    for h, r, t in all_combinations:
        if use_chat_template:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_template.format(entity1=h, entity2=t, relation=r)},
            ]
            full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            full_prompt = system_message + "\n\n" +user_template.format(entity1=h, entity2=t, relation=r)
        prefix = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
        
        # Add this prefix 'times' times to the batch
        for _ in range(times):
            prefixes.append(prefix)
            combo_indices.append((h, r, t))  # Keep track of which combination this is

    # Step 3: Get token IDs for "Yes" and "No"
    yes_with_space = " Yes"
    no_with_space = " No"
    unknown_with_space = " Unknown"

    yes_token_id = tokenizer.encode(yes_with_space, add_special_tokens=False)[0]  # Just get the first token
    no_token_id = tokenizer.encode(no_with_space, add_special_tokens=False)[0]    # Just get the first token
    unknown_token_id = tokenizer.encode(unknown_with_space, add_special_tokens=False)[0]    # Just get the first token
    
    yes_decoded = tokenizer.decode([yes_token_id]).replace("Ġ", " ").strip()
    no_decoded = tokenizer.decode([no_token_id]).replace("Ġ", " ").strip()
    unknown_decoded = tokenizer.decode([unknown_token_id]).replace("Ġ", " ").strip()

    # Step 4: Initialize results container
    # We'll store all results and later group them by combination
    all_results = []  # List of (combo, {'Yes': (logprob, prob), 'No': (logprob, prob)})
    
    # Step 5: Single batch call to vLLM with all prompts
    batch_outputs = generate_text_batch(
        model,
        tokenizer,
        prefixes,
        max_tokens=1,
        temperature=temperature,
        top_p=1.0,
        use_vllm=use_vllm,
        return_logprobs=True,
        logprobs_top_k=logprobs_top_k,
    )
    
    # Process all outputs in one go
    for i, output in enumerate(batch_outputs):
        combo = combo_indices[i]
        logprobs_data = output[1]
        logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
        
        yes_found = no_found = unknown_found = False
        yes_logprob = yes_prob = no_logprob = no_prob = unknown_logprob = unknown_prob = None
        
        # Find both Yes and No probabilities
        for tok_key, logprob_obj in logprobs_dict.items():
            if hasattr(logprob_obj, 'decoded_token'):
                decoded = logprob_obj.decoded_token.replace("Ġ", " ").strip()
                
                if decoded == yes_decoded and not yes_found:
                    yes_logprob = logprob_obj.logprob
                    yes_prob = math.exp(yes_logprob)
                    yes_found = True
                
                if decoded == no_decoded and not no_found:
                    no_logprob = logprob_obj.logprob
                    no_prob = math.exp(no_logprob)
                    no_found = True
                
                if decoded == unknown_decoded and not unknown_found:
                    unknown_logprob = logprob_obj.logprob
                    unknown_prob = math.exp(unknown_logprob)
                    unknown_found = True

                # Break early if we found both
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
    
    # Step 6: Group results by combination
    results = {combo: [] for combo in all_combinations}
    for combo, result in all_results:
        results[combo].append(result)
    
    # Step 7: Calculate statistics and confidence scores
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
    for combo in all_combinations:
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
    
    # Calculate confidence scores: mean("Yes" prob) * (1 - std("Yes" prob))
    """ confidence_scores = [
        stats[combo]["Yes_prob_mean"]
        for combo in all_combinations
    ] """

    confidence_scores = []
    pred_indices = []
    cnt = 0
    for combo in all_combinations:
        prob_means = {
            "Yes": stats[combo]["Yes_prob_mean"],
            "No": stats[combo]["No_prob_mean"],
            "Unknown": stats[combo]["Unknown_prob_mean"]
        }
        max_answer = max(prob_means, key=prob_means.get)
        max_prob = prob_means[max_answer]
        if max_answer != "Unknown":
            confidence_scores.append((stats[combo]["No_prob_mean"], stats[combo]["Yes_prob_mean"], stats[combo]["Unknown_prob_mean"]))
            pred_indices.append(1 if max_answer == "Yes" else 0)
        else:
            cnt += 1
            # print(combo)
    # print(cnt)
    
    sum_result = [
        stats[combo]["Yes_prob_mean"] + stats[combo]["No_prob_mean"] + stats[combo]["Unknown_prob_mean"]
        for combo in all_combinations
    ]
    
    def _tuple_to_tensor(prob):
        """
        Transform a [(Noprob, Yesprob, Unknownprob)]
        to a pytorch tensor [[Noprob, Yesprob, Unknownprob, raminingprob]]
        """
        confidence_tensor = torch.tensor(prob).to(device)
        remaining = 1 - confidence_tensor.sum(dim=1, keepdim=True)
        tensor_prob = torch.cat([confidence_tensor, remaining], dim=1)
        return tensor_prob
    tensor_scores = _tuple_to_tensor(confidence_scores)
    return tensor_scores, sum_result

def compute_tf_logprob_metric(model, tokenizer, triples, system_message, user_template, use_vllm=True, times=5, logprobs_top_k=1000):
    """
    Efficient batch computation of True/False log probabilities over relation tokens for a list of (head, relation, tail) triples.

    Args:
        model: vLLM engine
        tokenizer: Hugging Face tokenizer
        triples (list): List of (head, relation, tail) tuples
        system_message (str): System message for the chat template
        user_template (str): User message template with {entity1}, {entity2}, {relation}
        use_vllm (bool): Must be True for vLLM batch mode
        times (int): Number of times for vLLM inference
        logprobs_top_k (int): Top-k logprobs to retrieve per token

    Returns:
        List[List]: Average log probability of True and False tokens for each triple
    """
    from collections import defaultdict
    system_message = TF_LOGPROB_SYSTEM_MESSAGE
    user_template = TF_LOGPROB_USER_TEMPLATES
    # Step 1: Build prompts up to "Answer:" and relation token info
    prefixes = []  # Prompt prefix for each triple

    for h, r, t in triples:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_template.format(entity1=h, entity2=t, relation=r)},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # split_prompt = full_prompt.split("Answer:")
        # split_prompt = split_prompt[:-1]
        # prefix = "Answer:".join(split_prompt) + "Answer:"
        prefix = full_prompt
        prefixes.append(prefix)
    batch_prompts = prefixes

    # Step 2: Get token IDs for " Yes" and " No"; These two should be single token in most cases.
    yes_with_space = " True"
    no_with_space = " False"
    yes_tokens = tokenizer.encode(yes_with_space, add_special_tokens=False)
    no_tokens = tokenizer.encode(no_with_space, add_special_tokens=False)
    assert len(yes_tokens) == len(no_tokens)
    assert len(yes_tokens) == 1
    yes_decoded = [tokenizer.decode([tid]) for tid in yes_tokens]
    no_decoded = [tokenizer.decode([tid]) for tid in no_tokens]

    # Increase the token probability for yes and no answeers
    logit_bias = {yes_tokens[0]: 100, no_tokens[0]: 100}

    num_triples = len(triples)

    # Step 3: Init output containers
    logprobs_result = [[[],[]] for _ in range(num_triples)]
    probs_result = [[[],[]] for _ in range(num_triples)]

    # Step 4: Run the computation 'times' times
    for time in range(times):
        from vllm import SamplingParams
        sampling_params = SamplingParams(max_tokens=25, temperature=0.1)
        reasoning_outputs = model.generate(batch_prompts, sampling_params)

        answer_batch_prompts = []
        for i, (h, r, t) in enumerate(triples):
            messages = [
                {"role": "user", "content": user_template.format(entity1=h, entity2=t, relation=r)},
                {"role": "assistant", "content": reasoning_outputs[i].outputs[0].text.strip()},
                {"role": "user", "content": "Now, answer with True or False only. Answer:"}
            ]
            answer_batch_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            answer_batch_prompts.append(answer_batch_prompt)

        batch_outputs = generate_text_batch(
            model,
            tokenizer,
            answer_batch_prompts,
            max_tokens=1,
            temperature=0.1,
            top_p=1.0,
            use_vllm=True,
            return_logprobs=True,
            logprobs_top_k=logprobs_top_k,
            logit_bias=logit_bias
        )

        # Step 5: Match predicted token with expected token
        for j, (token_id, token_str) in enumerate(zip(no_tokens + yes_tokens, no_decoded + yes_decoded)):
            for idx in range(num_triples):
                logprobs_data = batch_outputs[idx][1]
                logprobs_dict = logprobs_data.get("logprobs", [{}])[0]
                logprob = None
                try:
                    # Match by token id
                    logprob_obj = logprobs_dict[token_id]
                    logprob = logprob_obj.logprob
                except KeyError:
                    # The expected token " Yes" or " No" is not in the top k outputs
                    # Fallback penalty
                    logprob = -200.0
                logprobs_result[idx][j].append(logprob)
                probs_result[idx][j].append(math.exp(logprob))
    # Step 7: Return average logprob [[False logprob], [True logprob]] per triple
    return [
        [
            sum(logprobs)/times if logprobs else -float("inf")
            for logprobs in pair
        ]
        for pair in logprobs_result
    ]


def compute_llm_score_metric(model, tokenizer, triple, device, max_new_tokens=200, use_vllm=False):
    """
    Compute the confidence score generated by LLM for a given triple (entity1, relation, entity2).
    
    Args:
        model: HuggingFace model or vLLM LLM instance.
        tokenizer: Corresponding tokenizer.
        triple (tuple): A triple (entity1, relation, entity2).
        device: Torch device.
        max_new_tokens (int): Maximum tokens to generate.
        use_vllm (bool): Whether to use vLLM for generation.
        
    Returns:
        tuple: (raw_confidence_text, integer_score)
    """
    entity1, relation, entity2 = triple

    # Convert to chat format
    messages = [
        {"role": "system", "content": LLM_SCORE_SYSTEM_MESSAGE},
        {"role": "user", "content": LLM_SCORE_USER_TEMPLATES.format(entity1=entity1, entity2=entity2, relation=relation)}
    ]
    
    # Use the same parameters for both vLLM and non-vLLM to ensure consistency
    generation_temperature = 0.3
    generation_top_k = 20
    generation_top_p = 0.85
    
    if use_vllm:
        # For vLLM, use generate_text with exactly the same parameters as non-vLLM
        chat_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        full_prompt = tokenizer.decode(chat_ids[0], skip_special_tokens=False) 
        
        raw_response = generate_text(
            model, 
            tokenizer, 
            full_prompt, 
            max_tokens=max_new_tokens,
            temperature=generation_temperature,
            top_k=generation_top_k,
            top_p=generation_top_p,
            use_vllm=True
        )
    else:
        # For standard models, use the original approach
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
                temperature=generation_temperature,
                top_k=generation_top_k,
                top_p=generation_top_p
            )

        # Extract only the newly generated response
        response_tokens = outputs[0][input_ids.shape[-1]:]
        raw_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Shared response processing for both methods
    def extract_relation_and_score():
        response = raw_response.strip()
        response = re.sub(r"\s+", " ", response)
        match = re.findall(r"RELATIONSHIP:\s*([/\w_-]+)\s*;\s*CONFIDENCE:\s*([\w\s']+)", response, re.IGNORECASE)
        
        if not match: 
            # Sometimes LLM will not generate RELATIONSHIP: <relation>; CONFIDENCE: <score>
            logging.info(f"No valid RELATIONSHIP and CONFIDENCE pairs found in the response. Raw response:\n {raw_response}")
            conf = next((rating for rating in LLM_SCORE_RAW_RATING if rating.lower() in response.lower()), None) 
        else: 
            _, conf = match[0]        
        
        # Find the best match from valid_scores based on the input score
        closest_match = difflib.get_close_matches(conf if conf else "", LLM_SCORE_RAW_RATING, n=1, cutoff=0.2)
        
        if not closest_match:
            logging.warning(f"Confidence score cannot be matched. Raw response:\n {raw_response}")
            score = 0 
            return (conf if conf else "Unknown", score)
        
        matched_score = closest_match[0].lower().strip()  # Normalize the matched score
        
        # Mapping the matched score to a numeric value
        if "not meaningful" in matched_score: score = 1
        elif "i don't know" in matched_score: score = 1
        elif "plausible" in matched_score: score = 2
        elif "definite" in matched_score: score = 3
        else: score = 0
        
        return (conf if conf else matched_score, score)
    
    return extract_relation_and_score()

# Modified to support vLLM
def evaluate_thresholds(model, model_name, tokenizer, kg, method="perplexity", calibrate=True, multi_gpu=False,  num_samples=100, output_dir=None, thresholds=None, cache_path=None, use_vllm=False, explicit=False, use_batch=False):
    """
    Evaluate model performance for different thresholds using either the perplexity or log probability method.
    Now with vLLM support.

    Args:
        model: HuggingFace model or vLLM LLM instance.
        tokenizer: Corresponding tokenizer.
        kg: KnowledgeGraph instance.
        method (str): "perplexity" or "logprob" to select the scoring method.
        multi_gpu (bool): Whether to use multi-GPU.
        num_samples (int): Number of samples per type.
        thresholds (list or None): List of thresholds to evaluate. If None, sampled from ROC thresholds.
        cache_path (str or None): Path to cache directory for storing/loading computed scores.
        use_vllm (bool): Whether to use vLLM for computations.

    Returns:
        dict: Evaluation results.
    """
    # First, try to load scores from cache
    cache_results = check_scores_cache(cache_path, method, kg.name, 
                                      model_name, 
                                      num_samples)
    positive_pred, negative_pred = None, None

    if cache_results is not None:
        positive_scores, negative_scores, positive_samples, negative_samples = cache_results
        logging.info(f"Using cached scores: {len(positive_scores)} positive and {len(negative_scores)} negative samples")
    else:
        # If no cache is available, compute scores normally
        positive_samples, negative_samples = load_evaluation_samples(kg, num_samples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not multi_gpu and not use_vllm:
            model.to(device)
        elif not use_vllm:
            device = model.device

        logging.info(f"Computing scores for positive samples using method '{method}' with vLLM={use_vllm}...")
        if method == "perplexity":
            positive_scores = [compute_perplexity_metric(model, tokenizer, triple, device, use_vllm)
                               for triple in tqdm(positive_samples)]
            
        elif method == "logprob" and not use_batch:
            positive_scores = [compute_logprob_metric(model, tokenizer, triple, device, use_vllm, explicit)
                               for triple in tqdm(positive_samples)]
        elif method == "tf_logprob" and use_batch:
            times = 5
            logprobs_top_k = 1000
            positive_scores = compute_tf_logprob_metric(model, tokenizer, positive_samples, device, use_vllm, times=times, logprobs_top_k=logprobs_top_k)
            # Use the 'Yes' score logporb
            # positive_scores = [positive_score[1] for positive_score in positive_scores]
            positive_scores = [positive_score[1] - positive_score[0] for positive_score in positive_scores]
        elif method == "logprob" and use_batch:
            logging.info("Using batch-based logprob computation with vLLM.")
            user_template = JUDGE_USER_TEMPLATES[model_name]
            system_message = JUDGE_SYSTEM_MESSAGE
            times = 5
            use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
            positive_prob, sum_result_positive = compute_judgement_logprob_metric_batch(model, tokenizer, positive_samples, times, device, system_message, user_template, use_chat_template, use_vllm, temperature=1.0, logprobs_top_k=5)
            positive_scores, positive_pred = torch.max(positive_prob, -1)
            positive_scores, positive_pred = positive_scores.tolist(), positive_pred.tolist()

        elif method == "llm_score":
            (positive_raw_rating, positive_scores) = zip(*[compute_llm_score_metric(model, tokenizer, triple, device, use_vllm=use_vllm)
                               for triple in tqdm(positive_samples)])            
        else:
            raise ValueError("Unsupported method. Choose 'perplexity', 'logprob', or 'llm_score'.")

        logging.info(f"Computing scores for negative samples using method '{method}'...")
        if method == "perplexity":
            negative_scores = [compute_perplexity_metric(model, tokenizer, triple, device, use_vllm)
                               for triple in tqdm(negative_samples)]
            
        elif method == "logprob" and not use_batch:
            negative_scores = [compute_logprob_metric(model, tokenizer, triple, device, use_vllm, explicit)
                               for triple in tqdm(negative_samples)]
        elif method == "tf_logprob" and use_batch:
            times = 5
            logprobs_top_k = 1000
            negative_scores = compute_tf_logprob_metric(model, tokenizer, negative_samples, device, use_vllm, times=times, logprobs_top_k=logprobs_top_k)
            # Use the 'Yes' score logporb
            # negative_scores = [negative_score[1] for negative_score in negative_scores]
            negative_scores = [negative_score[1] - negative_score[0] for negative_score in negative_scores]
        elif method == "logprob" and use_batch:
            logging.info("Using batch-based logprob computation with vLLM.")
            user_template = JUDGE_USER_TEMPLATES[model_name]
            system_message = JUDGE_SYSTEM_MESSAGE
            times = 5
            use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False
            negative_prob, sum_result_negative = compute_judgement_logprob_metric_batch(model, tokenizer, negative_samples, times, device, system_message, user_template, use_chat_template, use_vllm, temperature=1.0, logprobs_top_k=5)
            negative_scores, negative_pred = torch.max(negative_prob, -1)
            negative_scores, negative_pred = negative_scores.tolist(), negative_pred.tolist()            
            
            plot_probability_sum_distribution(sum_result_positive, sum_result_negative, "./sum_distribution.png")
        
        elif method == "llm_score":
            (negative_raw_rating, negative_scores) = zip(*[compute_llm_score_metric(model, tokenizer, triple, device, use_vllm=use_vllm)
                               for triple in tqdm(negative_samples)])
        # Save results to cache for future use
        # save_scores_cache(cache_path, method, kg.name, 
        #                  model_name,
        #                  num_samples, positive_scores, negative_scores, 
        #                  positive_samples, negative_samples)
        if method == "llm_score":
            save_raw_rating_cache(cache_path, method, kg.name, 
                            model_name,
                            num_samples, positive_raw_rating, negative_raw_rating, 
                            positive_samples, negative_samples)
    if calibrate:
        if isinstance(positive_prob, torch.Tensor):
            positive_prob_np = positive_prob.cpu().numpy()
            negative_prob_np = negative_prob.cpu().numpy()
        prob = np.concatenate([positive_prob_np, negative_prob_np], axis=0)
        y_true = np.array([1] * len(positive_prob_np) + [0] * len(negative_prob_np))
        y_pred =  positive_pred +  negative_pred
        _model_name = model_name.split("/")[1]
        np.savez(f"./plots/calibration/uncal_{kg.name}_{_model_name}.npz", prob=prob, y_true=y_true)

        # Plot reliability diagram from uncalibrated confidence scores
        if output_dir:
            dist_path = os.path.join(output_dir, f"uncal_reliability_diagram_{method}.png")
            plot_histogram_reliability(y_true, y_pred, positive_scores + negative_scores, 10, dist_path)
        else:
            plot_histogram_reliability(y_true, y_pred, positive_scores + negative_scores, 10)        

        print(positive_pred, negative_pred)
        accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        logging.info(f"Uncalibrated Accuracy: {accuracy:.4f}")
        temp_model = temp_scaling(model, tokenizer, positive_prob, negative_prob, times, device, system_message, user_template, use_chat_template)
        logging.info("Finish Calibration !!")
        # positive_prob, sum_result_positive = compute_judgement_logprob_metric_batch(model, tokenizer, positive_samples, times, device, system_message, user_template, use_chat_template, use_vllm, temperature=best_temperature, logprobs_top_k=5)
        positive_prob = temp_model(positive_prob[:,:3])
        positive_scores, positive_pred = torch.max(positive_prob[:,:3], -1)
        positive_scores, positive_pred = positive_scores.tolist(), positive_pred.tolist()
        negative_prob = temp_model(negative_prob[:,:3])
        # negative_prob, sum_result_negative = compute_judgement_logprob_metric_batch(model, tokenizer, negative_samples, times, device, system_message, user_template, use_chat_template, use_vllm, temperature=best_temperature, logprobs_top_k=5)
        negative_scores, negative_pred = torch.max(negative_prob[:,:3], -1)
        negative_scores, negative_pred = negative_scores.tolist(), negative_pred.tolist()
        if isinstance(positive_prob, torch.Tensor):
            positive_prob_np = positive_prob.detach().cpu().numpy()
            negative_prob_np = negative_prob.detach().cpu().numpy()
        prob = np.concatenate([positive_prob_np, negative_prob_np], axis=0)
        np.savez(f"./plots/calibration/cal_{kg.name}_{_model_name}.npz", prob=prob, y_true=y_true)
        temp = temp_model.temperature.detach().cpu()
        torch.save(temp, f"./plots/calibration/temp_{kg.name}_{_model_name}.pt")

        # Plot reliability diagram from calibrated confidence scores        
        if output_dir:
            dist_path = os.path.join(output_dir, f"cal_reliability_diagram_{method}.png")
            plot_histogram_reliability(y_true, positive_pred +  negative_pred, positive_scores + negative_scores, 10, dist_path)
        else:
            plot_histogram_reliability(y_true, positive_pred +  negative_pred, positive_scores + negative_scores, 10)        

    # Prepare labels and scores for ROC curve.
    # For perplexity: lower scores are better, so use negative scores.
    # For log probability: higher scores are better.
    if method == "perplexity":
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_score = [-s for s in positive_scores] + [-s for s in negative_scores]
    elif method in ["logprob", "llm_score", "tf_logprob"]:
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_score = positive_scores + negative_scores

    fpr, tpr, threshold_values = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # If thresholds not provided, sample a few points from the ROC thresholds.
    if thresholds is None:
        sampled_indices = slice(0, len(threshold_values), max(1, len(threshold_values) // 10))
        if method == "perplexity":
            thresholds = [-t for t in threshold_values[sampled_indices]]
        elif method in ["logprob", "llm_score", "tf_logprob"]:
            thresholds = list(threshold_values[sampled_indices])

    results = {}
    for th in thresholds:
        if method == "perplexity":
            true_positives = sum(1 for s in positive_scores if s <= th)
            true_negatives = sum(1 for s in negative_scores if s > th)
        elif method in ["logprob", "llm_score", "tf_logprob"]:
            true_positives = sum(1 for s in positive_scores if s >= th)
            true_negatives = sum(1 for s in negative_scores if s < th)

        precision = true_positives / (true_positives + (len(negative_scores) - true_negatives)) if true_positives > 0 else 0
        recall = true_positives / len(positive_scores) if positive_scores else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / (len(positive_scores) + len(negative_scores))

        results[th] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }

    best_threshold = max(results.keys(), key=lambda t: results[t]["f1"])
    
    if method == "logprob" and use_batch:
        y_pred =  positive_pred +  negative_pred
        accuracy = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)
        logging.info(f"Final Accuracy: {accuracy:.4f}")
    else:
        accuracy = -99

    return {
        "thresholds": results,
        "auc": roc_auc,
        "best_threshold": best_threshold,
        "best_f1": results[best_threshold]["f1"],
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": threshold_values,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "method": method,
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "positive_pred": positive_pred,
        "negative_pred": negative_pred
    }

def plot_roc_curve(evaluation_results, output_path=None):
    """
    Plot the ROC curve and display the AUC value.
    
    Args:
        evaluation_results (dict): Evaluation results containing fpr, tpr, and auc.
        output_path (str or None): If provided, save the plot to this path.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(
        evaluation_results["fpr"],
        evaluation_results["tpr"],
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {evaluation_results["auc"]:.3f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Based on {evaluation_results["method"].capitalize()}')
    plt.legend(loc="lower right")
    
    best_threshold = evaluation_results["best_threshold"]
    best_f1 = evaluation_results["best_f1"]
    plt.annotate(
        f'Best threshold: {best_threshold:.2f}\nF1: {best_f1:.3f}',
        xy=(0.5, 0.5),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
    )
    
    if output_path:
        plt.savefig(output_path)
        logging.info(f"ROC curve saved to {output_path}")
    plt.show()


def plot_score_distributions(positive_scores, negative_scores, method="perplexity", best_threshold=None, output_path=None):
    """
    Plot the score distributions for positive and negative samples.

    Args:
        positive_scores (list): Pre-computed scores for positive samples.
        negative_scores (list): Pre-computed scores for negative samples.
        method (str): "perplexity" or "logprob".
        best_threshold (float): The best threshold from evaluation.
        output_path (str or None): If provided, save the plot to this path.
    """
    plt.figure(figsize=(12, 6))
    all_scores = positive_scores + negative_scores
    bins = np.linspace(min(all_scores), max(all_scores), 30)

    label_pos = 'Positive Samples'
    label_neg = 'Negative Samples'
    color_pos = 'green'
    color_neg = 'red'

    plt.hist(positive_scores, bins=bins, alpha=0.5, label=label_pos, color=color_pos)
    plt.hist(negative_scores, bins=bins, alpha=0.5, label=label_neg, color=color_neg)

    plt.axvline(
        x=np.median(positive_scores),
        color=color_pos,
        linestyle='--',
        label=f'{label_pos} median: {np.median(positive_scores):.2f}'
    )
    plt.axvline(
        x=np.median(negative_scores),
        color=color_neg,
        linestyle='--',
        label=f'{label_neg} median: {np.median(negative_scores):.2f}'
    )

    if best_threshold is not None:
        plt.axvline(
            x=best_threshold,
            color='blue',
            linestyle='-',
            label=f'Best threshold: {best_threshold:.2f}'
        )

    plt.title(f'{method.capitalize()} Distributions for Positive and Negative Samples')
    plt.xlabel('Score')
    plt.ylabel('Number of Samples')
    plt.legend()

    if output_path:
        plt.savefig(output_path)
        logging.info(f"Score distribution plot saved to {output_path}")

    plt.show()

def plot_histogram_reliability(y_true, y_pred, confidences, n_bins=10, output_path=None):
    """
    Plot a histogram-style reliability diagram showing model calibration.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        confidences (list): Confidence scores for the predicted labels.
        n_bins (int): Number of bins to divide the predicted probabilities into.

    """
    # Bin edges and centers
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confidences = np.array(confidences)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_ids = np.digitize(confidences, bin_edges, right=True)

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if np.any(mask):
            bin_conf[i - 1] = confidences[mask].mean()
            bin_acc[i - 1] = np.mean(y_true[mask] == y_pred[mask])
            bin_counts[i - 1] = mask.sum()

    # Normalize for plotting (optional)
    bin_counts_norm = bin_counts / bin_counts.max()

    # Expected Calibration Error (ECE)
    ece = np.sum(np.abs(bin_acc - bin_conf) * bin_counts) / np.sum(bin_counts)

    # Plotting
    plt.figure(figsize=(10, 8))
    for i in range(n_bins):
        # Main accuracy bar
        plt.bar(bin_centers[i], bin_acc[i], width=0.08, color='blue', alpha=0.8,
                label='Outputs' if i == 0 else "")
        # Gap bar (error between confidence and accuracy)
        plt.bar(bin_centers[i], np.abs(bin_acc[i] - bin_conf[i]),
                bottom=min(bin_acc[i], bin_conf[i]), width=0.08,
                color='red', alpha=0.3, hatch='//', edgecolor='r',
                label='Gap' if i == 0 else "")

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Add error text box
    plt.text(0.5, 0.05, f'ECE={ece:.2f}', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Reliability diagram saved to {output_path}")

    plt.show()

def analyze_relation_distribution(kg, positive_samples, positive_scores, method="logprob", 
                                  threshold=None, fpr_value=0.01, top_k=7, output_dir=None):
    """
    Analyze the distribution of relations in positive samples, focusing on those
    that would be correctly classified at low FPR thresholds.

    Args:
        kg (KnowledgeGraph): The knowledge graph instance.
        positive_samples (list): List of positive triples.
        positive_scores (list): Scores for positive samples.
        method (str): "perplexity" or "logprob".
        threshold (float): Score threshold at the specified FPR value.
        fpr_value (float): The FPR value of interest.
        output_dir (str): Directory to save plots.

    Returns:
        dict: Relation distribution statistics.
    """
    # Determine which positive samples would be correctly classified at the threshold
    if method == "perplexity":
        # For perplexity, lower is better
        correctly_classified = [sample for sample, score in zip(positive_samples, positive_scores) 
                               if score <= threshold]
    else:
        # For logprob, higher is better
        correctly_classified = [sample for sample, score in zip(positive_samples, positive_scores)
                               if score >= threshold]

    # Extract relations from correctly classified samples
    relations = [sample[1] for sample in correctly_classified]

    # Count relation frequencies
    relation_counts = {}
    for relation in relations:
        if relation in relation_counts:
            relation_counts[relation] += 1
        else:
            relation_counts[relation] = 1

    # Sort relations by frequency
    sorted_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)

    # Create a bar chart of relation distribution
    plt.figure(figsize=(12, 8))
    rel_names = [rel for rel, _ in sorted_relations[:15]]  # Top 15 relations
    rel_counts = [count for _, count in sorted_relations[:15]]

    plt.bar(range(len(rel_names)), rel_counts, color='skyblue')
    plt.xticks(range(len(rel_names)), rel_names, rotation=45, ha='right')
    plt.title(f'Distribution of Relations in Positive Samples at FPR={fpr_value}')
    plt.xlabel('Relation')
    plt.ylabel('Frequency')
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"relation_distribution_fpr_{fpr_value}.png"))
        logging.info(f"Relation distribution plot saved for FPR={fpr_value}")

    plt.show()

    # Create a pie chart for the top relations
    plt.figure(figsize=(10, 10))
    total_count = sum(relation_counts.values())

    if len(sorted_relations) > top_k:
        top_relations = sorted_relations[:top_k]
        other_count = sum(count for _, count in sorted_relations[top_k:])

        labels = [f"{rel} ({count/total_count*100:.1f}%)" for rel, count in top_relations]
        labels.append(f"Other ({other_count/total_count*100:.1f}%)")

        sizes = [count for _, count in top_relations] + [other_count]
    else:
        labels = [f"{rel} ({count/total_count*100:.1f}%)" for rel, count in sorted_relations]
        sizes = [count for _, count in sorted_relations]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title(f'Relation Distribution in True Positives at FPR={fpr_value}')

    if output_dir:
        plt.savefig(os.path.join(output_dir, f"relation_pie_chart_fpr_{fpr_value}.png"))

    plt.show()

    # Calculate Shannon entropy to measure how evenly distributed the relations are
    entropy = 0
    for _, count in relation_counts.items():
        p = count / total_count
        entropy -= p * np.log2(p)

    # Perfect uniformity would have entropy of log2(num_relations)
    max_entropy = np.log2(len(relation_counts))
    uniformity_ratio = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "total_correct_positives": len(correctly_classified),
        "unique_relations": len(relation_counts),
        "top_relations": sorted_relations[:5],
        "entropy": entropy,
        "max_entropy": max_entropy,
        "uniformity_ratio": uniformity_ratio  # 1.0 means perfectly uniform
    }


def compute_confidence_thresholds(evaluation_results, method="fpr", fpr_values=None, ratio_values=None):
    """
    Compute confidence score thresholds based on either fixed FPR values or positive ratio in score distribution.

    Args:
        evaluation_results (dict): Evaluation results from evaluate_thresholds function.
        method (str): Method to compute thresholds, either "fpr" or "ratio".
        fpr_values (list): Three FPR values to use for computing thresholds, e.g. [0.01, 0.05, 0.1].
        ratio_values (list): Three positive ratio values to use for computing thresholds, e.g. [0.7, 0.9, 0.95].

    Returns:
        list: Three threshold values to be used for confidence score conversion.
    """
    if method == "fpr":
        if fpr_values is None:
            fpr_values = [0.01, 0.05, 0.1]  # Default FPR values

        if len(fpr_values) != 3:
            raise ValueError("Must provide exactly 3 FPR values")

        # Get thresholds for the specified FPR values
        fpr = evaluation_results["fpr"]
        roc_thresholds = evaluation_results["roc_thresholds"]

        # Interpolate to get thresholds at exact FPR values
        thresholds = np.interp(fpr_values, fpr, roc_thresholds)

        # If method is perplexity, negate thresholds (since lower is better)
        if evaluation_results["method"] == "perplexity":
            thresholds = -thresholds

        return thresholds.tolist()

    elif method == "ratio":
        if ratio_values is None:
            ratio_values = [0.7, 0.9, 0.95]  # Default ratio values

        if len(ratio_values) != 3:
            raise ValueError("Must provide exactly 3 ratio values")

        # Get scores for positive and negative samples
        positive_scores = evaluation_results["positive_scores"]
        negative_scores = evaluation_results["negative_scores"]

        # Combine all scores and labels
        all_scores = positive_scores + negative_scores
        all_labels = [1] * len(positive_scores) + [0] * len(negative_scores)

        # Sort scores and labels together
        combined = sorted(zip(all_scores, all_labels), key=lambda x: x[0])
        sorted_scores, sorted_labels = zip(*combined)

        # For each score, calculate positive ratio to its right
        thresholds = []
        for ratio_value in ratio_values:
            found = False

            # For perplexity (lower is better), iterate in reverse order
            if evaluation_results["method"] == "perplexity":
                # Convert to list for easier manipulation
                sorted_scores_list = list(sorted_scores)
                sorted_labels_list = list(sorted_labels)

                # Iterate from high scores (worst) to low scores (best)
                for i in range(len(sorted_scores_list)):
                    # Calculate positive ratio for all samples with score <= current score
                    samples_to_left = sorted_labels_list[:i+1]
                    if samples_to_left:
                        positive_ratio = sum(samples_to_left) / len(samples_to_left)

                        # If ratio is greater than or equal to the target, use this score as threshold
                        if positive_ratio >= ratio_value:
                            thresholds.append(sorted_scores_list[i])
                            found = True
                            break

            else:  # For logprob (higher is better)
                # Iterate from low scores (worst) to high scores (best)
                for i in range(len(sorted_scores)):
                    # Calculate positive ratio for all samples with score >= current score
                    samples_to_right = sorted_labels[i:]
                    if samples_to_right:
                        positive_ratio = sum(samples_to_right) / len(samples_to_right)

                        # If ratio is greater than or equal to the target, use this score as threshold
                        if positive_ratio >= ratio_value:
                            thresholds.append(sorted_scores[i])
                            found = True
                            break

            # If no threshold found for this ratio, use a fallback
            if not found:
                if evaluation_results["method"] == "perplexity":
                    thresholds.append(min(sorted_scores))  # Use lowest perplexity as fallback
                else:
                    thresholds.append(max(sorted_scores))  # Use highest logprob as fallback

        return thresholds

    else:
        raise ValueError("Method must be either 'fpr' or 'ratio'")

def run_threshold_selection(model, model_name, tokenizer, kg, method="perplexity", calibrate=True, multi_gpu=False, num_samples=100, 
                           output_dir=None, cache_path=None, fpr_values=[0.01, 0.05, 0.2], 
                           ratio_values=[0.7, 0.9, 0.95], use_vllm=False, explicit=False, use_batch=False):
    """
    Run the complete threshold selection process including evaluation and visualization.
    Now with consistent vLLM support.

    Args:
        model: HuggingFace model or vLLM LLM instance.
        tokenizer: Corresponding tokenizer.
        kg: KnowledgeGraph instance.
        method (str): "perplexity", "logprob", or "llm_score" to select the scoring method.
        multi_gpu (bool): Whether to use multi-GPU.
        num_samples (int): Number of samples per type.
        output_dir (str or None): Output directory to save plots.
        cache_path (str or None): Path to directory for caching computed scores.
        fpr_values (list): FPR values for threshold computation.
        ratio_values (list): Ratio values for threshold computation.
        use_vllm (bool): Whether to use vLLM for computations.

    Returns:
        dict: Evaluation results.
    """
    logging.info(f"Starting threshold selection process with method={method}, vLLM={use_vllm}...")

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # First, get all the evaluation results including the positive and negative scores
    evaluation_results = evaluate_thresholds(
        model, model_name, tokenizer, kg, method, calibrate, multi_gpu, num_samples, output_dir=output_dir,
        cache_path=cache_path, use_vllm=use_vllm, explicit=explicit, use_batch=use_batch
    )

    # logging.info(f"AUC: {evaluation_results['auc']:.4f}")
    # logging.info(f"Best threshold: {evaluation_results['best_threshold']:.2f} (F1={evaluation_results['best_f1']:.4f})")
    logging.info(f"Accuracy: {evaluation_results['accuracy']:.4f}")

    logging.info("Threshold performance details:")
    for th in sorted(evaluation_results["thresholds"].keys()):
        res = evaluation_results["thresholds"][th]
        logging.info(f"Threshold {th:.2f}: Precision={res['precision']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}, Accuracy={res['accuracy']:.4f}")

    if output_dir:
        roc_path = os.path.join(output_dir, f"roc_curve_{method}.png")
        plot_roc_curve(evaluation_results, roc_path)
    else:
        plot_roc_curve(evaluation_results)

    # Define low FPR values of interest
    low_fpr_values = [0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    # Retrieve ROC data from evaluation_results
    fpr = evaluation_results["fpr"]
    tpr = evaluation_results["tpr"]
    roc_thresholds = evaluation_results["roc_thresholds"]

    # Interpolate TPR and threshold values for the specified low FPR values
    interp_tpr = np.interp(low_fpr_values, fpr, tpr)
    interp_thresholds = np.interp(low_fpr_values, fpr, roc_thresholds)

    # Create output content for the text file
    output_txt_content = "Low FPR values with corresponding TPR and thresholds:\n"
    output_txt_content += "FPR\tTPR\tThreshold\n"
    for f_val, t_val, thr_val in zip(low_fpr_values, interp_tpr, interp_thresholds):
        # For perplexity, we need to negate the threshold (since lower is better)
        if method == "perplexity":
            threshold_val = -thr_val
        else:
            threshold_val = thr_val
        output_txt_content += f"{f_val:.2f}\t{t_val:.3f}\t{threshold_val:.3f}\n"

    # Save the output to a txt file
    if output_dir:
        txt_file_path = os.path.join(output_dir, f"{method}_thresholds.txt")
    else:
        txt_file_path = f"{method}_thresholds.txt"
    with open(txt_file_path, "w") as file:
        file.write(output_txt_content)
    logging.info(f"Thresholds saved to {txt_file_path}")

    # Use the scores from evaluation_results for plotting distributions
    positive_scores = evaluation_results["positive_scores"]
    negative_scores = evaluation_results["negative_scores"]

    # Plot score distributions using already computed scores
    if output_dir:
        dist_path = os.path.join(output_dir, f"score_distribution_{method}.png")
        plot_score_distributions(positive_scores, negative_scores, method, 
                               evaluation_results["best_threshold"], dist_path)
    else:
        plot_score_distributions(positive_scores, negative_scores, method, 
                               evaluation_results["best_threshold"])
     
    if method == "llm_score":
        return evaluation_results

    # Get positive samples to analyze relation distribution
    positive_samples = evaluation_results["positive_samples"]

    # Analyze relation distribution for various low FPR thresholds
    logging.info("Analyzing relation distribution at low FPR thresholds...")
    analysis_results = {}

    for fpr_value, threshold in zip(low_fpr_values, interp_thresholds):
        if fpr_value >= 0.2:  # Only analyze for FPR < 0.2
            continue

        # If using perplexity, we need to negate the threshold (since lower is better)
        if method == "perplexity":
            analysis_threshold = -threshold
        else:
            analysis_threshold = threshold

        top_k = 7
        analysis_result = analyze_relation_distribution(
            kg, positive_samples, positive_scores, method, 
            analysis_threshold, fpr_value, top_k, output_dir
        )

        analysis_results[fpr_value] = analysis_result
        logging.info(f"FPR={fpr_value:.2f}, TPR={interp_tpr[low_fpr_values.index(fpr_value)]:.3f}:")
        logging.info(f"  Correct positives: {analysis_result['total_correct_positives']}")
        logging.info(f"  Unique relations: {analysis_result['unique_relations']}")
        logging.info(f"  Uniformity ratio: {analysis_result['uniformity_ratio']:.3f} (1.0 = perfectly uniform)")
        logging.info(f"  Top relations: {analysis_result['top_relations']}")

    # Create a summary plot for relation uniformity
    fpr_list = [fpr for fpr in analysis_results.keys()]
    uniformity_list = [result['uniformity_ratio'] for result in analysis_results.values()]
    rel_variety_list = [result['unique_relations'] for result in analysis_results.values()]

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_list, uniformity_list, 'b-', marker='o', label='Uniformity Ratio')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('FPR')
    plt.ylabel('Uniformity Ratio')
    plt.title('Relation Distribution Uniformity at Different FPR Thresholds')
    plt.legend()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "relation_uniformity.png"))

    plt.show()

    # Add the relation analysis to the evaluation results
    evaluation_results["relation_analysis"] = analysis_results

    # Compute and output confidence thresholds based on two methods
    logging.info("Computing confidence thresholds for score-to-confidence conversion...")

    # Method 1: Based on fixed FPR values
    fpr_thresholds = compute_confidence_thresholds(evaluation_results, method="fpr", fpr_values=fpr_values)

    logging.info(f"Confidence thresholds based on FPR values {fpr_values}:")
    for i, (fpr_val, threshold) in enumerate(zip(fpr_values, fpr_thresholds)):
        level_name = ["low", "medium", "high"][i]
        logging.info(f"  {level_name.capitalize()} confidence (FPR={fpr_val}): {threshold:.4f}")

    # Method 2: Based on positive ratio
    ratio_thresholds = compute_confidence_thresholds(evaluation_results, method="ratio", ratio_values=ratio_values)

    logging.info(f"Confidence thresholds based on positive ratio values {ratio_values}:")
    for i, (ratio_val, threshold) in enumerate(zip(ratio_values, ratio_thresholds)):
        level_name = ["low", "medium", "high"][i]
        logging.info(f"  {level_name.capitalize()} confidence (ratio={ratio_val}): {threshold:.4f}")

    # Save thresholds to a file
    if output_dir:
        thresholds_path = os.path.join(output_dir, "confidence_thresholds.txt")
        with open(thresholds_path, "w") as f:
            f.write(f"Confidence thresholds for converting {method} scores to confidence scores (0-3):\n\n")

            f.write("Method 1: Based on fixed FPR values\n")
            f.write(f"FPR values: {fpr_values}\n")
            f.write(f"Thresholds: {[round(t, 6) for t in fpr_thresholds]}\n\n")

            f.write("Method 2: Based on positive ratio in score distribution\n")
            f.write(f"Ratio values: {ratio_values}\n")
            f.write(f"Thresholds: {[round(t, 6) for t in ratio_thresholds]}\n\n")

            if method == "perplexity":
                f.write("Usage instructions (perplexity - lower is better):\n")
                f.write("- For perplexity scores > thresholds[0]: confidence = 0\n")
                f.write("- For thresholds[1] < perplexity scores <= thresholds[0]: confidence = 1\n")
                f.write("- For thresholds[2] < perplexity scores <= thresholds[1]: confidence = 2\n")
                f.write("- For perplexity scores <= thresholds[2]: confidence = 3\n")
            else:
                f.write("Usage instructions (logprob/llm_score - higher is better):\n")
                f.write("- For scores < thresholds[0]: confidence = 0\n")
                f.write("- For thresholds[0] <= scores < thresholds[1]: confidence = 1\n")
                f.write("- For thresholds[1] <= scores < thresholds[2]: confidence = 2\n")
                f.write("- For scores >= thresholds[2]: confidence = 3\n")

        logging.info(f"Confidence thresholds saved to {thresholds_path}")

    # Add thresholds to evaluation results
    evaluation_results["confidence_thresholds"] = {
        "fpr_based": {
            "values": fpr_values,
            "thresholds": fpr_thresholds
        },
        "ratio_based": {
            "values": ratio_values,
            "thresholds": ratio_thresholds
        }
    }

    return evaluation_results

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    # Parse command-line arguments
    args = parse_args()

    # 1) Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {args.config}")

    # 2) Extract model and finetuning configuration
    hf_token = config.get("huggingface", {}).get("hf_token", None)
    model_name = config["model"]["name"]
    max_seq_length = config["model"].get("max_seq_length", 2048)
    checkpoint_path = config["model"].get("checkpoint_path", "")
    fpr_values = config.get("fpr_values", [0.01, 0.05, 0.2])
    ratio_values = config.get("ratio_values", [0.7, 0.9, 0.95])
    num_samples = config.get("num_samples", 100)

    # Get vLLM configuration
    use_vllm = config["model"].get("use_vllm", False)
    explicit = config["model"].get("explicit", False)
    vllm_kwargs = config["model"].get("vllm_kwargs", {})
    use_batch = config["model"].get("use_batch", False)
    merge_lora = config["model"].get("merge_lora", False)
    save_merged_model_path = config["model"].get("save_merged_model_path", "")
    if merge_lora:
        # Download repo locally
        local_lora_path = snapshot_download(
            repo_id=checkpoint_path,
            token=hf_token
        )

    instruct_finetuning = args.finetune
    use_lora = config["finetune"].get("use_lora", False)
    use_qlora = config["finetune"].get("use_qlora", False)
    if use_qlora:
        logging.info("use_qlora=True, the model will be loaded in 4-bit quantization mode (QLoRA).")

    multi_gpu = config["model"].get("multi_gpu", False)

    # 3) Load model and tokenizer using our pre-implemented load_model
    model, tokenizer = load_model(
        model_name=model_name,
        checkpoint_path=local_lora_path if merge_lora else checkpoint_path,
        hf_token=hf_token,
        multi_gpu=multi_gpu,
        instruct_finetuning=instruct_finetuning,
        use_lora=use_lora,
        load_in_4bit=use_qlora,
        use_vllm=use_vllm,
        vllm_kwargs=vllm_kwargs,
        merge_lora=merge_lora,
        save_merged_model_path=save_merged_model_path
    )
    logging.info(f"Model and tokenizer loaded for '{model_name}' with vLLM={use_vllm} and batch={use_batch}")

    # 4) Load KnowledgeGraph
    kg_name = config.get("kg", {}).get("name", "codex-medium")
    logging.info(f"Loading KnowledgeGraph '{kg_name}'...")
    kg = KnowledgeGraph(kg_name)
    kg.load(calculate_close_relation=True, mapping_method='bert', m=10, batch_size=16, bert_model="distilbert-base-uncased")

    # 5) Select metric method: "perplexity" or "logprob" (default: perplexity)
    method = config.get("metric", "perplexity")

    # 6) Get cache path from config
    cache_path = config.get("cache_path", None)
    if cache_path:
        logging.info(f"Using score cache directory: {cache_path}")

    # 7) Run threshold selection process and generate evaluation plots
    output_dir = os.path.join(kg.data_path, kg.name, f"{kg.name}_{model_name}_threshold_selection_results")
    results = run_threshold_selection(
        model, model_name, tokenizer, kg, method, True, multi_gpu, 
        num_samples=num_samples, output_dir=output_dir, 
        cache_path=cache_path, fpr_values=fpr_values, ratio_values=ratio_values,
        use_vllm=use_vllm, explicit=explicit, use_batch=use_batch  # Pass vLLM flag to the function
    )

    logging.info(f"Recommended threshold ({method}): {results['best_threshold']:.2f}")


if __name__ == "__main__":
    main()
