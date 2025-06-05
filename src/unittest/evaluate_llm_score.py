import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['HF_HOME'] = './output/model/pretrain'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import yaml
import logging
import argparse
import random
import torch
from tqdm import tqdm
from collections import defaultdict
import re

from utils.seed import set_seed
from models.model_loader import load_model
from src.train.trainer import instruct_finetune
from src.data.knowledge_graph import KnowledgeGraph
from evaluation.evaluator import query_llm_for_relations

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM Query, Finetuning, or Subgraph Construction.")
    parser.add_argument("--config", type=str, default="config/config.yml",
                        help="Path to the config YAML file (default: config/config.yml)")
    parser.add_argument("--finetune", action="store_true", default=False,
                        help="If set, run finetuning mode.")
    parser.add_argument("--generate_gt", action="store_true", default=False,
                        help="If set, generate ground truth triples.")
    return parser.parse_args()

def save_triples(triples_list: list, dir: str, output_name: str):
    """
    Save the extracted subgraph to a file, including confidence scores and optional IDR name.

    Args:
        triples_list (list): List of triples (head, relation, tail, confidence).
        dir (str): Directory to save the output file.
        output_name (str): Base file name to save the subgraph.
    """
    os.makedirs(dir, exist_ok=True)
    output_file = os.path.join(dir, output_name) 

    # Save the triples and confidence scores to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for head, relation, tail, conf in triples_list:
            f.write(f"{head},{relation},{tail},{conf}\n")  # Store confidence score

    logging.info(f"Saved triples to {output_file}.")

def load_triples(dir: str, output_name: str):
    """
    Load triples from a CSV file, including confidence scores.

    Args:
        dir (str): Directory to save the output file.
        output_name (str): Base file name to save the subgraph.
    """
    output_file = os.path.join(dir, output_name) 
    if not os.path.exists(output_file):
        logging.error(f"File not found: {output_file}")
        return []
    
    triples_list = []
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 4:
                head, relation, tail, conf = parts
                try:
                    conf = float(conf)  # Convert confidence to float
                    triples_list.append((head, relation, tail, conf))
                except ValueError:
                    logging.warning(f"Skipping line due to invalid confidence value: {line.strip()}")
            else:
                logging.warning(f"Skipping malformed line: {line.strip()}")
    
    logging.info(f"Loaded {len(triples_list)} triples from {output_file}.")
    return triples_list

def extract_triple_and_score(response, entity1, conf_level):
    """
    Extracts relationships and confidence scores from the model's response.
    Filters out relations with confidence below the given threshold.

    Args:
        response (str): The raw text output from the model (relation, entity2, confidence).
        entity1 (str): Entity 1 of the triple.
        conf_level (int): Confidence score required to keep a relation.

    Returns:
        list: A list of valid (entity1, relation, entity2, confidence) pairs above the threshold.
    """
    response = '\n'.join(response.rsplit('\n', 1)[:-1]) + '\n'
    response = response.strip()
    response = re.sub(r"\s+", " ", response)  # Normalize spaces

    # Extract all relation-confidence pairs
    matches = re.findall(r"RELATIONSHIP:\s*([/\w_-]+)\s*;\s*ENTITY2:\s*([^;]+)\s*;\s*CONFIDENCE:\s*(\d+)", response, re.IGNORECASE)


    valid_relations = []
    for rel, entity2, conf in matches:
        try:
            conf_int = int(conf)
            if conf_level <= conf_int <=1:
                valid_relations.append((entity1, rel, entity2, conf_int))
        except ValueError:
            continue  # Ignore invalid confidence values

    return valid_relations

def sample_from_llm(model, entity1: str, entity_options:set, relation_options: set, correct: tuple, conf_level: int, max_new_tokens: int=100):
    system_message = (
        "You are an expert in knowledge graphs. Your task is to analyze potential relationships "
        "between a given entity 1 and a set of entity 2 options. "
        "A set of known relation options is given for you to choice. "
        "Your response should be structured, precise, and strictly follow the required format."
    )

    user_message = (
        f"### Task:\n"
        f"For each relation in the relation options, select **one** entity from the provided entity 2 options that you have a confidence score of {conf_level}. "
        f"If no suitable entity is found, do not select any.\n\n"
        f"- **Entity 1**: '{entity1}'\n\n"
        f"### Entity 2 Options:\n"
        f"{entity_options}\n\n"
        f"### Confidence score Definitions:\n"
        f"- **0**: No meaningful relationship.\n"
        f"- **1**: Weak relationship, minor semantic association.\n"
        f"- **2**: Strong relationship, likely correct.\n"
        f"- **3**: Definite and unambiguous relationship.\n\n"
        f"### Relation Options:\n"
        f"{relation_options}\n\n"
        f"**Your response must be formatted exactly as follows:**\n"
        f"RELATIONSHIP: <relation>; ENTITY2: <chosen_entity>; CONFIDENCE: <score>.\n"
        f"Provide answers only for relations where a suitable entity 2 is found with a confidence level of {conf_level}.\n\n"
        f"**Example Output:**\n"
        f"RELATIONSHIP: hasGender; ENTITY2: Germany; CONFIDENCE: 0.\n"
        f"RELATIONSHIP: playsFor; ENTITY2: FC Barcelona; CONFIDENCE: 3.\n"
    )

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

    # # Define stopping tokens (EOS and end-of-turn ID)
    # terminators = [
    #     tokenizer.eos_token_id,
    #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
    # ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=terminators,  # Correct termination handling
            temperature=0.3,  # Lower randomness
            top_k=20,  # Reduce noise
            top_p=0.85  # Balance coherence and diversity
        )

    # Extract only the newly generated response
    response_tokens = outputs[0][input_ids.shape[-1]:]
    raw_response = tokenizer.decode(response_tokens, skip_special_tokens=True)    
    return extract_triple_and_score(raw_response, entity1, conf_level)

def sample_incorrect_from_random(correct_triple: tuple, relation_list: list):
    """
    Generate incorrect sample from a random incorrect relationship.
    """
    head, rel, tail, conf = correct_triple
    false_rel = random.sample([r for r in relation_list if r != rel], 1)[0]
    return (head, false_rel, tail, 0)

def _generate_prompt_options(kg_set: list, not_include: str=None):
    """
    Generate LLM prompt options given a set of entities and relation.
    """
    return "\n".join(
        [f"- {i.replace('_', ' ').replace('/', ' / ')}" for i in kg_set if i != not_include]
    )

def _generate_entity_set(kg: KnowledgeGraph, split='train'):
    """
    Generate entity sets based on train/val/test splitting.

    Args:
        kg (KnowledgeGraph): A KnowledgeGraph object.
        split (str): The split portion of triples to generate.
    Returns:
        head_set: A set of head entities.
        head_set: A set of tail entities.
        triple_dict: A dictionary where each key denotes a head entity and each value denotes a list of triples.
    """
    head_set = set()
    tail_set = set()
    triple_dict = defaultdict(list)
    for head, relation, tail in kg.triples[split]:
        head_set.add(head)
        tail_set.add(tail)
        triple_dict[head].append((head, relation, tail))
    return head_set, tail_set, triple_dict

def evaluate_target_llm(model, tokenizer, kg, num, save_dir, args, multi_gpu=False):
    # Change this after relation can be filtered out
    relation_list = sorted(set(kg.relations.values()))
    relation_descs = _generate_prompt_options(relation_list)
    entity1_set, entity2_set, triple_dict = _generate_entity_set(kg, 'train')
    
    # import pdb
    # pdb.set_trace()
    entity1_list = sorted(entity1_set)
    entity1_to_test = random.sample(entity1_list, num)

    # 1) Get Samples that we want to evaluate on target LLM.
    triples1 = []
    triples2 = []
    triples3 = []
    triples4 = []
    if args.generate_gt:
        logging.info(f"Generate ground truth triples.")
        for entity1 in tqdm(entity1_to_test):
            # 1.1) Sample a set of entities as possible entity 2 options.
            entity2_list = random.sample(list(entity2_set), 3)
            entity2_descs = _generate_prompt_options(entity2_list, not_include=entity1)

            # # 1.2) Evaluate the triple with definite and unambiguous relationships.
            # # # Generate from llm -> Currently still have hullucination
            # triple = sample_from_llm(model, entity1, entity2_descs, relation_descs, 3)
            # # # Generate directly from training set -> 100% correct
            triple = random.sample(triple_dict[entity1], 1)
            triple= [tuple(t+ (3,)) for t in triple]
            triples1.extend(triple)
            correct = triple[0]

            # Ablattion study: Randomly select the incorrect relationship
            triple = sample_incorrect_from_random(triple[0], relation_list)
            triple = [triple]
            triples2.extend(triple)

            # Ablation study: Let LLM select the incorrect relationship
            true_entity2_descs =  _generate_prompt_options([triple[0][2]])
            triple = sample_from_llm(model, triple[0][0], true_entity2_descs, relation_descs, correct, 0)
            triples3.extend(triple)

            # 1.3) Evaluate the triple that has no meaningful relationship.
            triple = sample_from_llm(model, entity1, entity2_descs, relation_descs, correct, 0)
            triples4.extend(triple)
        
        # save_triples(triples1, save_dir, "sampled_3.csv")
        # save_triples(triples2, save_dir, "sampled_0_ab1_raw.csv")
        # save_triples(triples3, save_dir, "sampled_0_ab2_raw.csv")
        # save_triples(triples4, save_dir, "sampled_0_raw.csv")
    else:
        triples1 = load_triples(save_dir, "sampled_3.csv")
        triples2 = load_triples(save_dir, "sampled_0_ab1.csv")
        triples4 = load_triples(save_dir, "sampled_0.csv")
    
    # 2) Evlauate the sampled triples using target LLM
    triples_list = [("evaluate_3_tmp.csv", triples1), ("evaluate_0_ab1_tmp.csv", triples2), ("evaluate_0_tmp.csv", triples4)]
    for filename, triples in triples_list:
        eval_triples = []
        for entity1, relation, entity2, _ in tqdm(triples):
            _relation = _generate_prompt_options([relation])
            out = query_llm_for_relations(model, tokenizer, entity1, entity2, _relation, multi_gpu=False, threshold=0)
            if not out: continue # Sometimes the output can be an empty list.
            _, conf = out[0]
            # print(tmp, relation)
            eval_triples.append((entity1, relation, entity2, conf))

        save_triples(eval_triples, save_dir, filename)

    # # An alternative to get definite and unambiguous relationships based on KG's topology.
    # eval_triples = []
    # for entity1 in tqdm(entities_to_test):
    #     potential_neighbors = list(kg.get_k_hop_neighbors(entity1, k=2, sample=True, max_sample=3))
    #     print(potential_neighbors)
    #     for nbr in potential_neighbors:
    #         out = query_llm_for_relations(model, tokenizer, entity1, nbr, relation_descs, multi_gpu=False, threshold=3)
    #         if not out: continue # Sometimes the output can be an empty list.
    #         rel, conf = out[0]
    #         # print(tmp, relation)
    #         eval_triples.append((entity1, rel, nbr, conf))

    # save_triples(eval_triples, save_dir, "evaluate_3_tmp.csv")



if __name__ == "__main__":
    args = parse_args()

    # 1) Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {args.config}")

    # 2) Fix random seed
    seed_val = config.get("seed", 42)
    set_seed(seed_val)
    logging.info(f"Random seed set to {seed_val}")

    # 3) Multi-GPU control
    multi_gpu = config["model"].get("multi_gpu", False)
    if multi_gpu:
        logging.info("MULTI-GPU mode enabled (device_map='auto').")
    else:
        logging.info("SINGLE-GPU or CPU mode (no device_map).")

    # 5) Load model & tokenizer
    hf_token = config.get("huggingface", {}).get("hf_token", None)
    print(hf_token)
    model_name = config["model"]["name"]
    max_seq_length = config["model"].get("max_seq_length", 2048)
    checkpoint_path = config["model"].get("checkpoint_path", "")
        
    model, tokenizer = load_model(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        hf_token=hf_token,
        multi_gpu=multi_gpu,
        instruct_finetuning=False,
        use_lora=False,
        load_in_4bit=False
    )
    logging.info(f"Model and tokenizer loaded for '{model_name}'")


    # 6) Load Knowledge Graph
    kg_name = config["dataset"]["kg_name"]
    kg = KnowledgeGraph(kg_name)
    kg.load(degree_threshold=10)
    logging.info(f"Loaded knowledge graph '{kg_name}'")

    # 7) Evaluate Target LLM
    save_dir = config["output"]["unit_test_dir"]
    evaluate_target_llm(model, tokenizer, kg, 100, save_dir, args)