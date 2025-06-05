"""
Main entry point for Target LLM Query or Finetuning.

This script loads configuration parameters from a YAML file and sets up the
environment (e.g., fixed random seed, CUDA options). It then loads a Hugging Face LLM
via the model_loader module. Depending on the configuration, it either performs finetuning
(using a finetuning dataset in text format) or runs queries on the model
(using a query dataset in JSON format).

Additionally, it constructs a **k-hop subgraph** from a target triple (A, relation, B)
by querying an LLM for relation confidence scores.

Usage Examples:
    # To finetune the model:
    $ python src/main_finetune.py --config config/config.yml --finetune

    # To run queries:
    $ python src/main_finetune.py --config config/config.yml --query

    # To construct a subgraph:
    $ python src/main_finetune.py --config config/config_subgraph.yml --subgraph

    # To perform unlearning:
    $ deepspeed --num_gpus=2 src/main_finetune.py --config config/config_unlearn.yml --unlearn

    # To judge LLM:
    $ python src/main_finetune.py --config config/config_judge.yml --judge

    # To test LLM utility:
    $ python src/main_finetune.py --config config/config_utility.yml --utility
"""

import os
os.environ["SET_CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["VLLM_WORKER_USE_RAY"] = "0"
os.environ["VLLM_ENABLE_STREAM_WORKER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Set environment variables for NCCL stability
os.environ["NCCL_TIMEOUT"] = "1800"  # 30 minutes timeout instead of 10
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_TIMEOUT"] = "0"
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800"
import sys
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import threading

def silent_thread_excepthook(args):
    print(f"[Thread Error Caught] {args.exc_type.__name__}: {args.exc_value}")

threading.excepthook = silent_thread_excepthook
import yaml
import logging
import argparse
import wandb
import random
import json
import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from utils.seed import set_seed
from models.model_loader import load_model
from src.train.trainer import instruct_finetune, dpo_finetune
from evaluation.evaluator import load_query_dataset, evaluate_model, save_evaluation_results
from evaluation.evaluator import load_existing_subgraphs, save_subgraph_to_json, is_triple_already_processed, batch_evaluate_triples, batch_evaluate_triples_entropy, batch_score_triples_entropy, save_entropy_results_to_json
from utils.utils import compute_relation_accuracy, get_top_k_relations
from src.data.knowledge_graph import KnowledgeGraph
from evaluation.evaluator import construct_subgraph, save_subgraph, precompute_relation_options, query_llm_for_relations_by_perplexity, estimate_llm_query_complexity, process_custom_relations, save_valid_subgraph, is_valid_subgraph
from evaluation.llm_evaluator import evaluate_all_subgraphs, process_unlearn_triples
from src.data.text_conversion import generate_finetune_dataset, generate_finetune_dataset_DPO
from src.unlearning.unlearn import perform_unlearning
from huggingface_hub import snapshot_download
from src.utility.utility import *
from src.utility.eval_utility import *
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM Query, Finetuning, or Subgraph Construction.")
    parser.add_argument("--config", type=str, default="config/config.yml",
                        help="Path to the config YAML file (default: config/config.yml)")
    parser.add_argument("--finetune", action="store_true", default=False,
                        help="If set, run finetuning mode.")
    parser.add_argument("--subgraph", action="store_true", default=False,
                        help="If set, construct a subgraph from a target triple.")
    parser.add_argument("--unlearn", action="store_true", default=False,
                        help="If set, perform unlearning on the model.")
    parser.add_argument("--query", action="store_true", default=False,
                        help="If set, perform querying on the model.")
    parser.add_argument("--judge", action="store_true", default=False,
                        help="If set, perform judging on the model.")
    parser.add_argument("--utility", action="store_true", default=False,
                    help="If set, perform utility evaluation on the model.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logging.info(f"Loaded configuration from {args.config}")

    # 2) Fix random seed
    seed_val = config.get("seed", 42)
    set_seed(seed_val)
    logging.info(f"Random seed set to {seed_val}")

    # 3) Determine mode
    if args.finetune:
        mode_str = "Finetuning"
    elif args.subgraph:
        mode_str = "Subgraph Construction"
    elif args.unlearn:
        mode_str = "Unlearning"
    elif args.judge:
        mode_str = "LLM Judging"
    elif args.utility:
        mode_str = "Utility Evaluation"
    else:
        mode_str = "Query"
    logging.info(f"Operation mode: {mode_str}")

    # 4) Multi-GPU control
    multi_gpu = config["model"].get("multi_gpu", False)
    if multi_gpu:
        logging.info("MULTI-GPU mode enabled (device_map='auto').")
    else:
        logging.info("SINGLE-GPU or CPU mode (no device_map).")

    # 5) Load model & tokenizer
    hf_token = config.get("huggingface", {}).get("hf_token", None)
    model_name = config["model"]["name"]
    max_seq_length = config["model"].get("max_seq_length", 2048)
    checkpoint_path = config["model"].get("checkpoint_path", "")

    # Added support for vLLM
    use_vllm = config["model"].get("use_vllm", False)
    vllm_kwargs = config["model"].get("vllm_kwargs", {})
    merge_lora = config["model"].get("merge_lora", False)
    save_merged_model_path = config["model"].get("save_merged_model_path", "")
    use_huggingface_self_trained_model = config["model"].get("use_huggingface_self_trained_model", False)

    if use_vllm:
        logging.info("Using vLLM for accelerated inference.")
    if use_huggingface_self_trained_model:
        # Download repo locally
        local_path = snapshot_download(
            repo_id=checkpoint_path,
            token=hf_token
        )

    instruct_finetuning = args.finetune
    use_lora = config["finetune"].get("use_lora", False)
    use_qlora = config["finetune"].get("use_qlora", False)
    if use_qlora:
        logging.info("use_qlora=True, the model will be loaded in 4-bit quantization mode (QLoRA).")

    # Initialize distributed environment for DeepSpeed
    if args.local_rank != -1:
        if not torch.distributed.is_initialized():
            # Set cuda device
            torch.cuda.set_device(args.local_rank)
            # Initialize process group
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))
        is_master = args.local_rank == 0
    else:
        is_master = True

    # When loading config, ensure DeepSpeed settings are properly set
    if args.local_rank != -1:
        config["model"]["multi_gpu"] = True
        config["model"]["use_deepspeed"] = True

    # Add DeepSpeed configuration 
    use_deepspeed = config["model"].get("use_deepspeed", False) 
    deepspeed_config_type = config["model"].get("deepspeed_config_type", "zero2")
    gradient_accumulation_steps = config["model"].get("gradient_accumulation_steps", 4)

    if not args.judge:
        # Load model with updated parameters including vLLM support
        model, tokenizer = load_model(
            model_name=model_name,
            checkpoint_path= local_path if use_huggingface_self_trained_model else checkpoint_path,
            hf_token=hf_token,
            multi_gpu=multi_gpu,
            use_deepspeed=use_deepspeed,
            instruct_finetuning=instruct_finetuning,
            use_lora=use_lora,
            load_in_4bit=use_qlora,
            use_vllm=use_vllm,
            vllm_kwargs=vllm_kwargs,
            merge_lora=merge_lora,
            save_merged_model_path=save_merged_model_path
        )
        logging.info(f"Model and tokenizer loaded for '{model_name}'")

    # 6) Perform Finetuning
    if args.finetune:
        # Load Knowledge Graph & Generate Finetune Dataset if not exists
        logging.info("Loading KG and datasets...")
        kg_name = config["dataset"]["kg_name"]
        kg = KnowledgeGraph(kg_name)
        kg.load(degree_threshold=10)

        dataset_dir = config["dataset"]["finetune_dir"]
        output_dir = config["output"]["model_dir"]
        epochs = config["finetune"]["epochs"]
        learning_rate = config["finetune"]["learning_rate"]
        batch_size = config["finetune"]["batch_size"]
        hub_model_id = config["huggingface"]["hub_model_id"]
        push_to_hub = config["huggingface"]["push_to_hub"]
        use_dpo = config["finetune"].get("use_dpo", False)
        beta = config["finetune"].get("beta", 0.1)

        # Initialize WandB
        wandb.init(
            project="llm_instruct_finetune",
            name=f"{kg_name}_{model_name}_finetune_epoch{epochs}_dpo{use_dpo}",
            config={
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "dataset_dir": dataset_dir,
                "use_qlora": use_qlora,
                "use_lora": use_lora
            }
        )

        if use_dpo:
            logging.info("Using DPO fine-tuning approach...")
            generate_finetune_dataset_DPO(kg, dataset_dir)

            model = dpo_finetune(
                model, tokenizer,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=int(epochs),
                lr=float(learning_rate),
                batch_size=int(batch_size),
                multi_gpu=multi_gpu,
                beta=beta,
                max_seq_length=max_seq_length,
                use_qlora=use_qlora,
                use_lora=use_lora,
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id if push_to_hub else None,
                hub_token=hf_token if push_to_hub else None
            )
        else:
            logging.info("Using INSTRUCTIONAL fine-tuning approach...")
            generate_finetune_dataset(kg, dataset_dir)
            model = instruct_finetune(
                model, tokenizer,
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=int(epochs),
                lr=float(learning_rate),
                batch_size=int(batch_size),
                multi_gpu=multi_gpu,
                max_seq_length=max_seq_length,
                use_qlora=use_qlora,
                use_lora=use_lora,
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id if push_to_hub else None,
                hub_token=hf_token if push_to_hub else None
            )

        logging.info("Finetuning completed and model saved.")

        # Finish WandB tracking
        wandb.finish()

    # 7) Perform Unlearning
    elif args.unlearn:
        # Only have master process print certain logs
        if is_master:
            logging.info("Starting unlearning process...")

        if multi_gpu:
            if use_deepspeed:
                logging.info(f"MULTI-GPU mode enabled with DeepSpeed ({deepspeed_config_type}).")
                # Ensure the distributed environment is properly set
                if "MASTER_ADDR" not in os.environ:
                    os.environ["MASTER_ADDR"] = "localhost"
                if "MASTER_PORT" not in os.environ:
                    os.environ["MASTER_PORT"] = "29500"
            else:
                logging.info("MULTI-GPU mode enabled (device_map='auto') without DeepSpeed.")
        else:
            logging.info("SINGLE-GPU or CPU mode.")

        # Load necessary parameters for unlearning
        use_QA_unlearning = config["unlearn"].get("use_QA_unlearning", True)
        peft_config = config["unlearn"].get("peft_config", False)
        # unlearning_triples = config["unlearn"].get("triples", [])
        unlearning_method = config["unlearn"].get("method", "gradient_ascent")
        output_dir = config["output"]["unlearn_model_dir"]
        batch_size=int(config["unlearn"].get("batch_size", 4))
        epochs=int(config["unlearn"].get("epochs", 10))   
        learning_rate=float(config["unlearn"].get("learning_rate", 5e-5))  
        push_to_hub = config["huggingface"].get("push_to_hub", False)
        hub_model_id = config["huggingface"].get("hub_model_id", None)
        resume_from_checkpoint = config["unlearn"].get("resume_from_checkpoint", "None")

        # Load the knowledge graph
        kg_name = config["dataset"]["kg_name"]
        kg = KnowledgeGraph(kg_name)
        kg.load(degree_threshold=100)

        # Load unlearning triples from .json (extract keys)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if model_name in ["Qwen/Qwen2.5-7B-Instruct"]:
            unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_200_qwen.json")
        elif model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
            unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_100_llama.json")

        with open(unlearn_json_path, "r") as f:
            unlearn_data = json.load(f)

        # Convert JSON keys (string triple repr) to actual triple lists
        unlearning_triples = [eval(key) for key in unlearn_data.keys()]
        logging.info(f"Loaded {len(unlearning_triples)} unlearning triples from {unlearn_json_path}")

        # **Construct the retain set**
        logging.info("Constructing retain set...")

        # **Use all triples for retain set**
        train_triples = set(map(tuple, kg.triples["total"]))
        # Convert unlearning triples to a set for efficient removal
        unlearning_set = set(map(tuple, unlearning_triples))
        # Remove unlearning triples from training set
        remaining_triples = list(train_triples - unlearning_set)
        # Select retain triples: size = epochs * len(unlearning_triples)
        retain_size = epochs * len(unlearning_triples)

        if len(remaining_triples) < retain_size:
            logging.warning("Not enough triples left for retain set. Using all available.")
            retain_triples = remaining_triples
        else:
            retain_triples = random.sample(remaining_triples, retain_size)  # **Randomly sample retain triples**

        logging.info(f"Retain set constructed with {len(retain_triples)} triples.")

        # if unlearning_method == "noisy_training":
        noise_multiplier = float(config["unlearn"].get("noise_multiplier", 1e-3))
        clip_norm = config["unlearn"].get("clip_norm", 1.0)

        # Execute unlearning
        model = perform_unlearning(
            model, 
            model_name,
            tokenizer,
            unlearning_triples=unlearning_triples,
            retain_triples=retain_triples,
            method=unlearning_method,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            noise_multiplier=noise_multiplier,
            clip_norm=clip_norm,
            use_QA_unlearning=use_QA_unlearning,
            peft_config=peft_config,
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
            use_deepspeed=use_deepspeed,
            deepspeed_config_type=deepspeed_config_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            resume_from_checkpoint=resume_from_checkpoint
        )

        logging.info("Unlearning process completed. Model updated.")

    # 8) Perform Querying
    elif args.query:
        logging.info("Quering testing triples...")

        # Load Knowledge Graph
        kg_name = config["dataset"]["kg_name"]
        kg = KnowledgeGraph(kg_name)
        kg.load(degree_threshold=10)
        logging.info(f"Loaded knowledge graph '{kg_name}'")

        # Check if a target triple is provided in the config file; otherwise, compute and use the first triple
        target_triple = config["subgraph"]["subgraph_target_triple"]
        logging.info(f"Using target triple defined in config: {target_triple}")

        A, relation, B = target_triple
        logging.info(f"Using target triple: ({A}, {relation}, {B})")
        

        relation_descs = precompute_relation_options(kg)

        valid_relations = query_llm_for_relations_by_perplexity(model, tokenizer, A, B, relation_descs, multi_gpu=False, max_new_tokens=100)
        top_k = get_top_k_relations(valid_relations, k=10)
        logging.info(f"Top {len(top_k)} relations: {top_k}")

    # 9) Construct a Subgraph
    elif args.subgraph:
        logging.info("Starting subgraph construction...")
        if use_vllm:
            logging.info("Using vLLM acceleration for subgraph construction")

        # Load the knowledge graph
        kg_name = config["dataset"]["kg_name"]
        kg = KnowledgeGraph(kg_name)
        kg.load(degree_threshold=100)
        logging.info(f"Loaded knowledge graph '{kg_name}'")

        # Load configuration parameters
        k_hops = config["subgraph"]["k_hops"]
        thresholds = config["subgraph"]["thresholds"]
        max_estimated_queries = config["subgraph"].get("max_estimated_queries", 1000)
        min_subgraph_size = config["subgraph"].get("min_subgraph_size", 10)
        num_valid_subgraphs = config["subgraph"].get("num_valid_subgraphs", 10)
        use_query_relations = config["subgraph"].get("use_query_relations", False)
        times=config["subgraph"].get("times", 5)
        batch_size=config["subgraph"].get("batch_size", 2000)
        thresholds=config["subgraph"].get("thresholds", [0.5, 0.8, 0.9])
        confidence_threshold=config["subgraph"].get("confidence_threshold", 2)
        use_entropy=config["subgraph"].get("use_entropy", False)
        entropy_threshold=config["subgraph"].get("entropy_threshold", 1)

        # Select relation descriptions
        if use_query_relations and "query_relations" in config["subgraph"]:
            custom_relations = config["subgraph"]["query_relations"]
            relation_descs = process_custom_relations(custom_relations)
            logging.info(f"Using {len(custom_relations)} custom relations from config")
        else:
            relation_descs = precompute_relation_options(kg)
            logging.info("Using all relations from knowledge graph")

        # Prepare output path
        output_dir = os.path.dirname(config["output"]["subgraph_dir"])
        os.makedirs(output_dir, exist_ok=True)

        valid_subgraphs = []
        tried_triples = set()
        
        score_unlearn_triples = config["subgraph"].get("score_unlearn_triples", False)
        score_dir = os.path.dirname(config["output"]["score_dir"])
        unlearn_method = config["subgraph"].get("unlearn_method", "unknown_method")
        if score_unlearn_triples:
            json_output_path = os.path.join(score_dir, f"score_unlearn_triples_{config['dataset']['kg_name']}_{unlearn_method}.json")

            # Load unlearning triples from .json (extract keys)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if model_name in ["Qwen/Qwen2.5-7B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_200_qwen.json")
            elif model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_100_llama.json")
            with open(unlearn_json_path, "r") as f:
                unlearn_data = json.load(f)

            # Convert JSON keys (string triple repr) to actual triple lists
            unlearning_triples = [eval(key) for key in unlearn_data.keys()]
            logging.info(f"Loaded {len(unlearning_triples)} unlearning triples from {unlearn_json_path}")

            scores_results = batch_score_triples_entropy(
                    model, 
                    model_name,
                    tokenizer, 
                    unlearning_triples, 
                    batch_size=batch_size,
                    entropy_threshold=entropy_threshold,
                )
            
            save_entropy_results_to_json(scores_results, json_output_path)
            
            return

        # Handle specific target triple
        specific_target_triple = config["subgraph"].get("subgraph_target_triple", None)
        if specific_target_triple:
            json_output_path = os.path.join(output_dir, f"subgraphs_{config['dataset']['kg_name']}.json")
            A, relation, B = specific_target_triple
            triple_key = (A, relation, B)
            if not is_triple_already_processed(json_output_path, triple_key):
                subgraph = construct_subgraph(model, model_name, tokenizer, kg, triple_key,
                                            k=k_hops, times=times, confidence_threshold=confidence_threshold,
                                            thresholds=thresholds, entropy_threshold=entropy_threshold, batch_size=batch_size, multi_gpu=multi_gpu,
                                            relation_descs=relation_descs, use_vllm=use_vllm, use_entropy=use_entropy)
                if is_valid_subgraph(subgraph, min_subgraph_size):
                    valid_subgraphs.append((subgraph, triple_key))
                    save_subgraph_to_json(json_output_path, triple_key, subgraph)
                    logging.info(f"Saved valid subgraph for triple {triple_key}")
                else:
                    logging.info(f"Triple {triple_key} did not produce a valid subgraph")

        eval_unlearn_triples = config["subgraph"].get("eval_unlearn_triples", False)
        unlearn_method = config["subgraph"].get("unlearn_method", "unknown_method")
        if eval_unlearn_triples:
            json_output_path = os.path.join(output_dir, f"subgraphs_{config['dataset']['kg_name']}_{unlearn_method}.json")

            # Load unlearning triples from .json (extract keys)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if model_name in ["Qwen/Qwen2.5-7B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_200_qwen.json")
            elif model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_100_llama.json")
            with open(unlearn_json_path, "r") as f:
                unlearn_data = json.load(f)

            # Convert JSON keys (string triple repr) to actual triple lists
            unlearning_triples = [eval(key) for key in unlearn_data.keys()]
            logging.info(f"Loaded {len(unlearning_triples)} unlearning triples from {unlearn_json_path}")

            for triple in tqdm(unlearning_triples, desc="Constructing subgraphs for unlearned model"):
                A, relation, B = triple
                triple_key = (A, relation, B)
                tried_triples.add(triple_key)
                subgraph = construct_subgraph(model, model_name, tokenizer, kg, triple_key,
                                        k=k_hops, times=times, confidence_threshold=confidence_threshold,
                                        thresholds=thresholds, entropy_threshold=entropy_threshold, batch_size=batch_size, multi_gpu=multi_gpu,
                                        relation_descs=relation_descs, use_vllm=use_vllm, use_entropy=use_entropy)

                save_subgraph_to_json(json_output_path, triple_key, subgraph)
                logging.info(f"Saved subgraph for triple {triple_key}")

        # Construct additional subgraphs if needed
        if len(valid_subgraphs) < num_valid_subgraphs and not eval_unlearn_triples:
            full_text_triples = kg.generate_full_text_triples()
            random.shuffle(full_text_triples)
            triple_range = config["subgraph"].get("triple_range", [])
            if triple_range and len(triple_range) == 2:
                a, b = triple_range
                # Make sure a and b are within valid range
                a = max(0, a)
                b = min(len(full_text_triples), b)
                selected_triples = full_text_triples[a:b]
                logging.info(f"Using triples from range [{a}:{b}] out of {len(full_text_triples)} total triples")
            else:
                selected_triples = full_text_triples
                logging.info(f"Using all {len(full_text_triples)} triples")

            logging.info(f"Evaluating {len(selected_triples)} triples in parallel")

            json_output_path = os.path.join(output_dir, f"subgraphs_{config['dataset']['kg_name']}_range_{a}_{b}.json")

            new_selected_triples = []
            for triple_str in selected_triples:
                candidate = triple_str.split(",")
                if len(candidate) != 3:
                    continue
                new_selected_triples.append(candidate)

            if use_entropy:
                evaluate_result = batch_evaluate_triples_entropy(
                    model, 
                    model_name,
                    tokenizer, 
                    new_selected_triples, 
                    batch_size=batch_size,
                    entropy_threshold=entropy_threshold,
                )
                # Filter for high confidence triples (entropy <= entropy_threshold)
                high_confidence_triples = []
                for triple, entropy in zip(new_selected_triples, evaluate_result):
                    if entropy <= entropy_threshold:
                        high_confidence_triples.append(triple)
            else:
                evaluate_result = batch_evaluate_triples(
                    model, 
                    model_name,
                    tokenizer, 
                    new_selected_triples, 
                    times=times, 
                    batch_size=batch_size,
                    thresholds=thresholds,
                )
                # Filter for high confidence triples (confidence >= 2)
                high_confidence_triples = []
                for triple, confidence in zip(new_selected_triples, evaluate_result):
                    if confidence >= confidence_threshold:
                        high_confidence_triples.append(triple)
            
            logging.info(f"Found {len(high_confidence_triples)} high-confidence triples out of {len(new_selected_triples)} candidates")

            for triple in tqdm(high_confidence_triples, desc="Constructing subgraphs"):
                if len(valid_subgraphs) >= num_valid_subgraphs:
                    break
                A, relation, B = triple
                triple_key = (A, relation, B)
                if triple_key in tried_triples or is_triple_already_processed(json_output_path, triple_key):
                    continue
                tried_triples.add(triple_key)

                if use_query_relations:
                    estimated_queries, _ = estimate_llm_query_complexity(kg, triple_key, k_hops, len(custom_relations))
                else:
                    estimated_queries, _ = estimate_llm_query_complexity(kg, triple_key, k_hops)

                if estimated_queries <= max_estimated_queries:
                    subgraph = construct_subgraph(model, model_name, tokenizer, kg, triple_key,
                                            k=k_hops, times=times, confidence_threshold=confidence_threshold,
                                            thresholds=thresholds, entropy_threshold=entropy_threshold, batch_size=batch_size, multi_gpu=multi_gpu,
                                            relation_descs=relation_descs, use_vllm=use_vllm, use_entropy=use_entropy)
                    if is_valid_subgraph(subgraph, min_subgraph_size):
                        valid_subgraphs.append((subgraph, triple_key))
                        save_subgraph_to_json(json_output_path, triple_key, subgraph)
                        logging.info(f"Saved valid subgraph for triple {triple_key}")
                    else:
                        logging.info(f"Triple {triple_key} did not produce a valid subgraph")

        if not valid_subgraphs and not eval_unlearn_triples:
            raise ValueError("No valid subgraphs found.")

        logging.info(f"Subgraph construction finished. Saved to {json_output_path}")

    # 10) Judge LLM
    elif args.judge:
        judge_superficial = config["input"].get("judge_superficial", False)
        if judge_superficial:
            judge_superficial_path = config["input"].get("judge_superficial_path", "")
            output_superficial_path = config["output"].get("output_superficial_path", "")
            entropy_threshold=config["input"].get("entropy_threshold", 1)
            process_unlearn_triples(judge_superficial_path, entropy_threshold, output_superficial_path)
        
        else:
            judge_subgraph_path = config["input"].get("judge_subgraph_path", "")
            if os.path.exists(judge_subgraph_path):
                with open(judge_subgraph_path, "r", encoding="utf-8") as f:
                    all_subgraphs = json.load(f)

            if len(all_subgraphs) == 0:
                raise ValueError("No subgraphs available in the JSON file to evaluate.")

            eval_output_path = config["output"].get("evaluation_output_file", "./output/subgraph/yago3-10-entropy1/judge_output/judge_all.json")
            evaluate_all_subgraphs(all_subgraphs, eval_output_path, use_openai=True)

    # 11) Perform Utility Evaluation
    elif args.utility:
        logging.info("Starting utility evaluation...")

        # Load Knowledge Graph
        kg_name = config["dataset"]["kg_name"]
        kg = KnowledgeGraph(kg_name)
        kg.load(degree_threshold=100)
        logging.info(f"Loaded knowledge graph '{kg_name}' with {len(kg.triples['total'])} triples.")

        run_normal = config["utility"].get("run_normal", True)
        run_mmlu = config["utility"].get("run_mmlu", False)
        run_bbh = config["utility"].get("run_bbh", False)
        run_triviaqa = config["utility"].get("run_triviaqa", False)
        run_truthfulqa = config["utility"].get("run_truthfulqa", False)
        run_fluency = config["utility"].get("run_fluency", False)

        if run_normal:
            # Load unlearning triples
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if model_name in ["Qwen/Qwen2.5-7B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_200_qwen.json")
            elif model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
                unlearn_json_path = os.path.join(script_dir, "unlearning", "unlearn_data", kg_name, f"{kg_name}_unlearn_100_llama.json")
            with open(unlearn_json_path, "r") as f:
                unlearn_data = json.load(f)

            unlearning_triples = [eval(key) for key in unlearn_data.keys()]
            logging.info(f"Loaded {len(unlearning_triples)} unlearning triples from {unlearn_json_path}")
            
            # Set sampling config
            sample_size = config["utility"]["sample_size"]
            loc = config["utility"]["loc"]
            loc_num = config["utility"]["loc_num"]

            # Sample or load non-target utility triples
            utility_dir = os.path.join(script_dir, "utility")
            os.makedirs(utility_dir, exist_ok=True)
            if model_name in ["Qwen/Qwen2.5-7B-Instruct"]:
                sample_save_path = os.path.join(utility_dir, f"{kg_name}_utility_samples_loc_{loc}_k_{loc_num}_qwen.json")
            elif model_name in ["meta-llama/Llama-3.1-8B-Instruct"]:
                sample_save_path = os.path.join(utility_dir, f"{kg_name}_utility_samples_loc_{loc}_k_{loc_num}_llama.json")

            if os.path.exists(sample_save_path):
                logging.info(f"Loading pre-sampled utility triples from {sample_save_path}")
                utility_triples = load_sampled_triples(sample_save_path)
            else:
                logging.info(f"Sampling {sample_size} utility triples with loc={loc}, loc_num={loc_num}...")
                utility_triples = sample_nontarget_triples(kg, unlearning_triples, sample_size, loc, loc_num)
                save_sampled_triples(utility_triples, sample_save_path)
                logging.info(f"Saved sampled utility triples to {sample_save_path}")
            
            # Score utility triples
            unlearn_method = config["unlearn"].get("unlearn_method", "no_unlearn")
            checkpoint_name = config["unlearn"].get("checkpoint_name", "no_checkpoint")
            score_dir = os.path.join(config["output"]["score_dir"])
            os.makedirs(score_dir, exist_ok=True)
            utility_score_path = os.path.join(score_dir, f"score_utility_triples_{kg_name}_{checkpoint_name}_loc_{loc}_k_{loc_num}.json")

            logging.info(f"Scoring utility triples using model: {config['model']['name']}...")

            utility_scores = batch_score_utility_triples_entropy(
                        model=model, 
                        model_name=model_name,
                        tokenizer=tokenizer, 
                        triples=utility_triples, 
                        batch_size=config["utility"].get("batch_size", 2000)
                    )
                
            save_utility_triples_results_to_json(utility_scores, utility_score_path)
                
            logging.info(f"Saved utility triple scores to {utility_score_path}")

        
        if run_mmlu or run_bbh or run_triviaqa or run_truthfulqa or run_fluency:
            checkpoint_name = config["unlearn"].get("checkpoint_name", "no_checkpoint")
            output_dir = os.path.join(config["output"]["score_dir"], checkpoint_name)
            os.makedirs(output_dir, exist_ok=True)
        
            targets = config["utility"].get("targets", [])
            all_summary_results = []
            for target in targets:

                utility_dataset_dir = os.path.join(config["utility"]["utility_dataset_dir"], target)

                summary_results = {
                "model_name": model_name,
                "checkpoint_path": local_path if use_huggingface_self_trained_model else checkpoint_path,
                "use_prompt": True,
                "batch_size": config["utility"].get("batch_size", 2000),
                "use_vllm": use_vllm,
                "target": target,
                "results": {}}
                
                if run_mmlu:
                    logging.info("Starting MMLU evaluation...")
                    try:
                        mmlu_dataset_path = os.path.join(utility_dataset_dir, "retain_mmlu.json")
                        mmlu_dataset = load_dataset(mmlu_dataset_path)
                        mmlu_output_path = os.path.join(output_dir, "mmlu.json")
                        
                        mmlu_accuracy = eval_mmlu(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=mmlu_dataset,
                            batch_size=config["utility"].get("batch_size", 2000),
                            output_result_dir=mmlu_output_path,
                            use_vllm=use_vllm
                        )
                        
                        summary_results["results"]["mmlu"] = {
                            "accuracy": float(mmlu_accuracy)
                        }
                        logging.info(f"MMLU Evaluation complete! Accuracy: {mmlu_accuracy:.4f}")
                    except Exception as e:
                        logging.error(f"Error in MMLU evaluation: {e}")
                        summary_results["results"]["mmlu"] = {
                        "error": str(e)}

                if run_bbh:
                    logging.info("Starting BBH evaluation...")
                    try:
                        bbh_dataset_path = os.path.join(utility_dataset_dir, "retain_bbh.json")
                        bbh_dataset = load_dataset(bbh_dataset_path)
                        bbh_output_path = os.path.join(output_dir, "bbh.json")
                        
                        bbh_em = eval_bbh(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=bbh_dataset,
                            batch_size=config["utility"].get("batch_size", 2000),
                            output_result_dir=bbh_output_path,
                            use_vllm=use_vllm
                        )
                        
                        summary_results["results"]["bbh"] = {
                            "exact_match": float(bbh_em)
                        }
                        logging.info(f"BBH Evaluation complete! Exact Match: {bbh_em:.4f}")
                    except Exception as e:
                        logging.error(f"Error in BBH evaluation: {e}")
                        summary_results["results"]["bbh"] = {
                            "error": str(e)
                        }
                
                if run_triviaqa:
                    logging.info("Starting TriviaQA evaluation...")
                    try:
                        triviaqa_dataset_path = os.path.join(utility_dataset_dir, "triviaqa.json")
                        triviaqa_dataset = load_dataset(triviaqa_dataset_path)
                        triviaqa_output_path = os.path.join(output_dir, "triviaqa.json")
                        
                        triviaqa_em, triviaqa_f1 = eval_triviaqa(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=triviaqa_dataset,
                            batch_size=config["utility"].get("batch_size", 2000),
                            output_result_dir=triviaqa_output_path,
                            use_vllm=use_vllm
                        )
                        
                        summary_results["results"]["triviaqa"] = {
                            "exact_match": float(triviaqa_em),
                            "f1": float(triviaqa_f1)
                        }
                        logging.info(f"TriviaQA Evaluation complete! EM: {triviaqa_em:.4f}, F1: {triviaqa_f1:.4f}")
                    except Exception as e:
                        logging.error(f"Error in TriviaQA evaluation: {e}")
                        summary_results["results"]["triviaqa"] = {
                            "error": str(e)
                        }

                if run_truthfulqa:
                    logging.info("Starting TruthfulQA evaluation...")
                    try:
                        truthfulqa_dataset_path = os.path.join(utility_dataset_dir, "truthful.json")
                        truthfulqa_dataset = load_dataset(truthfulqa_dataset_path)
                        truthfulqa_output_path = os.path.join(output_dir, "truthful.json")
                        
                        mc1, mc2 = eval_truthfulqa(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=truthfulqa_dataset,
                            batch_size=config["utility"].get("batch_size", 2000),
                            output_result_dir=truthfulqa_output_path,
                            use_vllm=use_vllm
                        )
                        
                        summary_results["results"]["truthfulqa"] = {
                            "mc1": float(mc1),
                            "mc2": float(mc2)
                        }
                        logging.info(f"TruthfulQA Evaluation complete! MC1: {mc1:.4f}, MC2: {mc2:.4f}")
                    except Exception as e:
                        logging.error(f"Error in TruthfulQA evaluation: {e}")
                        summary_results["results"]["truthfulqa"] = {
                            "error": str(e)
                        }
                
                # Run Fluency evaluation
                if run_fluency:
                    logging.info("Starting Fluency evaluation...")
                    try:
                        fluency_dataset_path = os.path.join(utility_dataset_dir, "fluency.json")
                        fluency_dataset = load_dataset(fluency_dataset_path)
                        fluency_output_path = os.path.join(output_dir, "fluency.json")
                        
                        entropy = eval_fluency(
                            model=model,
                            tokenizer=tokenizer,
                            dataset=fluency_dataset,
                            batch_size=config["utility"].get("batch_size", 2000),
                            output_result_dir=fluency_output_path,
                            use_vllm=use_vllm
                        )
                        
                        summary_results["results"]["fluency"] = {
                            "entropy": float(entropy)
                        }
                        logging.info(f"Fluency Evaluation complete! Entropy: {entropy:.4f}")
                    except Exception as e:
                        logging.error(f"Error in Fluency evaluation: {e}")
                        summary_results["results"]["fluency"] = {
                            "error": str(e)
                        }
                
                all_summary_results.append(summary_results)

            # Save summary results
            with open(os.path.join(output_dir, "summary.json"), "w") as f:
                json.dump(all_summary_results, f, indent=4)
            
            logging.info(f"All evaluations complete! Results saved to {output_dir}")

    else:
        raise ValueError("Invalid mode. Please specify a valid mode: finetune, subgraph, judge, or query.")

if __name__ == "__main__":
    main()
