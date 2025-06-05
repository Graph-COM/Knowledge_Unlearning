import os
import logging
import torch
import copy
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.unlearning.methods.gradient_ascent import GradientAscentTrainer
from src.unlearning.methods.negative_preference_optimization import NegativePreferenceOptimizationTrainer
from src.unlearning.methods.random_label import RandomLabelTrainer
from src.unlearning.methods.ascent_plus_descent import AscentPlusDescentTrainer
from src.unlearning.methods.scrub import SCRUBTrainer
from src.unlearning.methods.dp import NoisyTrainer
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from utils.deepspeed_config import create_deepspeed_config
import datetime
import random

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def setup_distributed_environment():
    """
    Properly setup the distributed environment with explicit device mapping.
    This prevents the barrier warning during checkpoint saving.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Explicitly set the device
        torch.cuda.set_device(local_rank)
        
        # Initialize process group with explicit device specification
        if not torch.distributed.is_initialized():
            init_method = "env://"
            torch.distributed.init_process_group(
                backend="nccl", 
                init_method=init_method,
                rank=local_rank,
                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                timeout=datetime.timedelta(minutes=30)
            )
            # Set this flag to avoid the barrier warning
            torch.distributed.barrier(device_ids=[local_rank])
            
        logging.info(f"Distributed setup complete for rank {local_rank}, "
                    f"world size: {torch.distributed.get_world_size()}")
    else:
        logging.info("Not running in distributed mode")

def perform_unlearning(model, model_name, tokenizer, unlearning_triples, retain_triples, method, output_dir, batch_size, epochs, learning_rate, k_layers=2, alpha=0.999, beta=0.001, gamma=0.99, noise_multiplier=1e-3, clip_norm=1, ref_model=None, use_QA_unlearning=False, peft_config=False, push_to_hub=False, hub_model_id=None, use_deepspeed=False, deepspeed_config_type="zero2", gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Perform unlearning using the specified method, ensuring LoRA is enabled.

    Args:
        model: The pretrained or fine-tuned language model.
        tokenizer: Corresponding tokenizer for the model.
        unlearning_triples (list): List of triples to be unlearned.
        retain_triples (list): List of triples to be retained.
        method (str): The unlearning method to use (e.g., "gradient_ascent").
        output_dir (str): Directory where the updated model will be saved.
        batch_size (int): Batch size for unlearning.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for unlearning.
        k_layers (int): Number of LoRA layers to unfreeze in EUk and CFk.
        alpha (float): SCRUB parameter.
        beta (float): SCRUB parameter.
        gamma (float): SCRUB parameter.
        use_QA_unlearning (bool): Whether to use QA format for unlearning (default: False).
        peft_config: Configuration for PEFT. If None, LoRA is used. If False, full parameter training is used.
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.

    Returns:
        model: The updated model after unlearning.
    """
    logging.info(f"Starting unlearning process using method: {method}")

    # Create DeepSpeed config if needed
    ds_config = None
    if use_deepspeed:
        setup_distributed_environment()
        logging.info(f"Using DeepSpeed with configuration type: {deepspeed_config_type}")
        ds_config = create_deepspeed_config(
            config_type=deepspeed_config_type,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Increase timeout to 30 minutes (default is 10 minutes)
        os.environ["NCCL_TIMEOUT"] = "1800"  
        # Add NCCL debug info if needed
        os.environ["NCCL_DEBUG"] = "INFO"
        # Disable timeout checking on infiniband connections
        os.environ["NCCL_IB_TIMEOUT"] = "0"
        # Set appropriate socket timeout
        os.environ["NCCL_SOCKET_TIMEOUT"] = "1800"
        logging.info("Set NCCL environment variables for improved stability")

    # Handle PEFT configuration
    if peft_config == "LoRA":  # Use LoRA 
        logging.info("Applying LoRA for unlearning...")
        model = _attach_lora_adapter(model)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        logging.info("Frozen all non-LoRA parameters.")
    else:
        # Full parameter training - all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
        logging.info("Using full parameter training (all parameters trainable).")

    # Ensure unlearning uses the same precision as finetuning
    dtype = next(model.parameters()).dtype
    if dtype == torch.float16:
        model.half()
        logging.info("Using FP16 precision for unlearning.")
    elif dtype == torch.float32:
        model.to(torch.float32)
        logging.info("Using FP32 precision for unlearning.")

    # Handle resume from checkpoint
    if resume_from_checkpoint:
        if resume_from_checkpoint == "latest":
            # Find the latest checkpoint in the output directory
            if os.path.exists(output_dir):
                checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
                if checkpoint_dirs:
                    # Sort by checkpoint number
                    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                    resume_from_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
                    logging.info(f"Resuming from latest checkpoint: {resume_from_checkpoint}")
                else:
                    logging.warning(f"No checkpoints found in {output_dir}, starting training from scratch.")
                    resume_from_checkpoint = None
            else:
                logging.warning(f"Output directory {output_dir} does not exist, starting training from scratch.")
                resume_from_checkpoint = None
        else:
            logging.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            if not os.path.exists(resume_from_checkpoint):
                logging.warning(f"Checkpoint {resume_from_checkpoint} does not exist, starting training from scratch.")
                resume_from_checkpoint = None

    # Convert triples into dataset
    if use_QA_unlearning:
        logging.info("Using QA format for unlearning...")
        unlearn_dataset = _convert_triples_to_QA_dataset(model_name, unlearning_triples, tokenizer)
        retain_dataset = _convert_triples_to_QA_dataset(model_name, retain_triples, tokenizer) 
    else:
        logging.info("Using Sentence format for unlearning...")
        unlearn_dataset = _convert_triples_to_dataset(unlearning_triples, tokenizer)
        retain_dataset = _convert_triples_to_dataset(retain_triples, tokenizer)
    
    # Apply the selected unlearning method with resume capability
    if method == "gradient_ascent":
        model = __unlearn_gradient_ascent(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "random_label":
        model = __unlearn_random_label(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "EUk":
        model = __unlearn_EUk(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, k_layers, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "CFk":
        model = __unlearn_CFk(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, k_layers, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "ascent_plus_descent":
        model = __unlearn_ascent_plus_descent(model, tokenizer, unlearn_dataset, retain_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "scrub":
        model = __unlearn_scrub(model, tokenizer, unlearn_dataset, retain_dataset, output_dir, batch_size, epochs, learning_rate, alpha, beta, gamma, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "noisy_training":
        if noise_multiplier is None or clip_norm is None:
            raise ValueError("Noisy Training requires noise_multiplier and clip_norm.")
        model = __unlearn_noisy_training(model, tokenizer, retain_dataset, output_dir, batch_size, epochs, learning_rate, noise_multiplier, clip_norm, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    elif method == "negative_preference_optimization":
        model = __negative_preference_optimization(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, ref_model, use_QA_unlearning, use_deepspeed, ds_config, gradient_accumulation_steps, resume_from_checkpoint)
    else:
        raise NotImplementedError(f"Unlearn method '{method}' is not implemented!")
    
    # Push to Hugging Face Hub if enabled
    if push_to_hub:
        if not hub_model_id:
            raise ValueError("You must provide a `hub_model_id` to push to Hugging Face Hub.")
        # Only push from rank 0 when using DeepSpeed
        if not use_deepspeed or int(os.environ.get("LOCAL_RANK", "0")) == 0:
            logging.info(f"Pushing model and tokenizer to HuggingFace Hub at {hub_model_id}...")
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
            logging.info("Successfully pushed to HuggingFace Hub.")

    return model

def _attach_lora_adapter(model):
    """
    Attach a LoRA adapter to a model if not already present.

    Args:
        model: The base or fine-tuned model.

    Returns:
        model: Model with LoRA adapter attached.
    """
    logging.info("Attaching LoRA adapter for unlearning...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    logging.info("LoRA adapter successfully attached.")
    return model


def _convert_triples_to_dataset(triples, tokenizer):
    """
    Convert a list of knowledge triples into structured, natural language unlearning samples.
    Features:
    - Uses a system message prefix
    - Randomly selects from a variety of templates for each triple
    - Properly aligns labels for next-token prediction mechanism

    Returns:
        datasets.Dataset: Tokenized dataset with properly aligned labels.
    """
    logging.info("Tokenizing unlearning triples with structured format...")
    
    # System message prefix
    SYSTEM_MESSAGE = "You are an expert in knowledge graphs.\n\n"
    
    # Templates for triple formatting
    TEMPLATES = [
        "'{h}' is connected to '{t}' through the relation '{r}'.",
        "There is a '{r}' link from '{h}' to '{t}'.",
        "'{h}' has a '{r}' relationship pointing to '{t}'.",
        "The relation '{r}' holds from '{h}' to '{t}'.",
        "In terms of '{r}', '{h}' relates to '{t}'.",
        "'{t}' is the '{r}' target of '{h}'.",
        "From '{h}' to '{t}', the relationship is '{r}'.",
        "'{h}' plays the role of source in a '{r}' relation with '{t}'.",
        "There exists a directed relation '{r}' from '{h}' to '{t}'.",
        "The entity '{h}' is linked to '{t}' via '{r}'.",
        "Through the '{r}' relation, '{h}' refers to '{t}'.",
        "'{h}' demonstrates '{r}' toward '{t}'.",
        "The subject '{h}' holds the property '{r}' with respect to '{t}'.",
        "A '{r}' connection starts from '{h}' and ends at '{t}'.",
        "The knowledge triple indicates that '{h}' has '{r}' to '{t}'.",
        "'{r}' is the type of relationship linking '{h}' and '{t}'.",
        "'{h}' asserts a '{r}' relation over '{t}'.",
        "According to the knowledge base, '{h}' is related to '{t}' via '{r}'.",
        "The direction of the '{r}' relation goes from '{h}' to '{t}'."
    ]
    
    # Unlearn prompt template
    UNLEARN_PROMPT = """Below is a correct fact that you should remember.
    ### Fact to Remember: {}
    """

    # Generate formatted text with randomly selected templates
    formatted_texts = []
    for h, r, t in triples:
        # Randomly select a template
        template = random.choice(TEMPLATES)
        # Format the triple using the selected template
        triple_text = template.format(h=h, r=r, t=t)
        # Combine with system message and unlearn prompt
        complete_text = SYSTEM_MESSAGE + UNLEARN_PROMPT.format(triple_text)
        formatted_texts.append(complete_text)
    
    # Log a few examples
    for i in range(min(3, len(formatted_texts))):
        logging.info(f"Example {i+1}:")
        logging.info(formatted_texts[i][:200] + "..." if len(formatted_texts[i]) > 200 else formatted_texts[i])

    # Tokenize texts
    tokenized_data = tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    
    # Create properly aligned labels for next-token prediction
    labels = []
    for seq in tokenized_data["input_ids"]:
        # Create shifted labels where label[i] = input[i+1]
        # The last position gets -100 (no next token to predict)
        seq_labels = seq[1:] + [-100]
        labels.append(seq_labels)
    
    tokenized_data["labels"] = labels
    
    # Log label alignment for verification (only for first example)
    if len(tokenized_data["input_ids"]) > 0 and tokenizer is not None:
        try:
            example_ids = tokenized_data["input_ids"][0]
            example_labels = labels[0]
            
            logging.info("Label alignment verification (first 10 tokens):")
            for i in range(min(10, len(example_ids)-1)):
                current_token = tokenizer.decode([example_ids[i]])
                next_token = tokenizer.decode([example_ids[i+1]])
                label_token = tokenizer.decode([example_labels[i]]) if example_labels[i] != -100 else "-100"
                
                logging.info(f"Position {i}: Token: '{current_token}' → Next: '{next_token}' → Label: '{label_token}'")
        except Exception as e:
            logging.error(f"Error during label alignment verification: {str(e)}")

    dataset = Dataset.from_dict({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["labels"]
    })
    
    logging.info(f"Created dataset with {len(dataset)} examples")
    return dataset

def _convert_triples_to_QA_dataset(model_name, triples, tokenizer):
    """
    Convert a list of knowledge triples into question-answer pairs for QA unlearning.

    Args:
        triples (list): List of (head, relation, tail) triples to convert.
        tokenizer: Tokenizer for the model.

    Returns:
        datasets.Dataset: Tokenized dataset for training.
    """
    logging.info("Tokenizing unlearning triples as QA pairs...")

    questions = []
    answers = []

    user_template = {
    "Qwen/Qwen2.5-7B-Instruct": "Task: In the triple ({entity1}, ?, {entity2}), does the relation '{relation}' correctly complete it? Answer:",
    "meta-llama/Llama-3.1-8B-Instruct": "Task: Given that the head entity is '{entity1}' and the tail entity is '{entity2}', is the relationship '{relation}'? Answer:",}

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

    use_chat_template = True if model_name in ["Qwen/Qwen2.5-7B-Instruct"] else False

    for h, r, t in triples:
        if use_chat_template:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_template[model_name].format(entity1=h, entity2=t, relation=r)},
                ]
                full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            full_prompt = system_message + "\n\n" +user_template[model_name].format(entity1=h, entity2=t, relation=r)

        question = full_prompt.rsplit("Answer:", 1)[0] + "Answer:"
        answer = " Yes"

        questions.append(question)
        answers.append(answer)

    for i in range(min(3, len(questions))):
        logging.info(f"Example {i+1}:")
        logging.info(f"Full question with system message (truncated): {questions[i]}")
        logging.info(f"Answer: {answers[i]}")

    # Create combined QA texts
    qa_texts = [q + a for q, a in zip(questions, answers)]

    # Tokenize the combined texts
    tokenized_data = tokenizer(
        qa_texts,
        padding="max_length",
        truncation=True,
        max_length=1024,
    )

    # Process each example to create correctly aligned labels
    labels = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        # Create array of -100 labels (ignored in loss)
        example_labels = [-100] * len(tokenized_data["input_ids"][i])

        # Tokenize just the question to find where answer starts
        q_tokens = tokenizer(q, add_special_tokens=True)
        q_length = len(q_tokens["input_ids"])

        # Tokenize just the answer to get its token IDs
        answer_token_ids = tokenizer.encode(a, add_special_tokens=False)

        # Key correction: Set the label for the token BEFORE the answer
        # to be the first token of the answer
        if q_length - 1 >= 0 and q_length < len(tokenized_data["input_ids"][i]):
            # Position of "Answer:" gets label of " Yes"
            example_labels[q_length - 1] = answer_token_ids[0]

            # Log this critical position for verification
            if i < 3:  # Only for first 3 examples
                token_before_answer = tokenizer.decode([tokenized_data["input_ids"][i][q_length - 1]])
                answer_token = tokenizer.decode([answer_token_ids[0]])
                logging.info(f"Example {i+1} - Critical position:")
                logging.info(f"  Token at position {q_length - 1}: '{token_before_answer}'")
                logging.info(f"  Label (next token to predict): '{answer_token}' (ID: {answer_token_ids[0]})")
                logging.info(f"  Full answer tokens: {answer_token_ids}")
        else:
            logging.warning(f"Example {i+1} - Could not set label: q_length={q_length}, input_length={len(tokenized_data['input_ids'][i])}")

        labels.append(example_labels)

    dataset = Dataset.from_dict({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": labels
    })

    return dataset

def apply_EUk_or_CFk_freezing(model, k_layers, reinit=False):
    """
    Freeze all parameters except the last k transformer layers.
    Works for both full models and LoRA models.
    Reinitializes weights if reinit=True (for EUk), otherwise keeps them (for CFk).
    Converts quantized weights to float32 before reinitializing if needed.
    Compatible with DeepSpeed model wrapping.

    Args:
        model: The model (LoRA or full) to modify in-place.
        k_layers: Number of transformer layers from the end to unfreeze.
        reinit: If True, reinitialize the unfrozen parameters.
    """
    # Handle DeepSpeed model wrapping
    if hasattr(model, "module"):
        actual_model = model.module
    else:
        actual_model = model
        
    is_lora = any("lora_" in name for name, _ in actual_model.named_parameters())
    is_quantized = hasattr(actual_model, "quantization_config")
    
    # Handle different model architectures
    if hasattr(actual_model, "model") and hasattr(actual_model.model, "layers"):
        total_layers = len(actual_model.model.layers)
        layers_attr = "model.layers"
    elif hasattr(actual_model, "transformer") and hasattr(actual_model.transformer, "h"):
        # For models like GPT-2
        total_layers = len(actual_model.transformer.h)
        layers_attr = "transformer.h"
    else:
        raise ValueError("Unsupported model architecture for layer freezing")

    logging.info(f"[FreezeConfig] Model type: {'LoRA' if is_lora else 'Full'}, Quantized: {is_quantized}, k={k_layers}, reinit={reinit}")

    # Freeze all parameters first
    for name, param in actual_model.named_parameters():
        param.requires_grad = False

    # Unfreeze and optionally reinitialize the last k transformer layers
    for i in range(total_layers - k_layers, total_layers):
        if layers_attr == "model.layers":
            block = actual_model.model.layers[i]
        else:
            block = actual_model.transformer.h[i]
            
        for name, param in block.named_parameters():
            if is_lora:
                if "lora_" in name:
                    param.requires_grad = True
                    if reinit:
                        with torch.no_grad():
                            param.data = torch.randn_like(param) * 0.02
                            if is_quantized:
                                param.data = param.data.to(torch.float32)
            else:
                param.requires_grad = True
                if reinit:
                    with torch.no_grad():
                        param.data = torch.randn_like(param) * 0.02
                        if is_quantized:
                            param.data = param.data.to(torch.float32)

    # Log how many parameters are trainable
    total_params = 0
    trainable_params = 0

    for name, param in actual_model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num

    percent = trainable_params / total_params * 100
    logging.info(f"[FreezeConfig] Trainable parameters: {trainable_params:,} / {total_params:,} ({percent:.2f}%)")

def __unlearn_gradient_ascent(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn specific knowledge using gradient ascent with DeepSpeed support.

    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        unlearn_dataset: Dataset of samples to unlearn.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.

    Returns:
        model: The updated model after applying gradient ascent.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying Gradient Ascent unlearning with {format_str} format...")

    model.train()

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
        disable_tqdm=local_rank != 0
    )

    trainer = GradientAscentTrainer(
        model=model,
        args=train_args,
        train_dataset=unlearn_dataset,
        tokenizer=tokenizer,
        use_QA_unlearning=use_QA_unlearning  
    )

    # Train the model with capability to resume from checkpoint
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"Gradient ascent unlearning completed.")

    return model


def __unlearn_random_label(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn specific knowledge using random label method with DeepSpeed support.

    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        unlearn_dataset: Dataset of samples to unlearn.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.

    Returns:
        model: The updated model after applying random label unlearning.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying Random Label unlearning with {format_str} format...")

    model.train()

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = RandomLabelTrainer(
        model=model,
        args=train_args,
        train_dataset=unlearn_dataset,
        use_QA_unlearning=use_QA_unlearning
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"Random Label unlearning completed.")

    return model

def __unlearn_EUk(model, tokenizer, retain_dataset, output_dir, batch_size, epochs, learning_rate, k_layers, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn using EUk method: Unfreeze and reinitialize the last k layers.
    With DeepSpeed support for multi-GPU training.
    
    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        retain_dataset: Dataset of samples to retain.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        k_layers: Number of layers to unfreeze and reinitialize.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.
    
    Returns:
        model: The updated model after unlearning.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying EUk unlearning with {format_str} format (Reinitializing last {k_layers} layers)...")

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Apply freezing and reinitialization
    apply_EUk_or_CFk_freezing(model, k_layers, reinit=True)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=retain_dataset
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"EUk unlearning completed.")

    return model


def __unlearn_CFk(model, tokenizer, retain_dataset, output_dir, batch_size, epochs, learning_rate, k_layers, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn using CFk method: Unfreeze the last k layers without reinitialization.
    With DeepSpeed support for multi-GPU training.
    
    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        retain_dataset: Dataset of samples to retain.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        k_layers: Number of layers to unfreeze.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.
    
    Returns:
        model: The updated model after unlearning.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying CFk unlearning with {format_str} format (Fine-tuning last {k_layers} layers)...")

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Apply freezing but no reinitialization
    apply_EUk_or_CFk_freezing(model, k_layers, reinit=False)

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=retain_dataset
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"CFk unlearning completed.")

    return model


def __unlearn_ascent_plus_descent(model, tokenizer, unlearn_dataset, retain_dataset, output_dir, batch_size, epochs, learning_rate, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn using AscentPlusDescent: gradient ascent for unlearning and gradient descent for retain set.
    With DeepSpeed support for multi-GPU training.
    
    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        unlearn_dataset: Dataset of samples to unlearn.
        retain_dataset: Dataset of samples to retain.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.
    
    Returns:
        model: The updated model after unlearning.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying AscentPlusDescent unlearning with {format_str} format...")

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    retain_size_per_epoch = len(unlearn_dataset)
    combined_datasets = []

    for epoch in range(epochs):
        retain_epoch_dataset = retain_dataset.shuffle().select(range(retain_size_per_epoch))
        unlearn_epoch_dataset = unlearn_dataset.add_column("factor", [-1] * len(unlearn_dataset))
        retain_epoch_dataset = retain_epoch_dataset.add_column("factor", [1] * len(retain_epoch_dataset))

        combined_dataset = concatenate_datasets([unlearn_epoch_dataset, retain_epoch_dataset]).shuffle()
        combined_datasets.append(combined_dataset)

    combined_dataset = concatenate_datasets(combined_datasets)
    
    if "factor" not in combined_dataset.column_names:
        raise ValueError("The 'factor' column is missing in the training dataset.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Calculate total steps and save steps
    # Account for multi-GPU training by dividing by the world size if using DeepSpeed
    world_size = torch.distributed.get_world_size() if (use_deepspeed and torch.distributed.is_initialized()) else 1
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    total_samples = len(combined_dataset)
    total_steps = total_samples // effective_batch_size
    # If there's a remainder, add one more step
    if total_samples % effective_batch_size != 0:
        total_steps += 1
    
    # Set save_steps to save 'epochs' number of checkpoints
    save_steps = max(total_steps // epochs, 1)  # Ensure at least 1
    logging.info(f"Step-based checkpoint saving: total_steps={total_steps}, save_steps={save_steps}, will save {total_steps//save_steps} checkpoints")

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        learning_rate=learning_rate,
        logging_steps=min(10, max(1, save_steps // 10)),  # Adjust logging frequency 
        save_only_model=True,
        save_safetensors=True,
        save_strategy="steps",
        save_steps=save_steps,
        report_to="none",
        remove_unused_columns=False,
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        torch_compile=False,  # Disable torch.compile which can cause issues
        dataloader_num_workers=2,  # Increase data loading parallelism
        group_by_length=False,  # Disable length grouping to ensure balanced batches
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = AscentPlusDescentTrainer(
        model=model,
        args=train_args,
        train_dataset=combined_dataset,
        data_collator=data_collator,
        use_QA_unlearning=use_QA_unlearning  
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"AscentPlusDescent unlearning completed.")

    return model


def __unlearn_scrub(model, tokenizer, unlearn_dataset, retain_dataset, output_dir, batch_size, epochs, learning_rate, alpha, beta, gamma, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn using SCRUB: combines KL divergence tracking with gradient-based unlearning.
    With DeepSpeed support for multi-GPU training.
    
    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        unlearn_dataset: Dataset of samples to unlearn.
        retain_dataset: Dataset of samples to retain.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        alpha: SCRUB parameter - weight for KL divergence on retain samples.
        beta: SCRUB parameter - weight for retain loss.
        gamma: SCRUB parameter - weight for KL divergence on unlearn samples.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.
    
    Returns:
        model: The updated model after unlearning.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying SCRUB unlearning with {format_str} format...")

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    retain_size_per_epoch = len(unlearn_dataset)
    combined_datasets = []

    for epoch in range(epochs):
        retain_epoch_dataset = retain_dataset.shuffle().select(range(retain_size_per_epoch))
        unlearn_epoch_dataset = unlearn_dataset.add_column("factor", [-1] * len(unlearn_dataset))
        retain_epoch_dataset = retain_epoch_dataset.add_column("factor", [1] * len(retain_epoch_dataset))

        combined_dataset = concatenate_datasets([unlearn_epoch_dataset, retain_epoch_dataset]).shuffle()
        combined_datasets.append(combined_dataset)

    combined_dataset = concatenate_datasets(combined_datasets)

    # Create a copy of the initial model for KL divergence
    # In distributed training, each process needs its own copy
    # Don't move to any specific device yet - will be done in the trainer
    initial_model = copy.deepcopy(model).eval()
    
    # Set all parameters to not require gradients to save memory
    for param in initial_model.parameters():
        param.requires_grad = False

    # Calculate total steps and save steps
    # Account for multi-GPU training by dividing by the world size if using DeepSpeed
    world_size = torch.distributed.get_world_size() if (use_deepspeed and torch.distributed.is_initialized()) else 1
    effective_batch_size = batch_size * gradient_accumulation_steps * world_size
    total_samples = len(combined_dataset)
    total_steps = total_samples // effective_batch_size
    # If there's a remainder, add one more step
    if total_samples % effective_batch_size != 0:
        total_steps += 1
    
    # Set save_steps to save 'epochs' number of checkpoints
    save_steps = max(total_steps // epochs, 1)  # Ensure at least 1
    logging.info(f"Step-based checkpoint saving: total_steps={total_steps}, save_steps={save_steps}, will save {total_steps//save_steps} checkpoints")

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        learning_rate=learning_rate,
        logging_steps=min(10, max(1, save_steps // 10)),  # Adjust logging frequency
        save_only_model=True,
        save_safetensors=True,
        save_strategy="steps",
        save_steps=save_steps,
        report_to="none",
        remove_unused_columns=False,
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = SCRUBTrainer(
        model=model,
        initial_model=initial_model,
        args=train_args,
        train_dataset=combined_dataset,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        use_QA_unlearning=use_QA_unlearning 
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"SCRUB unlearning completed.")

    return model


def __unlearn_noisy_training(model, tokenizer, retain_dataset, output_dir, batch_size, epochs, learning_rate, noise_multiplier, clip_norm, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn using Noisy Training: apply DP-SGD with Gaussian noise and gradient clipping.
    With DeepSpeed support for multi-GPU training.

    Args:
        model: The language model to fine-tune.
        tokenizer: The corresponding tokenizer.
        retain_dataset: The dataset containing knowledge to be retained.
        output_dir: Directory to save the updated model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimization.
        noise_multiplier: Scale of Gaussian noise added to gradients.
        clip_norm: Clipping norm for gradients.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying Noisy Training (noise={noise_multiplier}, clip_norm={clip_norm}) with {format_str} format...")

    model.train()

    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # DeepSpeed requires special handling with Opacus
    # If DeepSpeed is used, we'll modify the approach for compatibility
    if use_deepspeed:
        logging.info("Using DeepSpeed with Noisy Training. Adapting approach for compatibility.")

    # Training arguments
    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = NoisyTrainer(
        model=model,
        args=train_args,
        train_dataset=retain_dataset,
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        use_QA_unlearning=use_QA_unlearning,
        use_deepspeed=use_deepspeed
    )

    # Train the model with DP-SGD
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"Noisy Training completed.")

    return model


def __negative_preference_optimization(model, tokenizer, unlearn_dataset, output_dir, batch_size, epochs, learning_rate, ref_model, use_QA_unlearning=False, use_deepspeed=False, ds_config=None, gradient_accumulation_steps=4, resume_from_checkpoint=None):
    """
    Unlearn specific knowledge using negative preference optimization.
    With DeepSpeed support for multi-GPU training.

    Args:
        model: The model being trained.
        tokenizer: Corresponding tokenizer.
        unlearn_dataset: Dataset of samples to unlearn.
        output_dir: Directory to save the model.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Learning rate.
        ref_model: Reference model for preference optimization.
        use_QA_unlearning: Whether to use QA format (default: False).
        use_deepspeed: Whether to use DeepSpeed (default: False).
        ds_config: DeepSpeed configuration dictionary (default: None).
        gradient_accumulation_steps: Number of gradient accumulation steps (default: 4).
        resume_from_checkpoint: Path to checkpoint to resume training from, or "latest" to use most recent checkpoint.

    Returns:
        model: The updated model after applying NPO.
    """
    format_str = "QA" if use_QA_unlearning else "Sentence"
    logging.info(f"Applying Negative Preference Optimization unlearning with {format_str} format...")
    
    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if ref_model is None:
        ref_model = copy.deepcopy(model).eval()
    
    model.train()

    # Configure parameter training based on whether we're using LoRA or full parameter training
    is_lora = any("lora" in name for name, _ in model.named_parameters())
    if is_lora:
        logging.info("Using LoRA for NPO. Only LoRA parameters will be trainable.")
        for name, param in model.named_parameters():
            param.requires_grad = "lora" in name
    else:
        logging.info("Using full parameter training for NPO.")
        for param in model.parameters():
            param.requires_grad = True

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_only_model=True,
        save_safetensors=True,
        save_strategy="epoch",
        report_to="none",
        # DeepSpeed configuration
        deepspeed=ds_config if use_deepspeed else None,
        # Enable mixed precision training if using DeepSpeed
        fp16=True if use_deepspeed and ds_config.get("fp16", {}).get("enabled", False) else False,
        # Set local_rank for distributed training
        local_rank=local_rank
    )

    trainer = NegativePreferenceOptimizationTrainer(
        model=model,
        args=train_args,
        train_dataset=unlearn_dataset,
        ref_model=ref_model,
        use_QA_unlearning=use_QA_unlearning,
        use_deepspeed=use_deepspeed
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Only save the model on the master process when using DeepSpeed
    if not use_deepspeed or local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        
        # If using DeepSpeed, we need to consolidate the model before saving
        if use_deepspeed:
            # Get unwrapped model
            if hasattr(model, "module"):
                unwrapped_model = model.module
            else:
                unwrapped_model = model
                
            # Save the model
            if hasattr(unwrapped_model, "save_pretrained"):
                unwrapped_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")
        else:
            # Standard saving
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logging.info(f"Model and tokenizer saved at {output_dir}")

    model.eval()
    logging.info(f"Negative Preference Optimization unlearning completed.")

    return model
