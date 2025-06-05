import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from vllm import LLM, SamplingParams

def load_model(
    model_name,
    checkpoint_path=None,
    hf_token=None,
    multi_gpu=False,
    use_deepspeed=False,
    instruct_finetuning=False,
    use_lora=False,
    load_in_4bit=False,
    use_vllm=False,  # New parameter to toggle VLLM usage
    vllm_kwargs=None,  # Additional VLLM configuration parameters
    merge_lora=False,  # Whether to merge LoRA weights for inference
    save_merged_model_path=None  # Where to save the merged model (if merging)
):
    """
    Load and prepare models with support for PEFT fine-tuning workflow and vLLM inference.
    
    This function supports the recommended workflow of:
    1. Fine-tuning with PEFT/LoRA (use_lora=True, use_vllm=False)
    2. Merging LoRA weights with base model (merge_lora=True, save_merged_model_path="path")
    3. Using vLLM for accelerated inference with merged model (use_vllm=True)

    Args:
        model_name (str): Model name from Hugging Face Hub or local path.
        checkpoint_path (str or None): Path to a LoRA/fine-tuned model checkpoint (if available).
        hf_token (str or None): Hugging Face token for private models.
        multi_gpu (bool): Whether to enable multi-GPU.
        instruct_finetuning (bool): Indicates whether the model is being prepared for fine-tuning.
        use_lora (bool): Whether to load and use LoRA weights.
        load_in_4bit (bool): Whether to load the model in 4-bit quantization (useful for QLoRA fine-tuning or inference).
        use_vllm (bool): Whether to use VLLM for accelerated inference.
        vllm_kwargs (dict or None): Additional arguments to pass to VLLM's LLM initialization.
        merge_lora (bool): Whether to merge LoRA weights with the base model (for inference).
        save_merged_model_path (str or None): Path to save the merged model (required if merge_lora=True).

    Returns:
        tuple: (model, tokenizer) - Model type depends on configuration options.
    """
    logging.info("Initializing base model loading...")

    # 1) Load tokenizer (always needed, even with VLLM for consistent tokenization)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2) Handle LoRA merging workflow (PEFT fine-tuning -> merged model -> vLLM inference)
    if merge_lora:
        if not save_merged_model_path:
            raise ValueError("save_merged_model_path must be provided when merge_lora=True")

        os.makedirs(save_merged_model_path, exist_ok=True)

        if len(os.listdir(save_merged_model_path)) == 0:
            logging.info(f"Loading base model for LoRA merging: {model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation='flash_attention_2',
                token=hf_token,
                torch_dtype="auto",
                load_in_4bit=load_in_4bit
            )


            logging.info(f"Attaching LoRA adapter from {checkpoint_path} for merging")
            peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

            logging.info("Merging LoRA weights with base model...")
            merged_model = peft_model.merge_and_unload()

            logging.info(f"Saving merged model to {save_merged_model_path}")
            merged_model.save_pretrained(save_merged_model_path)

        checkpoint_path = save_merged_model_path
        use_lora = False

        if not use_vllm:
            return merged_model, tokenizer

    # 3) Check if using VLLM for inference
    if use_vllm:
        logging.info(f"Loading model with VLLM for accelerated inference")

        if vllm_kwargs is None:
            vllm_kwargs = {}

        vllm_config = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) if multi_gpu else 1,
            "trust_remote_code": True,
            "max_logprobs": 10000
        }
        vllm_config.update(vllm_kwargs)

        model_path = checkpoint_path if checkpoint_path else model_name

        if hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        model = LLM(
            model=model_path,
            tokenizer=model_name,
            download_dir=os.environ.get("TRANSFORMERS_CACHE", None),
            **vllm_config
        )

        logging.info("VLLM model successfully loaded.")
        return model, tokenizer
    
    # 4) Standard Hugging Face model loading logic
    if checkpoint_path and not use_vllm:
        logging.info(f"Loading model from local checkpoint: {checkpoint_path}")
        if multi_gpu and not use_deepspeed:
            # Only use device_map="auto" if NOT using DeepSpeed
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype="auto",
                load_in_4bit=load_in_4bit
            )
        else:
            # For single GPU or when using DeepSpeed
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype="auto",
                load_in_4bit=load_in_4bit,
                # use_flash_attention_2=True
            )
            
        # Check if LoRA adapter needs to be attached
        if use_lora and not hasattr(model, "peft_config"):
            logging.info("Attaching LoRA adapter...")
            model = PeftModel.from_pretrained(model, checkpoint_path)
            # model.load_adapter(checkpoint_path, adapter_name="adapter_model.safetensors")
            logging.info("LoRA adapter successfully attached.")
        elif use_lora:
            logging.info("LoRA adapter already attached, skipping reattachment.")

    else:
        logging.info(f"Loading pretrained model from Hugging Face: {model_name}")
        if multi_gpu and not use_deepspeed:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                device_map="auto",
                torch_dtype="auto",
                load_in_4bit=load_in_4bit
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype="auto",
                load_in_4bit=load_in_4bit,
            )

        logging.info("Hugging Face model successfully loaded.")

    return model, tokenizer

def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_tokens=128, 
    temperature=0.7, 
    top_p=0.9, 
    use_vllm=False, 
    return_logprobs=False,  # New parameter to enable logprobs
    logprobs_top_k=5,  # Number of top logprobs to return per token
    **kwargs
):
    """
    Generate text using either vLLM or Hugging Face model with optional logprobs.
    
    Args:
        model: Either a vLLM LLM instance or a Hugging Face model
        tokenizer: Hugging Face tokenizer
        prompt (str): Input prompt for generation
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
        top_p (float): Top-p sampling parameter
        use_vllm (bool): Whether the model is a vLLM instance
        return_logprobs (bool): Whether to return log probabilities of generated tokens
        logprobs_top_k (int): Number of top logprobs to return per token
        **kwargs: Additional generation parameters
        
    Returns:
        If return_logprobs=False:
            str: Generated text
        If return_logprobs=True:
            tuple: (generated_text, logprobs_data)
                - generated_text (str): The generated text
                - logprobs_data: Dictionary containing token logprobs information
    """
    if use_vllm:
        # vLLM generation with logprobs support
        sampling_params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # Add logprobs parameter if requested
        if return_logprobs:
            sampling_params_dict["logprobs"] = logprobs_top_k
            
        # Add any additional parameters
        sampling_params_dict.update(kwargs)
        
        # Create sampling parameters
        sampling_params = SamplingParams(**sampling_params_dict)
        
        # Generate with vLLM
        outputs = model.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Return with logprobs if requested
        if return_logprobs:
            # Extract logprobs from vLLM output
            logprobs_data = {}
            
            # Extract token-level logprobs from the output
            output_obj = outputs[0].outputs[0]
            
            # Try to extract standard logprobs fields
            for field in ["logprobs", "token_logprobs", "tokens", "top_logprobs"]:
                if hasattr(output_obj, field):
                    logprobs_data[field] = getattr(output_obj, field)
            
            return generated_text, logprobs_data
        else:
            return generated_text
    else:
        # Hugging Face generation with logprobs support
        import torch
        import torch.nn.functional as F
        
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Set up generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        # Add parameters for returning scores if logprobs requested
        if return_logprobs:
            generation_kwargs.update({
                "output_scores": True,
                "return_dict_in_generate": True,
            })
            
        # Add any additional parameters
        generation_kwargs.update(kwargs)
        
        # Generate
        output = model.generate(input_ids, **generation_kwargs)
        
        # Process the output
        if return_logprobs and hasattr(output, "sequences") and hasattr(output, "scores"):
            # Get generated sequence (excluding prompt)
            sequences = output.sequences
            generated_ids = sequences[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Initialize logprobs data
            logprobs_data = {
                "tokens": [],
                "token_ids": generated_ids.tolist(),
                "token_logprobs": [],
            }
            
            # Process scores for each token
            for i, token_scores in enumerate(output.scores):
                if i < len(generated_ids):
                    # Get token ID
                    token_id = generated_ids[i].item()
                    
                    # Get logits and convert to log probs
                    logits = token_scores[0]
                    logprobs = F.log_softmax(logits, dim=-1)
                    
                    # Get log prob for the selected token
                    token_logprob = logprobs[token_id].item()
                    
                    # Get token text
                    token_text = tokenizer.decode([token_id])
                    
                    # Add to results
                    logprobs_data["tokens"].append(token_text)
                    logprobs_data["token_logprobs"].append(token_logprob)
                    
                    # Add top-k logprobs if requested
                    if logprobs_top_k > 0:
                        # Get top-k logprobs
                        topk_values, topk_indices = torch.topk(
                            logprobs, k=min(logprobs_top_k, len(logprobs))
                        )
                        
                        # Create mapping from token to logprob
                        top_logprobs_dict = {
                            tokenizer.decode([idx.item()]): logprob.item()
                            for idx, logprob in zip(topk_indices, topk_values)
                        }
                        
                        # Add to results
                        if "top_logprobs" not in logprobs_data:
                            logprobs_data["top_logprobs"] = []
                        logprobs_data["top_logprobs"].append(top_logprobs_dict)
            
            return generated_text, logprobs_data
        elif return_logprobs:
            # Fallback if structure doesn't match expectations
            if hasattr(output, "sequences"):
                generated_text = tokenizer.decode(
                    output.sequences[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            else:
                generated_text = tokenizer.decode(
                    output[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            
            # Return with empty logprobs data
            return generated_text, {}
        else:
            # Standard return without logprobs
            if hasattr(output, "sequences"):
                generated_text = tokenizer.decode(
                    output.sequences[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            else:
                generated_text = tokenizer.decode(
                    output[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            
            return generated_text

def generate_text_batch(
    model,
    tokenizer,
    prompts,
    max_tokens=128,
    temperature=0.7,
    top_p=0.9,
    use_vllm=False,
    return_logprobs=False,
    logprobs_top_k=5,
    **kwargs
):
    """
    Batched text generation for a list of prompts with optional logprobs.
    Returns a list of (generated_text, logprobs_data) tuples or just generated_text if return_logprobs=False.
    """
    if use_vllm:
        from vllm import SamplingParams

        sampling_params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if return_logprobs:
            sampling_params_dict["logprobs"] = logprobs_top_k
        sampling_params_dict.update(kwargs)
        sampling_params = SamplingParams(**sampling_params_dict)

        outputs = model.generate(prompts, sampling_params)
        results = []
        for out in outputs:
            text = out.outputs[0].text
            logprobs_data = {}
            if return_logprobs:
                output_obj = out.outputs[0]
                for field in ["logprobs", "token_logprobs", "tokens", "top_logprobs"]:
                    if hasattr(output_obj, field):
                        logprobs_data[field] = getattr(output_obj, field)
                results.append((text, logprobs_data))
            else:
                results.append(text)
        return results
    else:
        raise NotImplementedError("Batched inference for non-vLLM models is not implemented.")

