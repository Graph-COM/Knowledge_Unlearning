import torch

def create_deepspeed_config(config_type="zero2", batch_size=1, learning_rate=5e-5, gradient_accumulation_steps=4):
    """
    Create DeepSpeed configuration for distributed training.
    
    Args:
        config_type (str): DeepSpeed configuration type ("zero1", "zero2", or "zero3").
        batch_size (int): Per-device batch size.
        learning_rate (float): Learning rate for training.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
        
    Returns:
        dict: DeepSpeed configuration dictionary.
    """
    # Calculate global batch size
    train_batch_size = batch_size * torch.cuda.device_count() * gradient_accumulation_steps
    
    # Base configuration
    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }
        },
        "fp16": {
            "enabled": True
        },
        "communication_data_type": "fp16",
        "prescale_gradients": False,
        "gradient_predivide_factor": 1.0
    }
    
    # Add ZeRO-specific configurations
    if config_type == "zero1":
        config["zero_optimization"] = {
            "stage": 1
        }
    elif config_type == "zero2":
        config["zero_optimization"] = {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    elif config_type == "zero3":
        config["zero_optimization"] = {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 1e6,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_gather_16bit_weights_on_model_save": True,
            "ignore_unused_parameters": True
        }
    
    return config