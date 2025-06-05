import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any
import logging
import os

class RandomLabelTrainer(Trainer):
    """
    Custom Trainer implementing the random label method for unlearning.
    Works with both Sentence format and QA format with DeepSpeed support.
    """
    
    def __init__(self, *args, use_QA_unlearning=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_QA_unlearning = use_QA_unlearning
        self.vocab_size = self.model.config.vocab_size
        # Add tracking variables for debugging
        self.step_count = 0
        self.params_before = None
        
        # Log initialization
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank in [-1, 0]:
            logging.info(f"RandomLabelTrainer initialized with vocab_size={self.vocab_size}")
            logging.info(f"Using QA unlearning format: {self.use_QA_unlearning}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a single training step with random labels, compatible with DeepSpeed.
        
        Args:
            model: The model being trained.
            inputs: The input batch from the dataset.
            num_items_in_batch: Number of items in the batch (required by Trainer).
            
        Returns:
            torch.Tensor: The detached loss value after applying random labeling.
        """
        # Store a sample parameter for tracking changes
        if self.step_count % 50 == 0 and self.args.local_rank in [-1, 0]:
            self.params_before = self._get_param_sample(model)
            
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].clone()

        if self.use_QA_unlearning:
            # Only randomize answer tokens (non -100 values)
            mask = (labels != -100)
            
            # Generate random labels only for answer tokens
            random_tokens = torch.randint_like(labels, high=self.vocab_size)
            
            # Apply random labels only where mask is True (answer tokens)
            # Keep -100 values for question tokens and padding
            labels = torch.where(mask, random_tokens, labels)
        else:
            # Sentence mode: randomize all tokens
            labels = torch.randint_like(labels, high=self.vocab_size)
            
        # Replace original labels with random ones
        inputs["labels"] = labels

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Log loss components occasionally
        if self.step_count % 1 == 0 and self.args.local_rank in [-1, 0]:
            logging.info(f"Loss: {loss.item():.4f}")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Handle different training backends
        if self.args.deepspeed:
            # Let DeepSpeed handle the backward pass
            self.deepspeed.backward(loss)
            # CRITICAL FIX: Call step method on the DeepSpeed engine to update parameters
            self.deepspeed.step()
        elif self.args.use_amp:
            # Use native PyTorch AMP
            self.accelerator.backward(loss)
            # CRITICAL FIX: Apply optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        else:
            # Standard backward pass
            loss.backward()
            # CRITICAL FIX: Apply optimizer step 
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        # Increment step count
        self.step_count += 1
        
        # Check if parameters actually changed
        if self.step_count % 50 == 0 and self.args.local_rank in [-1, 0]:
            params_after = self._get_param_sample(model)
            if self.params_before is not None:
                params_changed = any(not torch.allclose(p_before, p_after) 
                                  for p_before, p_after in zip(self.params_before, params_after))
                if params_changed:
                    logging.info(f"Step {self.step_count}: Parameters successfully updated")
                else:
                    logging.warning(f"Step {self.step_count}: Parameters NOT changing! Check optimizer settings.")
                
                # Optional: Log parameter change magnitude for debugging
                if params_changed:
                    for i, (p_before, p_after) in enumerate(zip(self.params_before, params_after)):
                        change = (p_after - p_before).abs().mean().item()
                        logging.info(f"  Parameter {i} mean absolute change: {change:.6f}")

        return loss.detach()
    
    def _get_param_sample(self, model):
        """Get a sample of parameters to check if they're changing"""
        params = []
        # Find trainable parameters to sample
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store a copy of the parameter tensor
                params.append(param.detach().clone())
                # Only keep a few samples for efficiency
                if len(params) >= 5:
                    break
        return params