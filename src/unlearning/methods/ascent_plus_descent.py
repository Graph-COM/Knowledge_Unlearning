import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any, Optional
from torch.utils.data import SequentialSampler
import os
import logging

def has_length(dataset):
    """Check if the dataset implements `__len__()` without raising an error."""
    try:
        return len(dataset) is not None
    except TypeError:
        return False


class AscentPlusDescentTrainer(Trainer):
    def __init__(self, *args, beta=0.999, use_QA_unlearning=False, **kwargs):
        """
        Trainer that applies:
        - **Gradient Ascent** for unlearning samples (factor = -1)
        - **Gradient Descent** for retain samples (factor = 1)

        Args:
            beta (float): Weight for retain loss in the optimization objective.
            use_QA_unlearning (bool): Whether to use QA format for unlearning.
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.use_QA_unlearning = use_QA_unlearning
        # Add tracking variables for debugging
        self.step_count = 0
        self.params_before = None
        
        # Log initialization
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank in [-1, 0]:
            logging.info(f"AscentPlusDescentTrainer initialized with beta={self.beta}")
            logging.info(f"Using QA unlearning format: {self.use_QA_unlearning}")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss differently based on format (standard vs QA).
        
        For QA format, we calculate loss only on the answer tokens, not the question tokens.
        """
        if not self.use_QA_unlearning:
            # Standard format - calculate loss on all tokens
            return super().compute_loss(model, inputs, return_outputs)
        
        # QA format - calculate loss only on answer tokens
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Create a mask to identify which tokens are part of the answer (non-padding and non-question)
        # In QA format, tokens with label -100 should be ignored (question tokens and padding)
        loss_mask = (labels != -100)
        
        # Calculate loss only on answer tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()
        
        # Get mean loss over actual answer tokens
        loss = masked_loss.sum() / max(loss_mask.sum(), 1)
        
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Override the default sampler with SequentialSampler to disable shuffling.
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        return SequentialSampler(self.train_dataset)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a single training step with **gradient ascent** for unlearning
        and **gradient descent** for retain dataset. Compatible with DeepSpeed.
        
        Args:
            model: The model being trained.
            inputs: The input batch from the dataset.
            num_items_in_batch: Number of items in the batch (required by Trainer).
            
        Returns:
            torch.Tensor: The detached loss value.
        """
        # Store a sample parameter for tracking changes
        if self.step_count % 50 == 0 and self.args.local_rank in [-1, 0]:
            self.params_before = self._get_param_sample(model)
            
        model.train()
        inputs = self._prepare_inputs(inputs)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # Handle edge cases and improve error reporting
        if "factor" not in inputs:
            raise ValueError("Input batch is missing the 'factor' column needed to distinguish unlearn from retain samples")

        # Split inputs based on 'factor' values (-1 = unlearn, 1 = retain)
        try:
            unlearn_mask = inputs["factor"] == -1
            retain_mask = ~unlearn_mask
                
            unlearn_inputs = {key: val[unlearn_mask] for key, val in inputs.items() if val is not None}
            retain_inputs = {key: val[retain_mask] for key, val in inputs.items() if val is not None}

            unlearn_inputs.pop("factor", None)
            retain_inputs.pop("factor", None)
        except Exception as e:
            logging.error(f"Error splitting batch: {str(e)}")
            logging.error(f"Input keys: {list(inputs.keys())}")
            raise e

        retain_loss = None
        unlearn_loss = None

        # Compute retain loss (Gradient Descent)
        if torch.any(retain_mask):
            with self.compute_loss_context_manager():
                retain_loss = self.compute_loss(model, retain_inputs)
                # Check for NaN or infinity in the loss
                if torch.isnan(retain_loss) or torch.isinf(retain_loss):
                    logging.warning("NaN or Inf detected in retain loss, setting to zero.")
                    retain_loss = torch.tensor(0.0, device=retain_loss.device)

        # Compute unlearn loss (Gradient Ascent)
        if torch.any(unlearn_mask):
            with self.compute_loss_context_manager():
                unlearn_loss = self.compute_loss(model, unlearn_inputs)
                # Check for NaN or infinity in the loss
                if torch.isnan(unlearn_loss) or torch.isinf(unlearn_loss):
                    logging.warning("NaN or Inf detected in unlearn loss, setting to zero.")
                    unlearn_loss = torch.tensor(0.0, device=unlearn_loss.device)

        # Combine losses: gradient descent for retain and gradient ascent for unlearning
        try:
            if retain_loss is not None and unlearn_loss is not None:
                loss = self.beta * retain_loss - (1 - self.beta) * unlearn_loss
                # Log loss components occasionally
                if self.step_count % 1 == 0 and self.args.local_rank in [-1, 0]:
                    logging.info(f"Loss components - Retain: {retain_loss.item():.4f}, Unlearn: {unlearn_loss.item():.4f}, Combined: {loss.item():.4f}")

            elif retain_loss is not None:
                loss = retain_loss
                # Log loss components occasionally
                if self.step_count % 1 == 0 and self.args.local_rank in [-1, 0]:
                    logging.info(f"Loss components - Retain: {retain_loss.item():.4f}, Unlearn is None, Combined: {loss.item():.4f}")
           
            elif unlearn_loss is not None:
                loss = -unlearn_loss
                # Log loss components occasionally
                if self.step_count % 1 == 0 and self.args.local_rank in [-1, 0]:
                    logging.info(f"Loss components - Retain is None, Unlearn: {unlearn_loss.item():.4f}, Combined: {loss.item():.4f}")
            
            else:
                raise ValueError("Neither retain nor unlearn points found in the batch.")

            if self.args.n_gpu > 1:
                loss = loss.mean()
        except Exception as e:
            logging.error(f"Error combining losses: {str(e)}")
            raise e

        # Handle different training backends for backward pass
        try:
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
        except Exception as e:
            logging.error(f"Error in backward/step: {str(e)}")
            # Try to diagnose common issues
            if "CUDA out of memory" in str(e):
                logging.error("CUDA OOM detected. Try reducing batch size or model size.")
            elif "NCCL" in str(e):
                logging.error("NCCL error detected. Check network connectivity between GPUs.")
            raise e
        
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
                    
                    # Calculate and log parameter change magnitude
                    total_change = 0.0
                    for i, (p_before, p_after) in enumerate(zip(self.params_before, params_after)):
                        change = (p_after - p_before).abs().mean().item()
                        total_change += change
                    
                    logging.info(f"  Average parameter change: {total_change/len(self.params_before):.6f}")
                else:
                    logging.warning(f"Step {self.step_count}: Parameters NOT changing! Check optimizer settings.")

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