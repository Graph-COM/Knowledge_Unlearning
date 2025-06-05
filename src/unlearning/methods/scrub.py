import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any, Optional
from torch.utils.data import SequentialSampler
import copy
import logging
import os

def has_length(dataset):
    """Check if the dataset implements `__len__()` without raising an error."""
    try:
        return len(dataset) is not None
    except TypeError:
        return False


class SCRUBTrainer(Trainer):
    def __init__(self, initial_model, *args, alpha=0.999, beta=0.001, gamma=0.99, use_QA_unlearning=False, **kwargs):
        """
        SCRUBTrainer applies:
        - **KL divergence tracking** between `initial_model` and the current model.
        - **Gradient Descent** for retain samples (factor = 1).
        - **Gradient Ascent** for unlearning samples (factor = -1).

        Args:
            initial_model (nn.Module): A frozen copy of the initial model for KL divergence.
            alpha (float): Weight for KL divergence on retain samples.
            beta (float): Weight for retain loss.
            gamma (float): Weight for KL divergence on unlearn samples.
            use_QA_unlearning (bool): Whether to use QA format for unlearning.
        """
        super().__init__(*args, **kwargs)
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.local_rank != -1:
            torch.cuda.set_device(self.local_rank)
        self.initial_model = initial_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.use_QA_unlearning = use_QA_unlearning
        self.initial_model_device_set = False
        
        # Add tracking variables for debugging
        self.step_count = 0
        self.params_before = None
        
        # Log initialization
        if self.local_rank in [-1, 0]:
            logging.info(f"SCRUBTrainer initialized with alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
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

    def compute_kl_divergence(self, current_logits, initial_logits, labels=None):
        """
        Compute KL divergence between current and initial model outputs.
        For QA format, applies masking to focus only on answer tokens.
        """
        if not self.use_QA_unlearning or labels is None:
            # Standard format - calculate KL on all tokens
            return nn.functional.kl_div(
                torch.log_softmax(current_logits, dim=-1),
                torch.softmax(initial_logits, dim=-1),
                reduction="batchmean"
            )
        
        # QA format - calculate KL only on answer tokens
        # Create a mask for non-ignored tokens
        if labels is not None:
            mask = (labels != -100).unsqueeze(-1).expand_as(current_logits)
            
            # Apply mask to both logits
            masked_current = current_logits.masked_select(mask).view(-1, current_logits.size(-1))
            masked_initial = initial_logits.masked_select(mask).view(-1, initial_logits.size(-1))
            
            if masked_current.numel() == 0:
                # Return zero if no tokens to compute KL on
                return torch.tensor(0.0, device=current_logits.device)
            
            # Compute KL only on selected tokens
            return nn.functional.kl_div(
                torch.log_softmax(masked_current, dim=-1),
                torch.softmax(masked_initial, dim=-1),
                reduction="batchmean"
            )
        
        # Fallback to standard KL if no labels available
        return nn.functional.kl_div(
            torch.log_softmax(current_logits, dim=-1),
            torch.softmax(initial_logits, dim=-1),
            reduction="batchmean"
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a single training step with SCRUB's **loss function**:
        - Minimizing retain loss
        - Minimizing KL divergence for retain samples
        - Maximizing KL divergence for unlearning samples
        
        Compatible with DeepSpeed and multi-GPU training.
        
        Args:
            model: The model being trained.
            inputs: The input batch from the dataset.
            num_items_in_batch: Number of items in the batch (required by Trainer).
            
        Returns:
            torch.Tensor: The detached loss value.
        """
        # Store a sample parameter for tracking changes - now every 50 steps as requested
        if self.step_count % 50 == 0 and self.local_rank in [-1, 0]:
            self.params_before = self._get_param_sample(model)
            
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Ensure initial_model is on the same device as the current model
        if not self.initial_model_device_set:
            device = next(model.parameters()).device
            self.initial_model = self.initial_model.to(device)
            self.initial_model_device_set = True
            logging.info(f"Initial model moved to device: {device}")

        # Check for required inputs
        if "factor" not in inputs:
            raise ValueError("Input batch is missing the 'factor' column needed for SCRUB training")

        # Clear CUDA cache before processing
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        try:
            unlearn_mask = inputs["factor"] == -1
            retain_mask = ~unlearn_mask
            
            # Log batch composition occasionally
            if self.step_count % 50 == 0 and self.local_rank in [-1, 0]:
                logging.info(f"Batch composition: {unlearn_mask.sum().item()} unlearn samples, {retain_mask.sum().item()} retain samples")

            unlearn_inputs = {key: val[unlearn_mask] for key, val in inputs.items() if key != "factor" and val is not None}
            retain_inputs = {key: val[retain_mask] for key, val in inputs.items() if key != "factor" and val is not None}
        except Exception as e:
            logging.error(f"Error splitting batch: {str(e)}")
            logging.error(f"Input keys: {list(inputs.keys())}")
            raise e

        retain_loss = None
        unlearn_loss = None
        retain_kl = None
        unlearn_kl = None

        # Compute retain loss
        if torch.any(retain_mask):
            try:
                # Save labels for KL calculation if using QA format
                retain_labels = retain_inputs.get("labels", None)
                
                with self.compute_loss_context_manager():
                    retain_loss = self.compute_loss(model, retain_inputs.copy())

                # Compute KL divergence for retain samples
                with torch.no_grad():
                    initial_outputs = self.initial_model(**retain_inputs)
                current_outputs = model(**retain_inputs)
                
                retain_kl = self.compute_kl_divergence(
                    current_outputs.logits, 
                    initial_outputs.logits,
                    retain_labels if self.use_QA_unlearning else None
                )
                
                # Check for NaN or infinity in the loss
                if torch.isnan(retain_loss) or torch.isinf(retain_loss):
                    logging.warning("NaN or Inf detected in retain loss, setting to zero.")
                    retain_loss = torch.tensor(0.0, device=retain_loss.device)
                
                if torch.isnan(retain_kl) or torch.isinf(retain_kl):
                    logging.warning("NaN or Inf detected in retain KL, setting to zero.")
                    retain_kl = torch.tensor(0.0, device=retain_kl.device)
            except Exception as e:
                logging.error(f"Error computing retain components: {str(e)}")
                raise e

        # Compute KL divergence for unlearning samples
        if torch.any(unlearn_mask):
            try:
                # Save labels for KL calculation if using QA format
                unlearn_labels = unlearn_inputs.get("labels", None)
                
                with torch.no_grad():
                    initial_outputs = self.initial_model(**unlearn_inputs)
                current_outputs = model(**unlearn_inputs)
                
                unlearn_kl = -self.compute_kl_divergence(
                    current_outputs.logits,
                    initial_outputs.logits,
                    unlearn_labels if self.use_QA_unlearning else None
                )
                
                # Check for NaN or infinity in the loss
                if torch.isnan(unlearn_kl) or torch.isinf(unlearn_kl):
                    logging.warning("NaN or Inf detected in unlearn KL, setting to zero.")
                    unlearn_kl = torch.tensor(0.0, device=unlearn_kl.device)
            except Exception as e:
                logging.error(f"Error computing unlearn components: {str(e)}")
                raise e

        # Compute total loss
        try:
            loss = 0
            if retain_kl is not None:
                loss += self.alpha * retain_kl
            if retain_loss is not None:
                loss += self.beta * retain_loss
            if unlearn_kl is not None:
                loss += self.gamma * unlearn_kl

            if loss == 0:
                raise ValueError("Neither retain nor unlearn points found in the batch.")
                
            # Log loss components occasionally
            if self.step_count % 1 == 0 and self.local_rank in [-1, 0]:
                comp_msg = []
                if retain_kl is not None:
                    comp_msg.append(f"Retain KL: {retain_kl.item():.4f} (weight: {self.alpha})")
                else:
                    comp_msg.append(f"Retain KL is None (weight: {self.alpha})")

                if retain_loss is not None:
                    comp_msg.append(f"Retain Loss: {retain_loss.item():.4f} (weight: {self.beta})")
                else:
                    comp_msg.append(f"Retain Loss is None (weight: {self.alpha})")

                if unlearn_kl is not None:
                    comp_msg.append(f"Unlearn KL: {unlearn_kl.item():.4f} (weight: {self.gamma})")
                else:
                    comp_msg.append(f"Unlearn KL is None (weight: {self.alpha})")

                logging.info(f"Loss components - {', '.join(comp_msg)}, Total: {loss.item():.4f}")

            if self.args.n_gpu > 1:
                loss = loss.mean()
        except Exception as e:
            logging.error(f"Error computing total loss: {str(e)}")
            raise e

        # Handle different training backends
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
        
        # Check if parameters actually changed - now every 50 steps as requested
        if self.step_count % 50 == 0 and self.local_rank in [-1, 0]:
            params_after = self._get_param_sample(model)
            if self.params_before is not None:
                params_changed = any(not torch.allclose(p_before, p_after) 
                                  for p_before, p_after in zip(self.params_before, params_after))
                if params_changed:
                    logging.info(f"Step {self.step_count}: Parameters successfully updated")
                    
                    # Calculate and log parameter change magnitude
                    total_change = 0.0
                    num_params = 0
                    for i, (p_before, p_after) in enumerate(zip(self.params_before, params_after)):
                        if i < 3:  # Show details for first few parameters
                            change = (p_after - p_before).abs().mean().item()
                            logging.info(f"  Parameter {i} mean absolute change: {change:.6f}")
                        total_change += (p_after - p_before).abs().mean().item()
                        num_params += 1
                    
                    logging.info(f"  Average parameter change: {total_change/num_params:.6f}")
                else:
                    logging.warning(f"Step {self.step_count}: Parameters NOT changing! Check optimizer settings.")

        # Clear cache after processing
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
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