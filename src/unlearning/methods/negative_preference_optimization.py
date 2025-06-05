import os
import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
from typing import Dict, Union, Any
import logging

class NegativePreferenceOptimizationTrainer(Trainer):
    """
    Custom Trainer implementing negative preference optimization for unlearning.
    Supports both standard and QA unlearning formats with DeepSpeed.
    """
    def __init__(self, model, args, train_dataset, ref_model, use_QA_unlearning=False, use_deepspeed=False):
        super(NegativePreferenceOptimizationTrainer, self).__init__(model=model, args=args, train_dataset=train_dataset)
        self.ref_model = ref_model
        self.use_QA_unlearning = use_QA_unlearning
        self.use_deepspeed = use_deepspeed
        self.beta = 0.1
        
        # Add tracking variables for parameter update checking
        self.step_count = 0
        self.params_before = None
        
        # Properly place the reference model based on distributed settings
        if use_deepspeed:
            # When using DeepSpeed, local_rank determines the device
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.device_id = local_rank
            logging.info(f"Placing reference model on CUDA device {local_rank}")
            self.ref_model = self.ref_model.to(f'cuda:{local_rank}')
        else:
            # For single GPU, use the second GPU if available, otherwise use the same GPU
            if torch.cuda.device_count() > 1:
                self.device_id = 1
                logging.info(f"Placing reference model on separate CUDA device: cuda:1")
                self.ref_model = self.ref_model.to('cuda:1')
            else:
                self.device_id = 0
                logging.info(f"Only one CUDA device available, reference model will share device with training model")
                self.ref_model = self.ref_model.to('cuda:0')
        
        # Set reference model to evaluation mode
        self.ref_model.eval()
        
        # Log initialization information
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank in [-1, 0]:
            logging.info(f"NegativePreferenceOptimizationTrainer initialized with beta={self.beta}")
            logging.info(f"Using QA unlearning format: {self.use_QA_unlearning}")
            logging.info(f"Using DeepSpeed: {self.use_deepspeed}")

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute NPO loss with support for both standard and QA formats.
        Compatible with DeepSpeed distributed training.
        """
        labels = inputs.pop("labels").clone()  # Clone to avoid modifying the original

        # Get model outputs - DeepSpeed compatible
        outputs = model(**inputs)
        outputs_logits = outputs.logits

        # Prepare inputs for reference model - move to reference model device if different
        ref_inputs = {}
        for key in inputs.keys():
            # Only move tensors - some inputs might be other types
            if isinstance(inputs[key], torch.Tensor):
                ref_inputs[key] = inputs[key].to(self.ref_model.device)
            else:
                ref_inputs[key] = inputs[key]

        # Get reference model outputs without gradient tracking
        with torch.no_grad():
            outputs_f_ref_logits = self.ref_model(**ref_inputs).logits
            
            # Move reference logits to the same device as model logits if needed
            if outputs_f_ref_logits.device != outputs_logits.device:
                outputs_f_ref_logits = outputs_f_ref_logits.to(outputs_logits.device)
        
        # Calculate negative log ratio
        neg_log_ratio = outputs_f_ref_logits - outputs_logits
        
        if not self.use_QA_unlearning:
            # Standard format: compute loss on all tokens
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        else:
            # QA format: compute loss only on answer tokens
            # Create a mask for tokens that should contribute to loss (not -100)
            # Reshape labels to match token dimension
            labels_mask = (labels != -100).unsqueeze(-1).expand_as(neg_log_ratio)
            
            # Calculate log sigmoid on all tokens
            log_sigmoid = F.logsigmoid(self.beta * neg_log_ratio)
            
            # Apply mask to only include answer tokens
            masked_log_sigmoid = log_sigmoid * labels_mask.float()
            
            # Sum masked values and divide by number of answer tokens for proper mean
            token_count = labels_mask.sum().float()
            token_count = torch.clamp(token_count, min=1.0)  # Avoid division by zero
            
            # Compute final loss
            loss = -masked_log_sigmoid.sum() * 2 / (self.beta * token_count)
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with NPO, compatible with DeepSpeed.
        
        Args:
            model: The model being trained.
            inputs: The input batch from the dataset.
            num_items_in_batch: Number of items in the batch (required by Trainer).
            
        Returns:
            torch.Tensor: The detached loss value.
        """
        # Store a sample parameter for tracking changes every 50 steps as requested
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.step_count % 50 == 0 and local_rank in [-1, 0]:
            self.params_before = self._get_param_sample(model)
            logging.info(f"Step {self.step_count}: Capturing parameters before update")
        
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Clear CUDA cache before processing
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        try:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            # Log loss components occasionally
            if self.step_count % 1 == 0 and local_rank in [-1, 0]:
                logging.info(f"Loss: {loss.item():.4f}")

            if self.args.n_gpu > 1:
                loss = loss.mean()

            # Handle different training backends
            if self.args.deepspeed:
                # Let DeepSpeed handle the backward pass
                self.deepspeed.backward(loss)
                # CRITICAL: Call step method on the DeepSpeed engine to update parameters
                self.deepspeed.step()
            elif self.args.use_amp:
                # Use native PyTorch AMP
                self.accelerator.backward(loss)
                # Apply optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            else:
                # Standard backward pass
                loss.backward()
                # Apply optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
        except Exception as e:
            logging.error(f"Error in training step: {str(e)}")
            # Try to diagnose common issues
            if "CUDA out of memory" in str(e):
                logging.error("CUDA OOM detected. Try reducing batch size or model size.")
            elif "NCCL" in str(e):
                logging.error("NCCL error detected. Check network connectivity between GPUs.")
            raise e
            
        # Increment step count
        self.step_count += 1
        
        # Check if parameters actually changed every 50 steps as requested
        if self.step_count % 50 == 0 and local_rank in [-1, 0]:
            params_after = self._get_param_sample(model)
            if self.params_before is not None:
                try:
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
                except Exception as e:
                    logging.error(f"Error checking parameter changes: {str(e)}")
                    
        # Log loss occasionally
        if self.step_count % 20 == 0 and local_rank in [-1, 0]:
            logging.info(f"Step {self.step_count}: NPO loss = {loss.item():.4f}")

        # Clear cache after processing
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        return loss.detach()

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, **kwargs):
        """
        Override to handle proper logging in distributed training.
        Matches the signature expected by transformers.Trainer.
        Handles cases where grad_norm might not be a tensor.
        """
        if self.control.should_log:
            logs: Dict[str, float] = {}
            # Gather tr_loss across all processes for logging
            # Ensure tr_loss is a tensor before gathering
            if isinstance(tr_loss, torch.Tensor):
                tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
                 # Reset tr_loss to zero on all processes after gathering
                tr_loss -= tr_loss
            else: # If tr_loss is already scalar (e.g., float from single GPU)
                tr_loss_scalar = tr_loss
                tr_loss = 0.0 # Reset scalar loss


            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # Log gradient norm if available
            if grad_norm is not None:
                # Check if grad_norm is a tensor before gathering
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm_scalar = self._nested_gather(grad_norm).mean().item()
                else: # If grad_norm is already scalar (e.g., float)
                    grad_norm_scalar = grad_norm
                logs["grad_norm"] = round(grad_norm_scalar, 4)

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        # Evaluate
        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Save model checkpoint
        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        return tr_loss # Return the potentially modified tr_loss (e.g., reset to zero)
