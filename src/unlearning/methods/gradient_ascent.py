import torch
from torch import nn
from transformers import Trainer
from typing import Dict, Union, Any
import os
import logging

class GradientAscentTrainer(Trainer):
    """
    Custom Trainer implementing gradient ascent for unlearning.
    Works with both Sentence format and QA format with DeepSpeed support.
    """

    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None,
                 eval_dataset=None, tokenizer=None, model_init=None, compute_metrics=None,
                 callbacks=None, optimizers=(None, None), preprocess_logits_for_metrics=None,
                 use_QA_unlearning=False):  # Explicitly define all parameters, including use_QA_unlearning
        # Call the parent class initializer without custom arguments
        super().__init__(
            model=model, 
            args=args, 
            data_collator=data_collator, 
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer, 
            model_init=model_init, 
            compute_metrics=compute_metrics,
            callbacks=callbacks, 
            optimizers=optimizers, 
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # Initialize our own attributes
        self.use_QA_unlearning = use_QA_unlearning
        self.step_count = 0
        self.params_before = None
        self.debug_samples_logged = 0  # Track the number of samples logged

        # Verify tokenizer exists
        if self.tokenizer is None:
            logging.warning("No tokenizer provided to GradientAscentTrainer. Debugging output will be limited.")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss differently based on format (Sentence vs QA).

        For QA format, we calculate loss only on the answer tokens, not the question tokens.
        """
        if not self.use_QA_unlearning:
            # Sentence format - calculate loss on all tokens
            return super().compute_loss(model, inputs, return_outputs)

        # QA format - calculate loss only on answer tokens
        labels = inputs.pop("labels").clone()  # Clone to avoid modifying original input
        input_ids = inputs["input_ids"].clone()
        attention_mask = inputs["attention_mask"].clone()
        outputs = model(**inputs)
        logits = outputs.logits

        # Debug: analyze prediction probabilities at answer positions
        if self.tokenizer is not None and self.args.local_rank in [-1, 0]:
            batch_size = labels.shape[0]
            for i in range(min(batch_size, 3)):  # Display only first 3 samples
                try:
                    # Create a mask to identify answer tokens (non -100 labels)
                    answer_mask = (labels[i] != -100)

                    # Create a mask to identify padding tokens (where attention_mask is 0)
                    padding_mask = (attention_mask[i] == 0)

                    # Question tokens are those that are neither answer nor padding
                    question_mask = ~answer_mask & ~padding_mask

                    # Extract question part only (excluding answers and padding)
                    question_ids = input_ids[i][question_mask]
                    question_text = self.tokenizer.decode(question_ids, skip_special_tokens=True)

                    # Extract answer token positions
                    answer_positions = answer_mask.nonzero().flatten().tolist()

                    if len(answer_positions) > 0:
                        # Get the expected answer text
                        answer_ids = labels[i][answer_mask]
                        expected_answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

                        logging.info(f"\n==== Sample {i+1} Answer Token Analysis (Step {self.step_count}) ====")
                        logging.info(f"Pure Question: \"{question_text}\"")
                        logging.info(f"Expected Answer: '{expected_answer}'")
                        logging.info(f"Answer Token Positions: {answer_positions}")

                        # Inspect predicted probabilities at each answer position
                        for pos_idx, pos in enumerate(answer_positions):
                            # Get label ID (correct answer)
                            true_token_id = labels[i][pos].item()
                            true_token_text = self.tokenizer.decode([true_token_id])

                            # Get logits and probabilities at this position
                            position_logits = logits[i, pos]
                            position_probs = torch.softmax(position_logits, dim=-1)

                            # Check for key tokens like " Yes", " No", " Unknown"
                            key_answers = [" Yes", " No", " Unknown", "Yes", "No", "Unknown"]
                            key_answer_probs = []

                            for ans in key_answers:
                                try:
                                    ans_ids = self.tokenizer.encode(ans, add_special_tokens=False)
                                    if ans_ids:
                                        ans_id = ans_ids[0]
                                        prob = position_probs[ans_id].item()
                                        key_answer_probs.append((ans, ans_id, prob))
                                except Exception as token_err:
                                    logging.debug(f"Error getting token ID for '{ans}': {token_err}")

                            # Show correct label probability
                            logging.info(f"\n  Position {pos} (Answer Index {pos_idx+1}/{len(answer_positions)}):")
                            logging.info(f"  Correct Token: '{true_token_text}' (ID: {true_token_id}), Prob: {position_probs[true_token_id].item():.6f}")

                            # Show key answer token probabilities
                            logging.info("  Key Answer Token Probabilities:")
                            for ans_text, ans_id, prob in key_answer_probs:
                                logging.info(f"    '{ans_text}' (ID: {ans_id}): {prob:.6f}")

                            # Show top 5 most probable tokens
                            top_probs, top_tokens = position_probs.topk(5)
                            logging.info("  Top 5 Predicted Tokens:")
                            for token_id, prob in zip(top_tokens.tolist(), top_probs.tolist()):
                                token_text = self.tokenizer.decode([token_id])
                                logging.info(f"    '{token_text}' (ID: {token_id}): {prob:.6f}")

                    # Optional: use only question to predict next token
                    try:
                        if len(answer_positions) > 0:
                            first_answer_pos = answer_positions[0]
                            prefix_ids = input_ids[i][:first_answer_pos+1].unsqueeze(0)
                            prefix_mask = attention_mask[i][:first_answer_pos+1].unsqueeze(0)

                            with torch.no_grad():
                                prefix_inputs = {"input_ids": prefix_ids, "attention_mask": prefix_mask}
                                prefix_outputs = model(**prefix_inputs)
                                prefix_logits = prefix_outputs.logits

                            next_token_logits = prefix_logits[0, -1, :]
                            next_token_probs = torch.softmax(next_token_logits, dim=-1)

                            logging.info("\n  Predicting Next Token from Pure Question:")
                            for ans_text, ans_id, _ in key_answer_probs:
                                prob = next_token_probs[ans_id].item()
                                logging.info(f"    '{ans_text}' (ID: {ans_id}): {prob:.6f}")

                            top_probs, top_tokens = next_token_probs.topk(5)
                            logging.info("  Top 5 Predicted Tokens:")
                            for token_id, prob in zip(top_tokens.tolist(), top_probs.tolist()):
                                token_text = self.tokenizer.decode([token_id])
                                logging.info(f"    '{token_text}' (ID: {token_id}): {prob:.6f}")
                    except Exception as prefix_err:
                        logging.error(f"Error predicting next token from pure question: {str(prefix_err)}")

                    else:
                        logging.info(f"No answer tokens found (-100) in sample {i+1}")
                except Exception as e:
                    logging.error(f"Error analyzing answer tokens: {str(e)}")
                    import traceback
                    logging.error(traceback.format_exc())

        # Create a mask to identify which tokens are part of the answer (non-padding)
        loss_mask = (labels != -100)

        # Calculate loss only on answer tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        masked_loss = loss * loss_mask.view(-1).float()

        # Get mean loss over actual answer tokens
        loss = masked_loss.sum() / max(loss_mask.sum(), 1)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step with gradient ascent, compatible with DeepSpeed.

        Args:
            model: The model being trained.
            inputs: The input batch from the dataset.
            num_items_in_batch: Number of items in the batch (required by Trainer).

        Returns:
            torch.Tensor: The detached loss value after applying ascent.
        """
        # Save sample parameters for checking change
        if self.step_count % 50 == 0 and self.args.local_rank in [-1, 0]:
            self.params_before = self._get_param_sample(model)

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # Debug: Log original loss before negating
        if self.step_count % 10 == 0 and self.args.local_rank in [-1, 0]:
            logging.info(f"Original loss (gradient descent): {loss.item():.4f}")

        # Apply gradient ascent
        loss = -loss

        # Debug: Log negated loss
        if self.step_count % 10 == 0 and self.args.local_rank in [-1, 0]:
            logging.info(f"Gradient ascent loss: {loss.item():.4f}")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Handle different training backends
        if self.args.deepspeed:
            self.deepspeed.backward(loss)
            self.deepspeed.step()
        elif self.args.use_amp:
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        else:
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        self.step_count += 1

        # Check if parameters changed
        if self.step_count % 50 == 0 and self.args.local_rank in [-1, 0]:
            params_after = self._get_param_sample(model)
            if self.params_before is not None:
                params_changed = any(not torch.allclose(p_before, p_after) 
                                     for p_before, p_after in zip(self.params_before, params_after))
                if params_changed:
                    logging.info(f"Step {self.step_count}: Parameters successfully updated")
                    for i, (p_before, p_after) in enumerate(zip(self.params_before, params_after)):
                        if i < 3:
                            change = (p_after - p_before).abs().mean().item()
                            logging.info(f"  Param {i} Mean Abs Change: {change:.6f}")
                else:
                    logging.warning(f"Step {self.step_count}: Parameters NOT changing! Check optimizer settings.")

        return loss.detach()

    def _get_param_sample(self, model):
        """Get a sample of parameters to check if they're changing"""
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append(param.detach().clone())
                if len(params) >= 5:
                    break
        return params
