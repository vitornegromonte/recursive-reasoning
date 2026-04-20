import torch
import torch.nn as nn
import torch.nn.functional as F

class ACTLossHead(nn.Module):
    """
    Computes the loss for the ACT wrapper.
    Includes both the LM cross-entropy loss and the Q-learning halting loss.
    """
    def __init__(self, pad_idx=-100):
        super().__init__()
        self.pad_idx = pad_idx
        # Default Cross Entropy
        self.lm_loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="none")

    def forward(self, outputs: dict[str, torch.Tensor], labels: torch.Tensor, halted: torch.Tensor, steps: torch.Tensor):
        logits = outputs["logits"]
        q_halt_logits = outputs["q_halt_logits"]
        q_continue_logits = outputs["q_continue_logits"]

        # 1. Base LM Cross Entropy
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        lm_loss_full = self.lm_loss_fn(flat_logits, flat_labels).view(*labels.size())
        
        # Mask out padding (if applicable)
        mask = (labels != self.pad_idx)
        loss_counts = mask.sum(-1).clamp_min(1).float()  # (batch_size, )
        
        # Average LM loss per sequence
        lm_loss_seq = (lm_loss_full * mask.float()).sum(-1) / loss_counts
        lm_loss = lm_loss_seq.sum()  # Total over batch

        # 2. Extract correctness for the Q-learning Halting Loss
        # We want q_halt to predict whether the sequence is currently correct.
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == mask.sum(-1) # Only True if all valid tokens are correct

        # 3. Compute Q-Halt Loss (Binary Cross Entropy)
        # We want q_halt to be 1 if correct, 0 if incorrect
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, 
            seq_is_correct.to(q_halt_logits.dtype), 
            reduction="sum"
        )

        # 4. Compute Q-Continue Loss (Bootstrapped Target)
        q_continue_loss = 0.0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits, 
                outputs["target_q_continue"], 
                reduction="sum"
            )

        # Ensure metrics are detached for logging
        metrics = {
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }
        if "target_q_continue" in outputs:
            metrics["q_continue_loss"] = q_continue_loss.detach()
            
        # Accuracy metrics (for sequences that halted this step)
        with torch.no_grad():
            valid_metrics = halted & (mask.sum(-1) > 0)
            metrics["count"] = valid_metrics.sum()
            metrics["exact_accuracy"] = (valid_metrics & seq_is_correct).sum()
            metrics["steps"] = torch.where(valid_metrics, steps, 0).sum()

        # Total combined loss
        total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        return total_loss, metrics
