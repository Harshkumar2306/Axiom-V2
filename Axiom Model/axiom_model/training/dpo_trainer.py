import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = True):
    """
    Computes the log probabilities of the given labels under the given logits.
    When average_log_prob=True (default), returns length-normalized log-probabilities,
    preventing DPO length bias (which unfairly penalizes longer, detailed chosen responses).
    """
    # Shift so that tokens < n predict n
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    
    # Calculate cross entropy (which is -log(p)) for each token
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        reduction='none'
    ).view(shifted_labels.size())
    
    # Ignore index (-100) padding
    loss_mask = (shifted_labels != -100)
    
    # loss is -log(p), so we return -loss as the log prob
    log_probs = -loss * loss_mask
    
    if average_log_prob:
        return log_probs.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)
    else:
        return log_probs.sum(dim=-1)

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Math Engine.
    Executes implicit reward modeling with length normalization and conservative SFT anchor loss.
    """
    def __init__(self, policy_model, ref_model, optimizer, scaler, beta=0.2, sft_weight=0.1, grad_accum_steps=1, clip_grad=1.0):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scaler = scaler
        self.beta = beta
        self.sft_weight = sft_weight
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad = clip_grad
        
        # Security Lock: Reference model must be absolutely frozen
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train_step(self, chosen_ids, chosen_labels, rejected_ids, rejected_labels, is_last_accum_step=True):
        self.policy_model.train()
        
        # Concatenate chosen and rejected to do a single forward pass
        # This completely avoids PyTorch's checkpointing inplace modification error with RoPE buffers
        combined_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        
        # Forward Reference Model (No gradients, saving VRAM)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                ref_combined_logits = self.ref_model(combined_ids)
                
                ref_chosen_logits = ref_combined_logits[:chosen_ids.size(0)]
                ref_rejected_logits = ref_combined_logits[chosen_ids.size(0):]
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels, average_log_prob=True)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels, average_log_prob=True)
                
        # >>> VRAM SAFETY: Aggressively free massive logit tensors before Policy forward pass <<<
        del ref_combined_logits, ref_chosen_logits, ref_rejected_logits
        
        # Forward Policy Model (Active Training)
        with torch.amp.autocast('cuda'):
            policy_combined_logits = self.policy_model(combined_ids)
            
            policy_chosen_logits = policy_combined_logits[:chosen_ids.size(0)]
            policy_rejected_logits = policy_combined_logits[chosen_ids.size(0):]
            
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels, average_log_prob=True)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels, average_log_prob=True)
            
            # Compute DPO Loss
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            logits = pi_logratios - ref_logratios
            
            # The magic DPO loss formula: -log(sigmoid(beta * logits))
            dpo_loss = -F.logsigmoid(self.beta * logits).mean()
            
            # Auxiliary SFT anchor loss on chosen sequence (prevents language drift / degenerate degradation)
            shifted_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
            shifted_chosen_labels = chosen_labels[..., 1:].contiguous()
            sft_loss = F.cross_entropy(
                shifted_chosen_logits.view(-1, shifted_chosen_logits.size(-1)),
                shifted_chosen_labels.view(-1),
                ignore_index=-100
            )
            
            # Combined Loss: DPO + SFT Anchor
            total_loss = dpo_loss + self.sft_weight * sft_loss
            
            # Implicit Reward monitoring (Optional, good for logging)
            chosen_rewards = (self.beta * (policy_chosen_logps - ref_chosen_logps)).detach()
            rejected_rewards = (self.beta * (policy_rejected_logps - ref_rejected_logps)).detach()
            reward_margins = chosen_rewards - rejected_rewards

            raw_loss = total_loss.item()
            # Mathematically scale loss by accumulation steps so accumulated gradients are normalized
            total_loss = total_loss / self.grad_accum_steps

        self.scaler.scale(total_loss).backward()
        
        grad_norm = None
        if is_last_accum_step:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.policy_model.parameters(), self.clip_grad)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                return raw_loss, reward_margins.mean().item(), None
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        return raw_loss, reward_margins.mean().item(), (grad_norm.item() if grad_norm is not None else None)
