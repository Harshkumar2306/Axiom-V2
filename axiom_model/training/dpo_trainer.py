import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, average_log_prob: bool = False):
    """
    Computes the log probabilities of the given labels under the given logits.
    """
    # Shift so that tokens < n predict n
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    
    # Calculate cross entropy (which is -log(p)) for each token
    # We use reduction='none' so we can sum across the sequence
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
        return log_probs.sum(dim=-1) / loss_mask.sum(dim=-1)
    else:
        return log_probs.sum(dim=-1)

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Math Engine.
    Executes the implicit reward modeling loss function without needing an actual reward model.
    """
    def __init__(self, policy_model, ref_model, optimizer, scaler, beta=0.1):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scaler = scaler
        self.beta = beta
        
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
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)
                
        # >>> VRAM SAFETY: Aggressively free massive logit tensors before Policy forward pass <<<
        del ref_combined_logits, ref_chosen_logits, ref_rejected_logits
        
        # Forward Policy Model (Active Training)
        with torch.amp.autocast('cuda'):
            policy_combined_logits = self.policy_model(combined_ids)
            
            policy_chosen_logits = policy_combined_logits[:chosen_ids.size(0)]
            policy_rejected_logits = policy_combined_logits[chosen_ids.size(0):]
            
            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)
            
            # Compute DPO Loss
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            logits = pi_logratios - ref_logratios
            
            # The magic DPO loss formula: -log(sigmoid(beta * logits))
            loss = -F.logsigmoid(self.beta * logits).mean()
            
            # Implicit Reward monitoring (Optional, good for logging)
            chosen_rewards = (self.beta * (policy_chosen_logps - ref_chosen_logps)).detach()
            rejected_rewards = (self.beta * (policy_rejected_logps - ref_rejected_logps)).detach()
            reward_margins = chosen_rewards - rejected_rewards

        self.scaler.scale(loss).backward()
        
        grad_norm = None
        if is_last_accum_step:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
        return loss.item(), reward_margins.mean().item(), grad_norm
