import os
import time
import torch
import torch.nn.functional as F
import tiktoken
from axiom_model.core.model import AxiomV2

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def load_dpo_model(ckpt_path, device):
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    cfg = checkpoint['config']['model']
    model = AxiomV2(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        n_kv_heads=cfg['n_kv_heads'],
        max_seq_len=cfg['max_seq_len'],
        multiple_of=cfg['multiple_of'],
        norm_eps=cfg['norm_eps'],
        rope_theta=cfg['rope_theta'],
        gradient_checkpointing=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A'))
    step = checkpoint.get('step', 'N/A')
    print(f"✅ Phase 5 DPO Brain loaded! (Step: {step}, Loss: {val_loss})")
    return model

def ask_axiom(model, enc, user_query, device, max_new_tokens=250, temperature=0.2, top_p=0.9, rep_penalty=1.15):
    prompt = (
        "### System:\n"
        "You are a highly intelligent, logical, and helpful AI assistant named Axiom.\n\n"
        f"### User:\n{user_query}\n\n"
        "### Assistant:\n"
    )
    tokens = enc.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print(f"\n" + "="*70)
    print(f"💬 USER: {user_query}")
    print("="*70)
    print("🤖 AXIOM (DPO): ", end="", flush=True)
    
    kv_cache = None
    start_pos = 0
    generated = []
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        for i in range(max_new_tokens):
            if i == 0:
                cur_ids = input_ids
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = input_ids.shape[1] - 1
                
            try:
                logits, kv_cache = model(cur_ids, start_pos=start_pos, kv_cache=kv_cache, return_cache=True)
            except Exception as e:
                break
                
            next_logits = logits[:, -1, :]
            
            # Repetition Penalty
            if rep_penalty != 1.0:
                for t in input_ids[0]:
                    if next_logits[0, t] > 0:
                        next_logits[0, t] /= rep_penalty
                    else:
                        next_logits[0, t] *= rep_penalty
                        
            if temperature == 0.0:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                v, _ = torch.topk(next_logits, min(50, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_id = sample_top_p(probs, top_p)
                
            token_val = next_id.item()
            if token_val == 100257: # <|endoftext|>
                break
                
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated.append(token_val)
            print(enc.decode([token_val]), end="", flush=True)
            
    t1 = time.perf_counter()
    speed = len(generated) / (t1 - t0) if (t1 - t0) > 0 else 0
    print(f"\n" + "-"*70)
    print(f"Generated {len(generated)} tokens in {t1-t0:.2f}s ({speed:.1f} tok/s)")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Priority: checkpoints_dpo/best.pt -> checkpoints_dpo/latest.pt -> checkpoints_sft/best.pt
    ckpt = "checkpoints_dpo/best.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints_dpo/latest.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints_sft/best.pt"
    if not os.path.exists(ckpt):
        print(f"❌ Checkpoint not found at {ckpt}!")
        return
        
    model = load_dpo_model(ckpt, device)
    
    test_queries = [
        "Write a python code to add two numbers",
        "Write a python function to check if a number is prime.",
        "Who are you and what are your capabilities?",
        "What are the four nitrogenous bases in DNA?",
        "How do you make homemade mead?",
    ]
    
    print("\n🚀 Commencing Axiom V2 Phase 5 DPO Litmus Tests...\n")
    for q in test_queries:
        ask_axiom(model, enc, q, device)
        
    print("\n" + "="*70)
    print("🎉 DPO Verification Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
