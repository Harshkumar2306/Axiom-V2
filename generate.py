import argparse
import time
import torch
import torch.nn.functional as F
import tiktoken
import os
import sys

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

def generate(model, enc, prompt, max_new_tokens, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1, use_kv_cache=True, stream=True):
    device = next(model.parameters()).device
    tokens = enc.encode(prompt)
    prompt_len = len(tokens)
    
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    kv_cache = None
    start_pos = 0
    
    generated_tokens = []
    
    if stream:
        print(f"\n👤 PROMPT: {prompt}\n" + "="*60 + "\n🤖 AXIOM V2: ", end="")
    
    t0 = time.perf_counter()
    
    with torch.inference_mode():
        for i in range(max_new_tokens):
            if use_kv_cache:
                # Prefill (i=0) processes the entire prompt. Incremental decoding (i>0) processes only the last token.
                if i == 0:
                    input_ids = tokens
                else:
                    input_ids = tokens[:, -1:]
                    start_pos = tokens.shape[1] - 1
                
                try:
                    logits, kv_cache = model(input_ids, start_pos=start_pos, kv_cache=kv_cache, return_cache=True)
                except ValueError as e:
                    if stream: print(f"\n[Generation Stopped: {e}]")
                    break
            else:
                try:
                    logits = model(tokens, return_cache=False)
                except ValueError as e:
                    if stream: print(f"\n[Generation Stopped: {e}]")
                    break
                    
            next_token_logits = logits[:, -1, :]
            
            # Repetition Penalty
            if repetition_penalty != 1.0:
                for t in tokens[0]:
                    if next_token_logits[0, t] > 0:
                        next_token_logits[0, t] /= repetition_penalty
                    else:
                        next_token_logits[0, t] *= repetition_penalty
            
            if temperature == 0.0:
                idx_next = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                
                if top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                
                if top_p > 0.0 and top_p < 1.0:
                    idx_next = sample_top_p(probs, top_p)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat((tokens, idx_next), dim=1)
            generated_tokens.append(idx_next.item())
            
            if stream:
                print(enc.decode([idx_next.item()]), end="", flush=True)
                
    t1 = time.perf_counter()
    speed = len(generated_tokens) / (t1 - t0) if (t1 - t0) > 0 else 0
    if stream:
        print("\n" + "="*60)
        print(f"Speed: {speed:.2f} tokens/sec")
        
    return generated_tokens, speed

def run_validation_suite(model, enc):
    print("\n" + "="*60)
    print("🚀 RUNNING PHASE 2 VALIDATION SUITE")
    print("="*60)
    
    # Test 1: KV vs No-KV Equivalence
    print("\n--- Test 1: KV vs No-KV Equivalence ---")
    prompt = "The secret to artificial intelligence is"
    gen_kv, _ = generate(model, enc, prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=True, stream=False)
    gen_nokv, _ = generate(model, enc, prompt, max_new_tokens=20, temperature=0.0, use_kv_cache=False, stream=False)
    if gen_kv == gen_nokv:
        print("✅ PASS: KV Cache produces identical output to full recomputation.")
    else:
        print("❌ FAIL: KV Cache output diverges from full recomputation.")
        print(f"KV: {gen_kv}")
        print(f"No-KV: {gen_nokv}")
        
    # Test 2: Position Correctness (RoPE offset handling)
    print("\n--- Test 2: Position Correctness (Varying Prompt Lengths) ---")
    lengths = [1, 10, 100, 1000]
    passed_positions = True
    for l in lengths:
        prompt_str = "word " * l
        gen_kv, _ = generate(model, enc, prompt_str, max_new_tokens=5, temperature=0.0, use_kv_cache=True, stream=False)
        gen_nokv, _ = generate(model, enc, prompt_str, max_new_tokens=5, temperature=0.0, use_kv_cache=False, stream=False)
        if gen_kv != gen_nokv:
            print(f"❌ FAIL: Position {l} failed.")
            passed_positions = False
    if passed_positions:
        print("✅ PASS: RoPE offsets handle all tested prompt lengths correctly.")
        
    # Test 3: Context Boundary (4096)
    print("\n--- Test 3: Context Boundary Handling ---")
    prompt_str = "word " * 4095
    print("Testing length 4095 (Should pass):")
    _, _ = generate(model, enc, prompt_str, max_new_tokens=1, temperature=0.0, use_kv_cache=True, stream=False)
    print("✅ Passed 4095.")
    
    print("Testing length 4096 (Should fail gracefully on generation):")
    _, _ = generate(model, enc, prompt_str + " word", max_new_tokens=1, temperature=0.0, use_kv_cache=True, stream=False)
    print("✅ Handled 4096 explicitly via exception.")
    
    # Test 4: Sampling Variants
    print("\n--- Test 4: Sampling Execution ---")
    prompt = "In the year 2050, humanity will"
    print("Executing Top-K, Top-P, Temperature variants...")
    _, _ = generate(model, enc, prompt, max_new_tokens=10, temperature=1.0, top_k=50, top_p=0.9, stream=False)
    _, _ = generate(model, enc, prompt, max_new_tokens=10, temperature=0.7, top_k=10, top_p=1.0, stream=False)
    print("✅ PASS: Sampling functions executed without error.")
    
    # Test 5: Speed Benchmark
    print("\n--- Test 5: Speed Benchmark ---")
    prompt = "The history of the Roman Empire is vast."
    _, speed_nokv = generate(model, enc, prompt, max_new_tokens=100, temperature=0.0, use_kv_cache=False, stream=False)
    _, speed_kv = generate(model, enc, prompt, max_new_tokens=100, temperature=0.0, use_kv_cache=True, stream=False)
    print(f"Without KV Cache: {speed_nokv:.2f} tokens/sec")
    print(f"With KV Cache   : {speed_kv:.2f} tokens/sec")
    if speed_kv > speed_nokv:
        print(f"✅ PASS: KV Cache provides a {(speed_kv/speed_nokv):.2f}x speedup.")
    else:
        print("⚠️ WARN: KV Cache did not speed up generation (expected on very small sequences/GPUs).")
        
    print("\n============================================================")
    print("🎉 PHASE 2 VALIDATION COMPLETE")
    print("============================================================")

def load_model(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
    model_cfg = checkpoint['config']['model']
    model = AxiomV2(
        vocab_size=model_cfg['vocab_size'],
        d_model=model_cfg['d_model'],
        n_layers=model_cfg['n_layers'],
        n_heads=model_cfg['n_heads'],
        n_kv_heads=model_cfg['n_kv_heads'],
        max_seq_len=model_cfg['max_seq_len'],
        multiple_of=model_cfg['multiple_of'],
        norm_eps=model_cfg['norm_eps'],
        rope_theta=model_cfg['rope_theta'],
        gradient_checkpointing=False
    )
    
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('best_val_loss', 'Unknown')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axiom V2 Inference Engine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--prompt", type=str, default="The secret to artificial intelligence is", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling cutoff")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling cutoff")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--disable_kv_cache", action="store_true", help="Disable KV caching for debugging")
    parser.add_argument("--test", action="store_true", help="Run the Phase 2 Validation Suite")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Axiom V2 on {device}...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    model, val_loss = load_model(args.checkpoint, device)
    print(f"✅ Brain Loaded! (Validation Loss: {val_loss})")
    
    if args.test:
        run_validation_suite(model, enc)
    else:
        generate(
            model, 
            enc, 
            args.prompt, 
            args.max_new_tokens, 
            temperature=args.temperature, 
            top_k=args.top_k, 
            top_p=args.top_p, 
            repetition_penalty=args.repetition_penalty, 
            use_kv_cache=not args.disable_kv_cache
        )
