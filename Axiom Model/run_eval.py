import argparse
import json
import torch
import torch.nn.functional as F
import tiktoken
import os

from axiom_model.core.model import AxiomV2

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    from lm_eval import evaluator, tasks
except ImportError:
    print("Please install lm-eval: pip install lm-eval")
    raise

class AxiomLMEvalAdapter(LM):
    def __init__(self, checkpoint_path, device='cuda'):
        super().__init__()
        self._device = device
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        # Load Model
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
        model_cfg = checkpoint['config']['model']
        self.model = AxiomV2(
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
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval()

    @property
    def eot_token_id(self):
        # We don't have a strict EOT token in cl100k_base for base models, using <|endoftext|>
        return self.enc.eot_token

    @property
    def max_length(self):
        return self.model.max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1  # AxiomV2 does not currently support padding masks, so batch_size must be 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        tokens = self.enc.encode(string)
        if left_truncate_len is not None:
            tokens = tokens[-left_truncate_len:]
        return tokens

    def tok_decode(self, tokens):
        return self.enc.decode(tokens)

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        res = []
        with torch.inference_mode():
            for req in requests:
                context, continuation = req.args
                ctx_tokens = self.tok_encode(context)
                cont_tokens = self.tok_encode(continuation)
                
                # Check limits
                if len(ctx_tokens) + len(cont_tokens) > self.max_length:
                    # Truncate context to fit
                    ctx_tokens = ctx_tokens[-(self.max_length - len(cont_tokens)):]
                
                inps = torch.tensor(ctx_tokens + cont_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                
                logits = self.model(inps)
                logits = logits[0, len(ctx_tokens)-1 : len(ctx_tokens) + len(cont_tokens) - 1, :]
                
                cont_tensor = torch.tensor(cont_tokens, dtype=torch.long, device=self.device)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Gather log probs for the continuation tokens
                gathered_log_probs = torch.gather(log_probs, dim=-1, index=cont_tensor.unsqueeze(-1)).squeeze(-1)
                
                ll = gathered_log_probs.sum().item()
                
                # Calculate greedy exact match
                greedy_tokens = logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_tensor).all().item()
                
                res.append((ll, is_greedy))
                
        return res

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        res = []
        with torch.inference_mode():
            for req in requests:
                string = req.args[0]
                tokens = self.tok_encode(string)
                
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                    
                inps = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.model(inps)
                
                log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
                target_tokens = torch.tensor(tokens[1:], dtype=torch.long, device=self.device)
                
                gathered_log_probs = torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                res.append(gathered_log_probs.sum().item())
                
        return res

    def generate_until(self, requests: list[Instance]) -> list[str]:
        # Fallback to simple generate for tasks requiring text generation
        res = []
        from generate import generate
        for req in requests:
            context = req.args[0]
            until = req.args[1].get('until', [self.enc.decode([self.eot_token_id])])
            max_gen = req.args[1].get('max_gen_toks', 50)
            
            gen_tokens, _ = generate(self.model, self.enc, context, max_new_tokens=max_gen, temperature=0.0, use_kv_cache=True, stream=False)
            gen_text = self.enc.decode(gen_tokens)
            
            for stop in until:
                if stop in gen_text:
                    gen_text = gen_text[:gen_text.find(stop)]
            res.append(gen_text)
            
        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--tasks", type=str, default="arc_easy,piqa,hellaswag", help="Comma-separated list of tasks")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval samples (for fast testing)")
    args = parser.parse_args()
    
    print(f"Loading AxiomV2 Adapter from {args.checkpoint}...")
    lm = AxiomLMEvalAdapter(args.checkpoint)
    
    task_list = args.tasks.split(",")
    print(f"Running EleutherAI lm-eval on: {task_list}")
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=args.limit,
        device=lm.device,
        batch_size=1
    )
    
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    console.print(Panel(Text("🚀 PHASE 3 BASELINE EVALUATION COMPLETE", justify="center", style="bold green"), border_style="green"))
    
    table = Table(title="Axiom V2 (476M) - Immutable Control Baseline", show_header=True, header_style="bold magenta")
    table.add_column("Benchmark", style="cyan", justify="left")
    table.add_column("Metric", style="yellow", justify="left")
    table.add_column("Score", style="bold white", justify="right")
    
    for task_name, task_metrics in results['results'].items():
        for k, v in task_metrics.items():
            if k == "alias": continue
            if not k.endswith("stderr"):
                # Handle lists/dicts if any, otherwise format float
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                table.add_row(task_name.upper(), k, val_str)
                
    console.print(table)
                
    # Save to disk
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/baseline_report.json", "w") as f:
        json.dump(results, f, indent=4, default=str)
        
    print("\n✅ Baseline metrics saved to evaluation/baseline_report.json")
