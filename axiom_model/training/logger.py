import os
import csv
import logging
import wandb

logger = logging.getLogger(__name__)

class TrainingLogger:
    def __init__(self, log_dir="./logs", use_wandb=False, config=None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.log_dir, "train.log")
        self.metrics_file = os.path.join(self.log_dir, "metrics.csv")
        
        # Initialize CSV header if it doesn't exist
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "val_loss", "perplexity", "lr", "grad_norm", "tokens_per_sec", "gpu_allocated_gb"])
        
        if self.use_wandb:
            wandb.init(project="axiom-v2", config=config)

    def log_metrics(self, step, train_loss, val_loss=None, perplexity=None, lr=None, grad_norm=None, profiler_stats=None):
        # Format for terminal/log file
        log_str = f"Step {step} | Loss: {train_loss:.4f}"
        if val_loss is not None: log_str += f" | Val Loss: {val_loss:.4f}"
        if perplexity is not None: log_str += f" | PPL: {perplexity:.2f}"
        if lr is not None: log_str += f" | LR: {lr:.2e}"
        if profiler_stats:
            log_str += f" | Tok/s: {profiler_stats.get('tokens_per_sec', 0):.0f}"
        logger.info(log_str)
        
        # Write to CSV
        with open(self.metrics_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, 
                train_loss, 
                val_loss if val_loss is not None else "", 
                perplexity if perplexity is not None else "", 
                lr if lr is not None else "", 
                grad_norm if grad_norm is not None else "",
                profiler_stats.get('tokens_per_sec', "") if profiler_stats else "",
                profiler_stats.get('allocated_gb', "") if profiler_stats else ""
            ])

        # W&B
        if self.use_wandb:
            metrics = {"train/loss": train_loss}
            if val_loss is not None: metrics["val/loss"] = val_loss
            if perplexity is not None: metrics["val/perplexity"] = perplexity
            if lr is not None: metrics["train/lr"] = lr
            if grad_norm is not None: metrics["train/grad_norm"] = grad_norm
            if profiler_stats:
                metrics["perf/tokens_per_sec"] = profiler_stats.get('tokens_per_sec', 0)
                metrics["perf/gpu_allocated_gb"] = profiler_stats.get('allocated_gb', 0)
            wandb.log(metrics, step=step)

    def close(self):
        if self.use_wandb:
            wandb.finish()
