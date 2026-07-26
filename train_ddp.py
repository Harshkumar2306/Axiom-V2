import os
import sys
import yaml
import torch
import signal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse

from axiom_model.core.model import AxiomV2
from axiom_model.training.trainer import Trainer
from axiom_model.training.dataloader import create_dataloader
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.evaluator import Evaluator
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global flag for graceful exit
pause_requested = False

def handle_sigint(signum, frame):
    global pause_requested
    logger.info("\n[Graceful Exit] Pause signal received (SIGINT). Will save checkpoint and exit after current step...")
    pause_requested = True

from datetime import timedelta

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 30 minute timeout to prevent silent deadlocks on Kaggle
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description='Axiom V2 Pretraining')
    parser.add_argument('--config', type=str, default='axiom_model/configs/500M.yaml', help='Path to config file')
    parser.add_argument('--train_data', type=str, default='./data/bin/train.bin', help='Path to train.bin')
    parser.add_argument('--val_data', type=str, default='./data/bin/val.bin', help='Path to val.bin')
    return parser.parse_args()

def main():
    global pause_requested
    signal.signal(signal.SIGINT, handle_sigint)
    
    rank, local_rank, world_size, is_distributed = setup()
    is_rank_zero = (rank == 0)
    
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get('training', {}).get('seed', 42) + rank)
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 1. Build Model
    model_cfg = config['model']
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
        gradient_checkpointing=True
    ).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        
    # 2. Build Optimizer & Dataloaders
    train_cfg = config.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(train_cfg.get('learning_rate', 3e-4)), 
        weight_decay=train_cfg.get('weight_decay', 0.1)
    )
    scheduler_mgr = SchedulerManager(optimizer, config.get('scheduler', {}))
    
    train_loader, train_sampler = create_dataloader(
        args.train_data, 
        batch_size=train_cfg.get('batch_size', 8), 
        seq_len=model_cfg['max_seq_len'], 
        is_distributed=is_distributed, 
        is_train=True
    )
    val_loader, _ = create_dataloader(
        args.val_data, 
        batch_size=train_cfg.get('batch_size', 8), 
        seq_len=model_cfg['max_seq_len'], 
        is_distributed=is_distributed, 
        is_train=False
    )
    
    # 3. Setup Managers
    ckpt_mgr = CheckpointManager()
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)
    evaluator = Evaluator(model, val_loader, device, is_distributed, config)
    train_logger = TrainingLogger(use_wandb=False, max_steps=train_cfg.get('max_steps', 100000)) if is_rank_zero else None
    
    # 4. Resume Checkpoint
    start_epoch, start_step, best_val_loss = ckpt_mgr.load(model, optimizer, scheduler_mgr, trainer.scaler)
    
    # 5. Train Loop
    if is_rank_zero:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        effective_batch = train_cfg.get('batch_size', 2) * config.get('training', {}).get('grad_accum_steps', 16) * world_size
        tokens_per_step = effective_batch * model_cfg['max_seq_len']
        logger.info("\n" + "="*50)
        if start_step == 0:
            logger.info(f"🚀 AXIOM V2 ENGINE IGNITION")
        else:
            logger.info(f"🔄 AXIOM V2 RESUME SEQUENCE INITIATED")
        logger.info("="*50)
        logger.info(f"Model   : {total_params:.1f} Million Parameters")
        logger.info(f"Vocab   : {model_cfg['vocab_size']} (cl100k_base)")
        logger.info(f"Context : {model_cfg['max_seq_len']} Tokens")
        logger.info(f"Math    : Batch {train_cfg.get('batch_size', 2)} x {config.get('training', {}).get('grad_accum_steps', 16)} Accum x {world_size} GPUs = {effective_batch} Effective Batch")
        logger.info(f"Tokens  : {tokens_per_step:,} tokens per step")
        logger.info("="*50)
        if start_step == 0:
            logger.info(f"Starting Phase 3 Pretraining from scratch (Step 0)...")
        else:
            logger.info(f"Resuming from Checkpoint:")
            logger.info(f" -> Epoch        : {start_epoch}")
            logger.info(f" -> Global Step  : {start_step:,}")
            logger.info(f" -> Best Val Loss: {best_val_loss:.4f}")
            logger.info(f" -> Datastream   : Fast-forwarding {start_step:,} batches...")
        logger.info("="*50 + "\n")
        
    if train_sampler:
        train_sampler.set_epoch(start_epoch)
        
    train_iter = iter(train_loader)
    
    # Fast-forward iterator if resuming
    if start_step > 0:
        if is_rank_zero: logger.info(f"Fast-forwarding dataloader by {start_step} steps to resume...")
        for _ in range(start_step):
            try:
                next(train_iter)
            except StopIteration:
                start_epoch += 1
                if train_sampler: train_sampler.set_epoch(start_epoch)
                train_iter = iter(train_loader)
                next(train_iter)
                
    for step in range(start_step, train_cfg.get('max_steps', 100000)):
        
        # Check for pause file
        if os.path.exists("pause.flag"):
            if is_rank_zero: logger.info("\n[Graceful Exit] 'pause.flag' detected. Initiating pause sequence...")
            pause_requested = True
            os.remove("pause.flag")
            
        try:
            x, y = next(train_iter)
        except StopIteration:
            start_epoch += 1
            if train_sampler: train_sampler.set_epoch(start_epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        if is_rank_zero and (step % trainer.grad_accum_steps == 0):
            # Only start timer on the first micro-batch of the accumulation cycle
            trainer.profiler.start_step(x.size(0) * trainer.grad_accum_steps, x.size(1))
            
        is_last_accum = (step + 1) % trainer.grad_accum_steps == 0
        
        try:
            if is_distributed and not is_last_accum:
                with model.no_sync():
                    loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_accum)
            else:
                loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_accum)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[OOM Safety] Out of memory caught on step {step}. Clearing cache and skipping batch.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                elif hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
                
                # We need to manually cycle the iterator to drop the bad batch
                if is_last_accum: optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e
        
        if is_last_accum:
            scheduler_mgr.step()
            optimizer_step = (step + 1) // trainer.grad_accum_steps
            
        if is_rank_zero and is_last_accum:
            stats = trainer.profiler.end_step()
            stats.update(trainer.profiler.get_gpu_memory())
            
            if optimizer_step % 1 == 0:
                train_logger.log_metrics(
                    step=optimizer_step, 
                    train_loss=loss, 
                    lr=scheduler_mgr.get_lr(), 
                    grad_norm=grad_norm,
                    profiler_stats=stats
                )
                
            eval_interval = train_cfg.get('eval_interval', 1000)
            if optimizer_step > 0 and optimizer_step % eval_interval == 0:
                val_loss, val_ppl = evaluator.evaluate()
                train_logger.log_metrics(step, loss, val_loss=val_loss, perplexity=val_ppl)
                
                is_best = val_loss < best_val_loss
                if is_best: best_val_loss = val_loss
                
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, step, best_val_loss, config, is_best=is_best)
                
        # Graceful Pause Exit
        if pause_requested and is_last_accum:
            if is_rank_zero:
                logger.info(f"Saving paused checkpoint at step {step}...")
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, step, best_val_loss, config, is_best=False)
                train_logger.close()
            cleanup()
            sys.exit(0)
                
    if is_rank_zero:
        train_logger.close()
    cleanup()

if __name__ == "__main__":
    main()
