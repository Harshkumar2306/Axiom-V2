import os
import sys
import yaml
import torch
import signal
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from axiom_model.core.model import AxiomV2
from axiom_model.training.trainer import Trainer
from axiom_model.training.sft_dataloader import create_sft_dataloader
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.evaluator import Evaluator
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pause_requested = False

def handle_sigint(signum, frame):
    global pause_requested
    logger.info("\n[Graceful Exit] Pause signal received. Exiting after current step...")
    pause_requested = True

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        from datetime import timedelta
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

def main():
    global pause_requested
    signal.signal(signal.SIGINT, handle_sigint)
    
    rank, local_rank, world_size, is_distributed = setup()
    is_rank_zero = (rank == 0)
    
    with open("axiom_model/configs/500M.yaml", 'r') as f:
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
        
    # 2. Build Optimizer & SFT Dataloaders
    train_cfg = config.get('training', {})
    
    # SFT Learning Rate is usually 10x smaller than pretraining
    sft_lr = 2.0e-5
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=sft_lr, 
        weight_decay=train_cfg.get('weight_decay', 0.1)
    )
    
    # Configure Cosine Decay without warm restarts for SFT
    sft_scheduler_cfg = {
        "type": "cosine",
        "T_max": 5000,
        "eta_min": 1e-6
    }
    scheduler_mgr = SchedulerManager(optimizer, sft_scheduler_cfg)
    
    train_loader, train_sampler = create_sft_dataloader(
        "./dataset/sft/sft_data.pt", 
        batch_size=train_cfg.get('batch_size', 4), 
        is_distributed=is_distributed, 
        is_train=True
    )
    
    # We use the same for val for now, or you can split dataset later
    val_loader, _ = create_sft_dataloader(
        "./dataset/sft/sft_data.pt", 
        batch_size=train_cfg.get('batch_size', 4), 
        is_distributed=is_distributed, 
        is_train=False
    )
    
    ckpt_mgr = CheckpointManager(save_dir="./checkpoints_sft")
    
    # 3. Load Pretrained Weights (Transfer Learning)
    pretrained_path = "./checkpoints/best.pt"
    if os.path.exists(pretrained_path):
        if is_rank_zero: logger.info(f"Loading Base Foundation Model from {pretrained_path} for SFT...")
        # We only load the model weights! We drop the optimizer/scheduler states for fine-tuning.
        state = torch.load(pretrained_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
    else:
        if is_rank_zero: logger.warning(f"Base model not found at {pretrained_path}. Training from scratch!")
        
    # Check if we are resuming an interrupted SFT run
    start_epoch, start_step, best_val_loss = ckpt_mgr.load(model, optimizer, scheduler_mgr, None, path="./checkpoints_sft/latest.pt")
    
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)
    evaluator = Evaluator(model, val_loader, device, is_distributed)
    train_logger = TrainingLogger(use_wandb=False) if is_rank_zero else None
    
    if is_rank_zero:
        logger.info(f"Starting Phase 4 Supervised Fine-Tuning from step {start_step}...")
        
    if train_sampler:
        train_sampler.set_epoch(start_epoch)
        
    train_iter = iter(train_loader)
    
    # Fast-forward iterator if resuming
    if start_step > 0:
        if is_rank_zero: logger.info(f"Fast-forwarding dataloader by {start_step} steps to resume SFT...")
        for _ in range(start_step):
            try:
                next(train_iter)
            except StopIteration:
                start_epoch += 1
                if train_sampler: train_sampler.set_epoch(start_epoch)
                train_iter = iter(train_loader)
                next(train_iter)
                
    for step in range(start_step, 5000): # SFT requires far fewer steps
        if os.path.exists("pause.flag"):
            if is_rank_zero: logger.info("\n[Graceful Exit] 'pause.flag' detected...")
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
        
        if is_rank_zero:
            trainer.profiler.start_step(x.size(0), x.size(1))
            
        is_last_accum = (step + 1) % trainer.grad_accum_steps == 0
        
        try:
            loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_accum)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"[OOM Safety] OOM caught. Clearing cache.")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                elif hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
                if is_last_accum: optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise e
        
        if is_last_accum:
            scheduler_mgr.step()
            
        if is_rank_zero and is_last_accum:
            stats = trainer.profiler.end_step()
            stats.update(trainer.profiler.get_gpu_memory())
            
            if step % 10 == 0:
                train_logger.log_metrics(step=step, train_loss=loss, lr=scheduler_mgr.get_lr(), grad_norm=grad_norm, profiler_stats=stats)
                
            eval_interval = 200
            if step > 0 and step % eval_interval == 0:
                val_loss, val_ppl = evaluator.evaluate()
                train_logger.log_metrics(step, loss, val_loss=val_loss, perplexity=val_ppl)
                
                is_best = val_loss < best_val_loss
                if is_best: best_val_loss = val_loss
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, step, best_val_loss, config, is_best=is_best)
                
        if pause_requested and is_last_accum:
            if is_rank_zero:
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, step, best_val_loss, config, is_best=False)
                train_logger.close()
            cleanup()
            sys.exit(0)
                
    if is_rank_zero:
        train_logger.close()
    cleanup()

if __name__ == "__main__":
    main()
