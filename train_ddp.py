import os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from axiom_model.core.model import AxiomV2
from axiom_model.training.trainer import Trainer
from axiom_model.training.dataloader import create_dataloader
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.evaluator import Evaluator
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
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
    rank, local_rank, world_size, is_distributed = setup()
    is_rank_zero = (rank == 0)
    
    set_seed(42 + rank)
    
    with open("axiom_model/configs/500M.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler_mgr = SchedulerManager(optimizer, config.get('scheduler', {}))
    
    train_loader, train_sampler = create_dataloader("./data/bin/train.bin", batch_size=4, seq_len=4096, is_distributed=is_distributed, is_train=True)
    val_loader, _ = create_dataloader("./data/bin/val.bin", batch_size=4, seq_len=4096, is_distributed=is_distributed, is_train=False)
    
    # 3. Setup Managers
    ckpt_mgr = CheckpointManager()
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)
    evaluator = Evaluator(model, val_loader, device, is_distributed)
    train_logger = TrainingLogger(use_wandb=False) if is_rank_zero else None
    
    # 4. Resume Checkpoint
    start_epoch, start_step, best_val_loss = ckpt_mgr.load(model, optimizer, scheduler_mgr, trainer.scaler)
    
    # 5. Train Loop
    if is_rank_zero:
        logger.info("Starting Phase 3 Pretraining...")
        
    train_iter = iter(train_loader)
    
    for step in range(start_step, config.get('max_steps', 10000)):
        try:
            x, y = next(train_iter)
        except StopIteration:
            if train_sampler:
                train_sampler.set_epoch(step)
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            
        x, y = x.to(device), y.to(device)
        
        if is_rank_zero:
            trainer.profiler.start_step(x.size(0), x.size(1))
            
        is_last_accum = (step + 1) % trainer.grad_accum_steps == 0
        loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_accum)
        
        if is_last_accum:
            scheduler_mgr.step()
            
        if is_rank_zero and is_last_accum:
            stats = trainer.profiler.end_step()
            stats.update(trainer.profiler.get_gpu_memory())
            
            if step % 10 == 0:
                train_logger.log_metrics(
                    step=step, 
                    train_loss=loss, 
                    lr=scheduler_mgr.get_lr(), 
                    grad_norm=grad_norm,
                    profiler_stats=stats
                )
                
            if step > 0 and step % config.get('eval_interval', 1000) == 0:
                val_loss, val_ppl = evaluator.evaluate()
                train_logger.log_metrics(step, loss, val_loss=val_loss, perplexity=val_ppl)
                
                is_best = val_loss < best_val_loss
                if is_best: best_val_loss = val_loss
                
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, 0, step, best_val_loss, config, is_best=is_best)
                
    if is_rank_zero:
        train_logger.close()
    cleanup()

if __name__ == "__main__":
    main()
