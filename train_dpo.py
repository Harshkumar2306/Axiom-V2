import os
import sys
import yaml
import torch
import signal
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from copy import deepcopy

from axiom_model.core.model import AxiomV2
from axiom_model.training.dpo_trainer import DPOTrainer
from axiom_model.training.dpo_dataloader import create_dpo_dataloader
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.utils.reproducibility import set_seed
from axiom_model.training.profiler import Profiler

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

def parse_args():
    parser = argparse.ArgumentParser(description='Axiom V2 Direct Preference Optimization (DPO)')
    parser.add_argument('--config', type=str, default='axiom_model/configs/500M.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default='./dataset/dpo/dpo_data.pt', help='Path to dpo_data.pt')
    parser.add_argument('--pretrained', type=str, default='./checkpoints_sft/best.pt', help='Path to pretrained SFT checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_dpo', help='Checkpoint output directory')
    parser.add_argument('--max_steps', type=int, default=500, help='Max optimization steps')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO Beta penalty coefficient')
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

    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Build Models (Policy & Reference)
    model_cfg = config['model']
    train_cfg = config.get('training', {})

    def build_model():
        return AxiomV2(
            vocab_size=model_cfg['vocab_size'],
            d_model=model_cfg['d_model'],
            n_layers=model_cfg['n_layers'],
            n_heads=model_cfg['n_heads'],
            n_kv_heads=model_cfg['n_kv_heads'],
            max_seq_len=model_cfg['max_seq_len'],
            multiple_of=model_cfg['multiple_of'],
            norm_eps=model_cfg['norm_eps'],
            rope_theta=model_cfg['rope_theta'],
            gradient_checkpointing=train_cfg.get('gradient_checkpointing', True)
        ).to(device)

    policy_model = build_model()
    ref_model = build_model()

    # Load SFT Weights into both models
    pretrained_path = args.pretrained
    if os.path.exists(pretrained_path):
        if is_rank_zero: logger.info(f"Loading SFT Base Model from {pretrained_path} into Policy and Reference Models...")
        state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        policy_model.load_state_dict(state['model'])
        ref_model.load_state_dict(state['model'])
    else:
        if is_rank_zero: logger.error(f"FATAL: DPO requires an SFT model, but {pretrained_path} was not found.")
        sys.exit(1)

    if is_distributed:
        policy_model = DDP(policy_model, device_ids=[local_rank])

    # 2. Build Optimizer & DPO Dataloaders
    # DPO LR is usually very small (e.g., 5e-6)
    dpo_lr = 5.0e-6
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=dpo_lr,
        weight_decay=train_cfg.get('weight_decay', 0.1),
        fused=(device.type == 'cuda')
    )

    max_opt_steps = args.max_steps
    scheduler_mgr = SchedulerManager(optimizer, {
        "type": "cosine",
        "T_max": max_opt_steps,
        "eta_min": 1e-7,
    })

    train_loader, _ = create_dpo_dataloader(
        args.data,
        batch_size=train_cfg.get('batch_size', 2), # DPO uses 2x memory, reduce batch size
        is_distributed=is_distributed,
        is_train=True
    )

    ckpt_mgr = CheckpointManager(save_dir=args.save_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    trainer = DPOTrainer(policy_model, ref_model, optimizer, scaler, beta=args.beta)
    train_logger = TrainingLogger(use_wandb=False, max_steps=max_opt_steps) if is_rank_zero else None
    profiler = Profiler(device)

    grad_accum = train_cfg.get('grad_accum_steps', 4)
    log_interval = 5

    if is_rank_zero:
        logger.info(f"Starting Phase 5 Direct Preference Optimization (DPO)...")
        logger.info(f"DPO config: lr={dpo_lr:.2e} | beta={args.beta} | max_steps={max_opt_steps} | grad_accum={grad_accum}")

    train_iter = iter(train_loader)
    best_loss = float('inf')

    for opt_step in range(max_opt_steps):
        # Check for pause file
        pause_tensor = torch.tensor([1 if (is_rank_zero and os.path.exists("pause.flag")) else 0], device=device)
        if is_distributed: dist.broadcast(pause_tensor, src=0)
        if pause_tensor.item() == 1:
            if is_rank_zero and not pause_requested: logger.info("\n[Graceful Exit] 'pause.flag' detected...")
            pause_requested = True

        accum_loss = 0.0
        accum_margins = 0.0
        grad_norm = None

        oom_flag = torch.tensor([0], device=device)
        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            chosen_ids = batch["chosen_ids"].to(device, non_blocking=True)
            chosen_labels = batch["chosen_labels"].to(device, non_blocking=True)
            rejected_ids = batch["rejected_ids"].to(device, non_blocking=True)
            rejected_labels = batch["rejected_labels"].to(device, non_blocking=True)

            is_last_micro = (micro == grad_accum - 1)

            if is_rank_zero and micro == 0:
                profiler.start_step(chosen_ids.size(0) * grad_accum, chosen_ids.size(1))

            try:
                if is_distributed and not is_last_micro:
                    with policy_model.no_sync():
                        loss, margins, grad_norm = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels, is_last_accum_step=is_last_micro)
                else:
                    loss, margins, grad_norm = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels, is_last_accum_step=is_last_micro)
                accum_loss += loss
                accum_margins += margins
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"[OOM Safety] OOM caught on Rank {dist.get_rank() if is_distributed else 0}. Clearing cache.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    oom_flag[0] = 1
                else:
                    raise e

            if is_distributed: dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            if oom_flag.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                break

        scheduler_mgr.step()
        current_step = opt_step + 1

        if is_rank_zero:
            stats = profiler.end_step()
            stats.update(profiler.get_gpu_memory())

            avg_loss = accum_loss / grad_accum
            avg_margins = accum_margins / grad_accum

            if current_step % log_interval == 0:
                logger.info(f"Step {current_step}/{max_opt_steps} | DPO Loss: {avg_loss:.4f} | Margin: {avg_margins:.4f}")
                train_logger.log_metrics(
                    step=current_step,
                    train_loss=avg_loss,
                    lr=scheduler_mgr.get_lr(),
                    grad_norm=grad_norm,
                    profiler_stats=stats
                )

            # Save checkpoints every 100 steps
            if current_step % 100 == 0 or current_step == max_opt_steps:
                is_best = avg_loss < best_loss
                if is_best: best_loss = avg_loss
                ckpt_mgr.save(policy_model, optimizer, scheduler_mgr, scaler, 0, current_step, best_loss, config, is_best=is_best)

        # Graceful Pause Exit
        if pause_requested:
            if is_rank_zero:
                ckpt_mgr.save(policy_model, optimizer, scheduler_mgr, scaler, 0, current_step, best_loss, config, is_best=False)
                train_logger.close()
                if os.path.exists("pause.flag"):
                    try: os.remove("pause.flag")
                    except OSError: pass
            if is_distributed: dist.barrier()
            cleanup()
            sys.exit(0)

    if is_rank_zero:
        train_logger.close()
    cleanup()

if __name__ == "__main__":
    main()
