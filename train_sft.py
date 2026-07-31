import os
import sys
import yaml
import torch
import signal
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description='Axiom V2 Supervised Fine-Tuning')
    parser.add_argument('--config', type=str, default='axiom_model/configs/500M.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Override path to sft_data.pt')
    parser.add_argument('--pretrained', type=str, default='./checkpoints/best.pt', help='Path to pretrained base checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_sft', help='Checkpoint output directory')
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

    # TF32 matmuls: free throughput on Ampere+ GPUs (harmless no-op on T4/CPU).
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Build Model
    model_cfg = config['model']
    train_cfg = config.get('training', {})
    sft_cfg = config.get('sft', {})

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
        gradient_checkpointing=train_cfg.get('gradient_checkpointing', True)
    ).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # 2. Build Optimizer & SFT Dataloaders
    # SFT Learning Rate is usually ~10x smaller than pretraining
    sft_lr = float(sft_cfg.get('learning_rate', 2.0e-5))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=sft_lr,
        weight_decay=train_cfg.get('weight_decay', 0.1),
        fused=(device.type == 'cuda')
    )

    # Plain cosine decay (no warm restarts) for SFT.
    max_opt_steps = int(sft_cfg.get('max_steps', 500))
    scheduler_mgr = SchedulerManager(optimizer, {
        "type": "cosine",
        "T_max": max_opt_steps,
        "eta_min": float(sft_cfg.get('eta_min', 1e-6)),
    })

    sft_data_path = args.data or sft_cfg.get('data_path', "./dataset/sft/sft_data.pt")
    train_loader, train_sampler = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_cfg.get('batch_size', train_cfg.get('batch_size', 4)),
        is_distributed=is_distributed,
        is_train=True
    )

    # We use the same file for val for now, or you can split the dataset later
    val_loader, _ = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_cfg.get('batch_size', train_cfg.get('batch_size', 4)),
        is_distributed=is_distributed,
        is_train=False
    )

    ckpt_mgr = CheckpointManager(save_dir=args.save_dir)

    # 3. Load Pretrained Weights (Transfer Learning)
    pretrained_path = args.pretrained
    if os.path.exists(pretrained_path):
        if is_rank_zero: logger.info(f"Loading Base Foundation Model from {pretrained_path} for SFT...")
        # We only load the model weights! We drop the optimizer/scheduler states for fine-tuning.
        try:
            state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        except TypeError:
            state = torch.load(pretrained_path, map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(state['model'])
        else:
            model.load_state_dict(state['model'])
    else:
        if is_rank_zero: logger.warning(f"Base model not found at {pretrained_path}. Training from scratch!")

    # Trainer is created BEFORE the resume-load so the GradScaler state can be
    # restored alongside optimizer/scheduler (previously it was silently reset).
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)

    # Check if we are resuming an interrupted SFT run.
    # Note: for SFT, `step` is tracked in OPTIMIZER steps (not micro-steps).
    start_epoch, start_step, best_val_loss = ckpt_mgr.load(
        model, optimizer, scheduler_mgr, trainer.scaler,
        path=os.path.join(args.save_dir, "latest.pt")
    )

    evaluator = Evaluator(model, val_loader, device, is_distributed, config)
    train_logger = TrainingLogger(use_wandb=False, max_steps=max_opt_steps) if is_rank_zero else None

    grad_accum = trainer.grad_accum_steps
    eval_interval = int(sft_cfg.get('eval_interval', 50))
    log_interval = int(sft_cfg.get('log_interval', 10))

    if is_rank_zero:
        logger.info(f"Starting Phase 4 Supervised Fine-Tuning from optimizer step {start_step}...")
        logger.info(f"SFT config: lr={sft_lr:.2e} | max_steps={max_opt_steps} | grad_accum={grad_accum} | eval every {eval_interval} steps")

    if train_sampler:
        train_sampler.set_epoch(start_epoch)

    train_iter = iter(train_loader)

    # Fast-forward iterator if resuming (start_step is in optimizer steps,
    # each of which consumed `grad_accum` micro-batches).
    if start_step > 0:
        batches_to_skip = start_step * grad_accum
        if is_rank_zero: logger.info(f"Fast-forwarding dataloader by {batches_to_skip} micro-batches to resume SFT...")
        for _ in range(batches_to_skip):
            try:
                next(train_iter)
            except StopIteration:
                start_epoch += 1
                if train_sampler: train_sampler.set_epoch(start_epoch)
                train_iter = iter(train_loader)
                next(train_iter)

    for opt_step in range(start_step, max_opt_steps):

        # Check for pause file (detection only — removal happens once, on rank 0, at exit)
        if os.path.exists("pause.flag"):
            if is_rank_zero and not pause_requested:
                logger.info("\n[Graceful Exit] 'pause.flag' detected...")
            pause_requested = True

        accum_loss = 0.0
        grad_norm = None

        for micro in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                start_epoch += 1
                if train_sampler: train_sampler.set_epoch(start_epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            is_last_micro = (micro == grad_accum - 1)

            if is_rank_zero and micro == 0:
                trainer.profiler.start_step(x.size(0) * grad_accum, x.size(1))

            try:
                if is_distributed and not is_last_micro:
                    with model.no_sync():
                        loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_micro)
                else:
                    loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_micro)
                accum_loss += loss
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"[OOM Safety] OOM caught. Clearing cache.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
                    if is_last_micro: optimizer.zero_grad(set_to_none=True)
                    break
                else:
                    raise e

        scheduler_mgr.step()
        current_step = opt_step + 1  # 1-based count of completed optimizer steps

        if is_rank_zero:
            stats = trainer.profiler.end_step()
            stats.update(trainer.profiler.get_gpu_memory())

            if current_step % log_interval == 0:
                train_logger.log_metrics(
                    step=current_step,
                    train_loss=accum_loss / grad_accum,
                    lr=scheduler_mgr.get_lr(),
                    grad_norm=grad_norm,
                    profiler_stats=stats
                )

        # Evaluation: MUST run on every rank — evaluate() contains a
        # dist.all_reduce, so gating it behind is_rank_zero deadlocks DDP.
        if current_step % eval_interval == 0 or current_step == max_opt_steps:
            val_loss, val_ppl = evaluator.evaluate(num_batches=sft_cfg.get('eval_iters', 50))

            if is_rank_zero:
                train_logger.log_metrics(current_step, accum_loss / grad_accum, val_loss=val_loss, perplexity=val_ppl)

                is_best = val_loss < best_val_loss
                if is_best: best_val_loss = val_loss
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, current_step, best_val_loss, config, is_best=is_best)

            if is_distributed:
                dist.barrier()

        # Graceful Pause Exit
        if pause_requested:
            if is_rank_zero:
                ckpt_mgr.save(model, optimizer, scheduler_mgr, trainer.scaler, start_epoch, current_step, best_val_loss, config, is_best=False)
                train_logger.close()
                if os.path.exists("pause.flag"):
                    try: os.remove("pause.flag")
                    except OSError: pass

            if is_distributed:
                dist.barrier()
            cleanup()
            sys.exit(0)

    if is_rank_zero:
        train_logger.close()
    cleanup()

if __name__ == "__main__":
    main()
