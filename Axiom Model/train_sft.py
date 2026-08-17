import os
import sys
import yaml
import torch
import signal
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import gc

from axiom_model.core.model import AxiomV2
from axiom_model.training.trainer import Trainer
from axiom_model.training.sft_dataloader import create_sft_dataloader
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.evaluator import Evaluator
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True, stream=sys.stdout)
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
    parser.add_argument('--pretrained', type=str, default='./best.pt', help='Path to pretrained base checkpoint')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_sft', help='Checkpoint output directory')
    parser.add_argument('--replay_data', type=str, default=None, help='Path to train.bin for 10% Mixed Replay SFT')
    parser.add_argument('--max_steps', type=int, default=None, help='Override max optimization steps')
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
    max_opt_steps = args.max_steps if args.max_steps is not None else int(sft_cfg.get('max_steps', 500))
    scheduler_mgr = SchedulerManager(optimizer, {
        "type": "cosine_warmup",
        "warmup_steps": int(max_opt_steps * 0.1),
        "max_steps": max_opt_steps,
        "eta_min_ratio": 0.05,
    })

    sft_data_path = args.data or sft_cfg.get('data_path', "./dataset/sft/sft_data.pt")
    
    if is_rank_zero and args.replay_data:
        logger.info(f"EXPERIMENT B DETECTED: Initializing Mixed Replay Dataloader (90% SFT / 10% Pretrain)")
        logger.info(f" - SFT: {sft_data_path}")
        logger.info(f" - Replay: {args.replay_data}")
    elif is_rank_zero:
        logger.info(f"EXPERIMENT A DETECTED: Initializing Pure SFT Dataloader (100% SFT)")
        
    train_loader, _ = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_cfg.get('batch_size', train_cfg.get('batch_size', 4)),
        is_distributed=is_distributed,
        is_train=True,
        replay_data_path=args.replay_data,
        seq_len=model_cfg['max_seq_len']
    )

    # We use the pure SFT dataset for validation regardless of experiment type
    val_loader, _ = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_cfg.get('batch_size', train_cfg.get('batch_size', 4)),
        is_distributed=is_distributed,
        is_train=False,
        replay_data_path=None
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
            
        del state
        gc.collect()
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

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    grad_accum = trainer.grad_accum_steps
    eval_interval = int(sft_cfg.get('eval_interval', 50))
    log_interval = int(sft_cfg.get('log_interval', 1))

    if is_rank_zero:
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        banner = (
            "\n" + "="*50 + "\n"
            f"🚀 AXIOM V2 PHASE 4 SFT ENGINE IGNITED\n"
            + "="*50 + "\n"
            f"Model        : {total_params:.1f}M Parameters\n"
            f"Base Brain   : {args.pretrained}\n"
            f"Dataset      : {sft_data_path} ({len(train_loader.dataset):,} Samples)\n"
            f"Learning Rate: {sft_lr:.2e} (10% Cosine Warmup)\n"
            f"Total Steps  : {max_opt_steps} Steps (1 Epoch)\n"
            f"GPUs         : {world_size}x GPUs (Batch: {train_cfg.get('batch_size', 1)} x {grad_accum} Accum)\n"
            f"Output Dir   : {args.save_dir}\n"
            + "="*50 + "\n"
            f"Starting Phase 4 Supervised Fine-Tuning from optimizer step {start_step}...\n"
        )
        logger.info(banner)

    train_loader.set_epoch(start_epoch)

    # Fast-forward iterator if resuming (start_step is in optimizer steps,
    # each of which consumed `grad_accum` micro-batches).
    if start_step > 0:
        if is_rank_zero: logger.info(f"Fast-forwarding dataloader by {start_step} steps instantly via Modulo Sampler slicing...")
        batches_to_skip = start_step * grad_accum
        train_loader.fast_forward(batches_to_skip)
        
    train_iter = iter(train_loader)

    for opt_step in range(start_step, max_opt_steps):

        # Check for pause file (Broadcast from Rank 0 to prevent DDP deadlocks)
        pause_tensor = torch.tensor([1 if (is_rank_zero and os.path.exists("pause.flag")) else 0], device=device)
        if is_distributed:
            dist.broadcast(pause_tensor, src=0)
        if pause_tensor.item() == 1:
            if is_rank_zero and not pause_requested:
                logger.info("\n[Graceful Exit] 'pause.flag' detected...")
            pause_requested = True

        accum_loss = 0.0
        grad_norm = None

        oom_flag = torch.tensor([0], device=device)
        for micro in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                start_epoch += 1
                train_loader.set_epoch(start_epoch)
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
                    logger.error(f"[OOM Safety] OOM caught on Rank {dist.get_rank() if is_distributed else 0}. Clearing cache.")
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    elif hasattr(torch.mps, 'empty_cache'): torch.mps.empty_cache()
                    oom_flag[0] = 1
                else:
                    raise e

            if is_distributed:
                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            if oom_flag.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                break

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
