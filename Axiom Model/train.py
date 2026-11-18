#!/usr/bin/env python3
"""
Axiom V2 - Unified Training & Benchmarking Engine CLI
=====================================================
The unified, authoritative entry point for:
  1. pretrain  - Distributed next-token autoregressive pretraining with memory-mapped streaming
  2. sft       - Supervised fine-tuning from base weights with prompt masking & anti-forgetting replay
  3. eval      - Validation evaluation & perplexity analysis
  4. benchmark - Hardware throughput, MFU, and SDPA kernel benchmarking

Usage:
  python train.py pretrain --config axiom_model/configs/500M.yaml --train_data ./data/bin/train.bin
  python train.py sft --config axiom_model/configs/500M.yaml --pretrained best.pt --data dataset/sft/sft_data.pt
  python train.py eval --checkpoint best.pt --val_data ./data/bin/val.bin
  python train.py benchmark --batch_size 2 --seq_len 512
"""

import os
import sys
import time
import math
import yaml
import signal
import argparse
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from axiom_model.core.model import AxiomV2
from axiom_model.training.trainer import Trainer
from axiom_model.training.dataloader import create_dataloader, MemmapDataset
from axiom_model.training.sft_dataloader import create_sft_dataloader, FastForwardSampler
from axiom_model.training.checkpoint import CheckpointManager
from axiom_model.training.evaluator import Evaluator
from axiom_model.training.logger import TrainingLogger
from axiom_model.training.scheduler import SchedulerManager
from axiom_model.training.profiler import Profiler
from axiom_model.utils.reproducibility import set_seed, capture_environment_snapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("AxiomEngine")

# Global pause state for signal handling
pause_requested = False


def handle_sigint(signum, frame):
    global pause_requested
    logger.info("\n[Signal] SIGINT received. Initiating graceful pause & checkpoint sequence...")
    pause_requested = True


def setup_distributed() -> Tuple[int, int, int, bool]:
    """Initialize distributed process group if running under torchrun / DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        from datetime import timedelta
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    return 0, 0, 1, False


def cleanup_distributed():
    """Tear down distributed process group safely."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: int = 0) -> torch.device:
    """Detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(config_path: str) -> dict:
    """Load YAML experiment configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =====================================================================
# SUBCOMMAND: PRETRAIN
# =====================================================================
def run_pretrain(args):
    global pause_requested
    signal.signal(signal.SIGINT, handle_sigint)

    rank, local_rank, world_size, is_distributed = setup_distributed()
    is_rank_zero = (rank == 0)

    config = load_config(args.config)
    set_seed(config.get("training", {}).get("seed", 42) + rank)

    device = get_device(local_rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = config["model"]
    train_cfg = config.get("training", {})

    if is_rank_zero:
        logger.info(f"Initializing Axiom V2 Pretraining Engine on {device.type.upper()} ({world_size} rank(s))")

    # 1. Build Model
    model = AxiomV2(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        multiple_of=model_cfg["multiple_of"],
        norm_eps=model_cfg["norm_eps"],
        rope_theta=model_cfg["rope_theta"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
    ).to(device)

    # Optional torch.compile on Ampere/Hopper CUDA
    if train_cfg.get("compile", False) and hasattr(torch, "compile") and device.type == "cuda":
        if is_rank_zero:
            logger.info("Enabling torch.compile() for max kernel fusion...")
        model = torch.compile(model)

    raw_model = model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw_model = model.module

    # 2. Optimizer with decoupled weight decay & fused AdamW
    optimizer = raw_model.configure_optimizers(
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        learning_rate=float(train_cfg.get("learning_rate", 3.0e-4)),
        betas=(0.9, 0.95),
        device_type=device.type
    )

    scheduler_mgr = SchedulerManager(optimizer, config.get("scheduler", {}))

    # 3. Memory-Mapped Token Dataloaders
    train_data_path = args.train_data or "./data/bin/train.bin"
    val_data_path = args.val_data or "./data/bin/val.bin"

    if not os.path.exists(train_data_path):
        if is_rank_zero:
            logger.error(f"Training binary dataset not found: {train_data_path}")
        cleanup_distributed()
        sys.exit(1)

    batch_size = train_cfg.get("batch_size", 2)
    seq_len = model_cfg["max_seq_len"]

    train_loader, train_sampler = create_dataloader(
        train_data_path,
        batch_size=batch_size,
        seq_len=seq_len,
        is_distributed=is_distributed,
        is_train=True
    )

    has_val = os.path.exists(val_data_path)
    val_loader = None
    if has_val:
        val_loader, _ = create_dataloader(
            val_data_path,
            batch_size=batch_size,
            seq_len=seq_len,
            is_distributed=is_distributed,
            is_train=False
        )

    # 4. Managers & Resume Checkpoint
    output_dir = args.output_dir or "./checkpoints"
    ckpt_mgr = CheckpointManager(save_dir=output_dir)
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)
    evaluator = Evaluator(model, val_loader, device, is_distributed, config) if has_val else None

    max_opt_steps = args.max_steps or train_cfg.get("max_steps", 100000)
    train_logger = TrainingLogger(
        log_dir=os.path.join(output_dir, "logs"),
        use_wandb=args.wandb,
        config=config,
        max_steps=max_opt_steps
    ) if is_rank_zero else None

    latest_ckpt_path = os.path.join(output_dir, "latest.pt")
    start_epoch, start_step, best_val_loss = 0, 0, float("inf")
    if os.path.exists(latest_ckpt_path) or args.resume:
        load_path = args.resume if (args.resume and os.path.exists(args.resume)) else latest_ckpt_path
        start_epoch, start_step, best_val_loss = ckpt_mgr.load(
            raw_model, optimizer, scheduler_mgr, trainer.scaler, path=load_path
        )
        start_step += 1  # Resume at next micro-batch

    if is_rank_zero:
        total_params = raw_model.get_num_params(non_embedding=False) / 1e6
        grad_accum = trainer.grad_accum_steps
        effective_batch = batch_size * grad_accum * world_size
        tokens_per_step = effective_batch * seq_len
        logger.info("=" * 60)
        logger.info("🚀 AXIOM V2 PRETRAINING ENGINE LAUNCHED")
        logger.info(f"Model Parameters   : {total_params:.1f} M")
        logger.info(f"Context Length     : {seq_len} tokens")
        logger.info(f"Effective Batch    : {batch_size} x {grad_accum} accum x {world_size} GPUs = {effective_batch} ({tokens_per_step:,} tokens/step)")
        logger.info(f"Precision Mode     : {trainer.amp_dtype}")
        logger.info(f"Target Max Steps   : {max_opt_steps:,} optimizer steps")
        logger.info(f"Starting Micro-Step: {start_step:,}")
        logger.info("=" * 60)

    # 5. Fast-Forward Resumption
    if train_sampler:
        train_sampler.set_epoch(start_epoch)

    if start_step > 0 and train_sampler is not None:
        if is_rank_zero:
            logger.info(f"Fast-forwarding dataset stream by {start_step:,} micro-batches...")
        class FastForwardMemmapSampler:
            def __init__(self, base_sampler, skip_batches, bsz):
                self.base_sampler = base_sampler
                self.total = len(base_sampler)
                total_skip = skip_batches * bsz
                self.skip = total_skip % self.total if self.total > 0 else 0
            def __iter__(self):
                import itertools
                it = iter(self.base_sampler)
                list(itertools.islice(it, self.skip))
                self.skip = 0
                return it
            def __len__(self):
                return self.total - self.skip
            def set_epoch(self, epoch):
                if hasattr(self.base_sampler, "set_epoch"):
                    self.base_sampler.set_epoch(epoch)

        train_sampler = FastForwardMemmapSampler(train_loader.sampler, start_step, train_loader.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=False,
            drop_last=train_loader.drop_last
        )

    train_iter = iter(train_loader)
    grad_accum_steps = trainer.grad_accum_steps
    max_micro_steps = max_opt_steps * grad_accum_steps

    # 6. Pretraining Loop
    for step in range(start_step, max_micro_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            start_epoch += 1
            if train_sampler:
                train_sampler.set_epoch(start_epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        is_first_accum = (step % grad_accum_steps == 0)
        is_last_accum = ((step + 1) % grad_accum_steps == 0)

        if is_rank_zero and is_first_accum and trainer.profiler:
            trainer.profiler.start_step(x.size(0) * grad_accum_steps, x.size(1))

        # DDP no_sync() optimization: bypass inter-GPU all-reduce on micro-batches
        if is_distributed and not is_last_accum:
            with model.no_sync():
                loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=False)
        else:
            loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_accum)

        if is_last_accum:
            scheduler_mgr.step()
            opt_step = (step + 1) // grad_accum_steps

            if is_rank_zero:
                stats = trainer.profiler.end_step() if trainer.profiler else {}
                if trainer.profiler:
                    stats.update(trainer.profiler.get_gpu_memory())

                # Model FLOPs Utilization
                if stats.get("time_sec", 0) > 0:
                    try:
                        mfu = raw_model.estimate_mfu(
                            fwdbwd_per_iter=batch_size * grad_accum_steps * world_size,
                            dt=stats["time_sec"]
                        )
                        stats["mfu_pct"] = mfu * 100.0
                    except Exception:
                        pass

                if train_logger:
                    train_logger.log_metrics(
                        step=opt_step,
                        train_loss=loss,
                        lr=scheduler_mgr.get_lr(),
                        grad_norm=grad_norm,
                        profiler_stats=stats
                    )

            # Periodic Validation & Checkpointing
            eval_interval = train_cfg.get("eval_interval", 500)
            if evaluator and (opt_step > 0) and (opt_step % eval_interval == 0):
                val_loss, val_ppl = evaluator.evaluate(num_batches=train_cfg.get("eval_iters", 50))
                if is_rank_zero:
                    if train_logger:
                        train_logger.log_metrics(opt_step, loss, val_loss=val_loss, perplexity=val_ppl)
                    is_best = val_loss < best_val_loss
                    if is_best:
                        best_val_loss = val_loss
                    ckpt_mgr.save(
                        model, optimizer, scheduler_mgr, trainer.scaler,
                        start_epoch, step, best_val_loss, config, is_best=is_best
                    )

                if is_distributed:
                    dist.barrier()

            # Synchronized Pause Flag Check
            pause_val = 1 if (is_rank_zero and (pause_requested or os.path.exists("pause.flag"))) else 0
            pause_tensor = torch.tensor([pause_val], device=device)
            if is_distributed:
                dist.broadcast(pause_tensor, src=0)
            if pause_tensor.item() == 1:
                pause_requested = True

        # Graceful Pause Exit
        if pause_requested and is_last_accum:
            if is_rank_zero:
                logger.info(f"Saving paused checkpoint at micro-step {step}...")
                ckpt_mgr.save(
                    model, optimizer, scheduler_mgr, trainer.scaler,
                    start_epoch, step, best_val_loss, config, is_best=False
                )
                if train_logger:
                    train_logger.close()
                if os.path.exists("pause.flag"):
                    try:
                        os.remove("pause.flag")
                    except OSError:
                        pass
            if is_distributed:
                dist.barrier()
            cleanup_distributed()
            sys.exit(0)

    if is_rank_zero:
        logger.info("Pretraining completed successfully!")
        if train_logger:
            train_logger.close()
    cleanup_distributed()


# =====================================================================
# SUBCOMMAND: SFT (SUPERVISED FINE-TUNING)
# =====================================================================
def run_sft(args):
    global pause_requested
    signal.signal(signal.SIGINT, handle_sigint)

    rank, local_rank, world_size, is_distributed = setup_distributed()
    is_rank_zero = (rank == 0)

    config = load_config(args.config)
    set_seed(config.get("training", {}).get("seed", 42) + rank)

    device = get_device(local_rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    sft_cfg = config.get("sft", {})

    # 1. Instantiate Model Architecture
    model = AxiomV2(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        multiple_of=model_cfg["multiple_of"],
        norm_eps=model_cfg["norm_eps"],
        rope_theta=model_cfg["rope_theta"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True)
    ).to(device)

    raw_model = model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        raw_model = model.module

    # 2. Transfer Learning: Load Pretrained Weights
    pretrained_path = args.pretrained or "./best.pt"
    if os.path.exists(pretrained_path):
        if is_rank_zero:
            logger.info(f"Loading base foundation weights from {pretrained_path} for SFT...")
        try:
            ckpt_state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt_state = torch.load(pretrained_path, map_location="cpu")

        weights = ckpt_state.get("model", ckpt_state)
        raw_model.load_state_dict(weights, strict=True)
        del ckpt_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if is_rank_zero:
            logger.warning(f"Base weights not found at {pretrained_path}! Initializing SFT with fresh weights.")

    # 3. SFT Optimizer & Dataloaders
    sft_lr = float(args.learning_rate or sft_cfg.get("learning_rate", 1.5e-5))
    optimizer = raw_model.configure_optimizers(
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        learning_rate=sft_lr,
        betas=(0.9, 0.95),
        device_type=device.type
    )

    max_opt_steps = int(args.max_steps or sft_cfg.get("max_steps", 1562))
    scheduler_mgr = SchedulerManager(optimizer, {
        "type": "cosine_warmup",
        "warmup_steps": max(10, int(max_opt_steps * 0.05)),
        "max_steps": max_opt_steps,
        "eta_min_ratio": 0.05,
    })

    sft_data_path = args.data or sft_cfg.get("data_path", "./dataset/sft/sft_data.pt")
    if not os.path.exists(sft_data_path):
        # Check fallback paths
        alt_paths = ["sft_pipeline/dataset/sft/sft_data.pt", "dataset/sft/sft_data.pt"]
        for p in alt_paths:
            if os.path.exists(p):
                sft_data_path = p
                break

    if not os.path.exists(sft_data_path):
        if is_rank_zero:
            logger.error(f"SFT Dataset file not found at {sft_data_path}! Prepare data first.")
        cleanup_distributed()
        sys.exit(1)

    sft_bsz = int(args.batch_size or sft_cfg.get("batch_size", train_cfg.get("batch_size", 1)))
    train_loader, _ = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_bsz,
        is_distributed=is_distributed,
        is_train=True,
        replay_data_path=args.replay_data,
        seq_len=model_cfg["max_seq_len"]
    )
    val_loader, _ = create_sft_dataloader(
        sft_data_path,
        batch_size=sft_bsz,
        is_distributed=is_distributed,
        is_train=False,
        replay_data_path=None
    )

    # 4. Checkpoint & Trainer Setup
    save_dir = args.save_dir or "./checkpoints_sft"
    ckpt_mgr = CheckpointManager(save_dir=save_dir)
    trainer = Trainer(model, optimizer, train_loader, config, is_rank_zero)
    evaluator = Evaluator(model, val_loader, device, is_distributed, config)

    train_logger = TrainingLogger(
        log_dir=os.path.join(save_dir, "logs"),
        use_wandb=args.wandb,
        config=config,
        max_steps=max_opt_steps
    ) if is_rank_zero else None

    # Resume from interrupted SFT run
    start_epoch, start_opt_step, best_val_loss = ckpt_mgr.load(
        raw_model, optimizer, scheduler_mgr, trainer.scaler,
        path=os.path.join(save_dir, "latest.pt")
    )

    grad_accum = int(args.grad_accum_steps or train_cfg.get("grad_accum_steps", 16))
    trainer.grad_accum_steps = grad_accum

    if is_rank_zero:
        total_params = raw_model.get_num_params(non_embedding=False) / 1e6
        logger.info("=" * 60)
        logger.info("🎯 AXIOM V2 SFT ENGINE IGNITED")
        logger.info(f"Model Parameters   : {total_params:.1f} M")
        logger.info(f"Dataset            : {sft_data_path}")
        if args.replay_data:
            logger.info(f"10% Mixed Replay   : {args.replay_data} (Anti-Forgetting ACTIVE)")
        logger.info(f"Base Checkpoint    : {pretrained_path}")
        logger.info(f"Learning Rate      : {sft_lr:.2e} (Cosine Warmup)")
        logger.info(f"Batch Configuration: {sft_bsz} x {grad_accum} accum x {world_size} GPUs = {sft_bsz * grad_accum * world_size} effective")
        logger.info(f"Starting Step      : {start_opt_step} / {max_opt_steps}")
        logger.info("=" * 60)

    train_loader.set_epoch(start_epoch)
    if start_opt_step > 0:
        if is_rank_zero:
            logger.info(f"Fast-forwarding SFT dataloader by {start_opt_step * grad_accum} micro-batches...")
        train_loader.fast_forward(start_opt_step * grad_accum)

    train_iter = iter(train_loader)
    eval_interval = int(sft_cfg.get("eval_interval", 50))
    log_interval = int(sft_cfg.get("log_interval", 1))

    # 5. SFT Loop (Tracked in Optimizer Steps)
    for opt_step in range(start_opt_step, max_opt_steps):
        # Broadcast pause flag
        pause_val = 1 if (is_rank_zero and (pause_requested or os.path.exists("pause.flag"))) else 0
        pause_tensor = torch.tensor([pause_val], device=device)
        if is_distributed:
            dist.broadcast(pause_tensor, src=0)
        if pause_tensor.item() == 1:
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
            if is_rank_zero and micro == 0 and trainer.profiler:
                trainer.profiler.start_step(x.size(0) * grad_accum, x.size(1))

            try:
                if is_distributed and not is_last_micro:
                    with model.no_sync():
                        loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=False)
                else:
                    loss, grad_norm = trainer.train_step(x, y, is_last_accum_step=is_last_micro)
                accum_loss += loss
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"[OOM Warning] GPU out of memory on Rank {rank}. Clearing cache and skipping window.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    oom_flag[0] = 1
                else:
                    raise e

            if is_distributed:
                dist.all_reduce(oom_flag, op=dist.ReduceOp.MAX)
            if oom_flag.item() > 0:
                optimizer.zero_grad(set_to_none=True)
                break

        if oom_flag.item() > 0:
            continue

        scheduler_mgr.step()
        current_step = opt_step + 1

        if is_rank_zero:
            stats = trainer.profiler.end_step() if trainer.profiler else {}
            if trainer.profiler:
                stats.update(trainer.profiler.get_gpu_memory())

            if current_step % log_interval == 0 and train_logger:
                train_logger.log_metrics(
                    step=current_step,
                    train_loss=accum_loss / grad_accum,
                    lr=scheduler_mgr.get_lr(),
                    grad_norm=grad_norm,
                    profiler_stats=stats
                )

        # Synchronous validation across ranks
        if current_step % eval_interval == 0 or current_step == max_opt_steps:
            val_loss, val_ppl = evaluator.evaluate(num_batches=int(sft_cfg.get("eval_iters", 50)))
            if is_rank_zero:
                if train_logger:
                    train_logger.log_metrics(current_step, accum_loss / grad_accum, val_loss=val_loss, perplexity=val_ppl)
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                ckpt_mgr.save(
                    model, optimizer, scheduler_mgr, trainer.scaler,
                    start_epoch, current_step, best_val_loss, config, is_best=is_best
                )

            if is_distributed:
                dist.barrier()

        # Graceful Pause Exit
        if pause_requested:
            if is_rank_zero:
                logger.info(f"Saving paused SFT checkpoint at step {current_step}...")
                ckpt_mgr.save(
                    model, optimizer, scheduler_mgr, trainer.scaler,
                    start_epoch, current_step, best_val_loss, config, is_best=False
                )
                if train_logger:
                    train_logger.close()
                if os.path.exists("pause.flag"):
                    try:
                        os.remove("pause.flag")
                    except OSError:
                        pass
            if is_distributed:
                dist.barrier()
            cleanup_distributed()
            sys.exit(0)

    if is_rank_zero:
        logger.info("Supervised Fine-Tuning completed successfully!")
        if train_logger:
            train_logger.close()
    cleanup_distributed()


# =====================================================================
# SUBCOMMAND: EVAL
# =====================================================================
def run_eval(args):
    config = load_config(args.config)
    device = get_device()
    model_cfg = config["model"]

    logger.info(f"Loading checkpoint for evaluation: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

    weights = checkpoint.get("model", checkpoint)

    model = AxiomV2(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        max_seq_len=model_cfg["max_seq_len"],
        multiple_of=model_cfg["multiple_of"],
        norm_eps=model_cfg["norm_eps"],
        rope_theta=model_cfg["rope_theta"],
        gradient_checkpointing=False
    )
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    num_batches = args.num_batches or 100

    if args.sft_data and os.path.exists(args.sft_data):
        logger.info(f"Evaluating on SFT Dataset: {args.sft_data}")
        val_loader, _ = create_sft_dataloader(
            args.sft_data, batch_size=2, is_distributed=False, is_train=False
        )
    elif args.val_data and os.path.exists(args.val_data):
        logger.info(f"Evaluating on Pretraining Validation Tokens: {args.val_data}")
        val_loader, _ = create_dataloader(
            args.val_data, batch_size=2, seq_len=model_cfg["max_seq_len"],
            is_distributed=False, is_train=False
        )
    else:
        logger.error("No valid --val_data (.bin) or --sft_data (.pt) provided for evaluation.")
        sys.exit(1)

    evaluator = Evaluator(model, val_loader, device, is_distributed=False, config=config)
    t0 = time.time()
    val_loss, perplexity = evaluator.evaluate(num_batches=num_batches)
    dt = time.time() - t0

    logger.info("=" * 60)
    logger.info("📊 AXIOM V2 EVALUATION REPORT")
    logger.info(f"Checkpoint  : {args.checkpoint}")
    logger.info(f"Batches Run : {num_batches}")
    logger.info(f"Duration    : {dt:.2f} seconds")
    logger.info(f"Loss (CE)   : {val_loss:.4f}")
    logger.info(f"Perplexity  : {perplexity:.2f}")
    logger.info("=" * 60)


# =====================================================================
# SUBCOMMAND: BENCHMARK
# =====================================================================
def run_benchmark(args):
    device = get_device()
    logger.info(f"Benchmarking Axiom V2 Engine on {device.type.upper()}")

    # Determine model dimensions
    if args.tiny:
        logger.info("Using tiny benchmark architecture (CPU/low-memory mode)...")
        vocab_size, d_model, n_layers, n_heads, n_kv_heads, max_seq_len = 1000, 256, 4, 8, 2, 256
    else:
        config = load_config(args.config)
        cfg = config["model"]
        vocab_size = cfg["vocab_size"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        n_heads = cfg["n_heads"]
        n_kv_heads = cfg["n_kv_heads"]
        max_seq_len = cfg["max_seq_len"]

    bsz = args.batch_size
    seq_len = min(args.seq_len, max_seq_len)
    dtype_str = args.dtype or ("bfloat16" if device.type != "cpu" else "float32")
    torch_dtype = getattr(torch, dtype_str, torch.float32)

    model = AxiomV2(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        multiple_of=256,
        norm_eps=1e-5,
        rope_theta=500000.0,
        gradient_checkpointing=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    logger.info(f"Model Parameters: {total_params / 1e6:.2f}M (Non-embedding: {non_emb_params / 1e6:.2f}M)")
    logger.info(f"Configuration   : Batch Size={bsz}, Seq Len={seq_len}, Precision={dtype_str}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup steps
    warmup_steps = args.warmup or 5
    for _ in range(warmup_steps):
        x = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
        y = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
        with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=(device.type != "cpu")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed steps
    bench_steps = args.steps or 10
    t0 = time.perf_counter()
    for _ in range(bench_steps):
        x = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
        y = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
        with torch.autocast(device_type=device.type, dtype=torch_dtype, enabled=(device.type != "cpu")):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    time_per_step = elapsed / bench_steps
    tokens_per_step = bsz * seq_len
    tokens_per_sec = (bench_steps * tokens_per_step) / elapsed

    mfu = model.estimate_mfu(fwdbwd_per_iter=bsz, dt=time_per_step)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

    logger.info("=" * 60)
    logger.info("⚡ AXIOM V2 BENCHMARK RESULTS")
    logger.info(f"Avg Step Latency : {time_per_step * 1000.0:.2f} ms")
    logger.info(f"Throughput       : {tokens_per_sec:.1f} tokens/sec")
    logger.info(f"Estimated MFU    : {mfu * 100.0:.2f} %")
    if peak_vram > 0:
        logger.info(f"Peak VRAM        : {peak_vram:.2f} GB")
    logger.info("=" * 60)


# =====================================================================
# MAIN CLI ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Axiom V2 Unified LLM Training & Benchmarking Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Operational mode")

    # Subcommand: pretrain
    p_pretrain = subparsers.add_parser("pretrain", help="Run distributed autoregressive pretraining")
    p_pretrain.add_argument("--config", type=str, default="axiom_model/configs/500M.yaml", help="Path to config YAML")
    p_pretrain.add_argument("--train_data", type=str, default="./data/bin/train.bin", help="Path to train.bin")
    p_pretrain.add_argument("--val_data", type=str, default="./data/bin/val.bin", help="Path to val.bin")
    p_pretrain.add_argument("--output_dir", type=str, default="./checkpoints", help="Save directory")
    p_pretrain.add_argument("--resume", type=str, default=None, help="Resume from specific checkpoint")
    p_pretrain.add_argument("--max_steps", type=int, default=None, help="Override maximum optimizer steps")
    p_pretrain.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")

    # Subcommand: sft
    p_sft = subparsers.add_parser("sft", help="Run Supervised Fine-Tuning (SFT)")
    p_sft.add_argument("--config", type=str, default="axiom_model/configs/500M.yaml", help="Path to config YAML")
    p_sft.add_argument("--pretrained", type=str, default="./best.pt", help="Path to base pretrained model weights")
    p_sft.add_argument("--data", type=str, default=None, help="Path to sft_data.pt")
    p_sft.add_argument("--replay_data", type=str, default=None, help="Path to train.bin for 10% Mixed Replay")
    p_sft.add_argument("--save_dir", type=str, default="./checkpoints_sft", help="Save directory")
    p_sft.add_argument("--learning_rate", type=float, default=None, help="Override SFT learning rate")
    p_sft.add_argument("--max_steps", type=int, default=None, help="Override maximum SFT steps")
    p_sft.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    p_sft.add_argument("--grad_accum_steps", type=int, default=None, help="Gradient accumulation steps")
    p_sft.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")

    # Subcommand: eval
    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint loss and perplexity")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p_eval.add_argument("--config", type=str, default="axiom_model/configs/500M.yaml", help="Path to config YAML")
    p_eval.add_argument("--val_data", type=str, default=None, help="Path to validation token binary")
    p_eval.add_argument("--sft_data", type=str, default=None, help="Path to validation SFT dataset")
    p_eval.add_argument("--num_batches", type=int, default=50, help="Number of batches to evaluate")

    # Subcommand: benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark throughput, latency, and MFU")
    p_bench.add_argument("--config", type=str, default="axiom_model/configs/500M.yaml", help="Path to config YAML")
    p_bench.add_argument("--batch_size", type=int, default=2, help="Batch size")
    p_bench.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    p_bench.add_argument("--steps", type=int, default=10, help="Number of timed benchmark steps")
    p_bench.add_argument("--warmup", type=int, default=3, help="Warmup steps")
    p_bench.add_argument("--dtype", type=str, default="float16", help="Precision dtype (bfloat16, float16, float32)")
    p_bench.add_argument("--tiny", action="store_true", help="Use small architecture for quick CPU test")

    args = parser.parse_args()

    if args.subcommand == "pretrain":
        run_pretrain(args)
    elif args.subcommand == "sft":
        run_sft(args)
    elif args.subcommand == "eval":
        run_eval(args)
    elif args.subcommand == "benchmark":
        run_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

