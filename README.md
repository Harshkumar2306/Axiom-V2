<div align="center">

# 🚀 Axiom V2: Custom 476M Foundation Model

**A fully proprietary 476M Parameter Language Model and Distributed Pretraining Engine built from scratch.**

</div>

## 📖 Overview
Axiom V2 is a from-scratch, pre-trained language model project designed to prove that world-class AI engineering can be executed in resource-constrained environments. By combining a proprietary custom neural network architecture with a highly aggressive PyTorch `DistributedDataParallel` engine, this project successfully trains a 476M parameter model on consumer-grade NVIDIA T4 GPUs, achieving physical hardware limits of ~17.5% Model FLOPs Utilization (MFU).

---

## 🧠 Axiom Proprietary Architecture
Axiom V2 is a fully custom-built neural network designed from the ground up to achieve maximum intelligence density within a 500M parameter footprint. It leverages the most advanced mathematical paradigms in modern AI research:

### Core Hyperparameters
* **Parameters:** 476,000,000
* **Vocabulary:** 100,277 (OpenAI `cl100k_base` / tiktoken)
* **Context Window:** 2048 Tokens
* **Layers:** 24
* **Dimension (`d_model`):** 1024

### Deep Architectural Nuances
* **Grouped-Query Attention (GQA):** Uses 16 Query heads and 4 KV heads, achieving a 4:1 memory compression ratio during generation while retaining near Multi-Head Attention (MHA) performance.
* **SwiGLU Activations:** Implements the Swish-Gated Linear Unit with a custom `multiple_of=256` hidden dimension rounding to optimize GPU tile sizes.
* **Rotary Positional Embeddings (RoPE):** Operates with a long-context `rope_theta=500000.0`. The `freqs_cis` matrix is strictly registered as a non-persistent buffer (`persistent=False`), ensuring it safely travels between CPU and GPU without bloating the `state_dict` sizes.
* **RMSNorm:** Replaces standard LayerNorm for speed, utilizing a strict `norm_eps=1.0e-5` with explicit FP32 upcasting to prevent half-precision NaN explosions during deep backpropagation.
* **GPT-Style Scaled Initialization:** Implements a custom residual-branch projection scaling (`0.02 / math.sqrt(2 * n_layers)`) to stabilize the variance of the residual stream as network depth grows.

---

## 🗄️ Dataset (4.5 Billion Tokens)
Axiom V2 is pretrained on a custom-curated 4.5 Billion token dataset designed for maximum knowledge density.

* **Total Tokens:** 4,500,000,030 (4.43B Train | 68M Val)
* **Format:** Raw `uint32` binary memmap (18.00 GB total physical disk size)
* **Tokenizer:** OpenAI `cl100k_base` (100,277 vocab)

### 🧬 Corpus Distribution
To ensure a balanced ratio of reasoning, facts, and grammar, the dataset uses a strict 5-part recipe:
1. **Educational Web (55% | 2.47B Tokens):** Derived from FineWeb-Edu. Heavily filtered for textbooks, tutorials, and deep explainers to build general knowledge.
2. **Code & Technical Logic (27% | 1.21B Tokens):** High-quality Python and C++ repositories. Crucial for procedural reasoning and structured generation.
3. **Factual Foundations (10% | 450M Tokens):** Cleaned, plain-text encyclopedic articles selected for high linguistic consistency.
4. **Advanced Scientific Reasoning (5% | 225M Tokens):** Scientific preprints (arXiv) covering mathematics, physics, and machine learning.
5. **Narratives & Grammar (3% | 135M Tokens):** Short, coherent stories and dialogue to teach fundamental grammar and narrative coherence.

---

## ⚡ The Custom DDP Engine (`train_ddp.py`)
The Pretraining Engine is a custom PyTorch loop engineered to extract the absolute physical speed limit out of constrained hardware environments. 

### Hyper-Optimizations
* **Memory-Mapped Data Loader:** The dataset is read directly from disk via `numpy.memmap`, bypassing strict `/dev/shm` RAM limits found in cloud environments.
* **Asynchronous CPU Streaming:** Uses `non_blocking=True` and `pin_memory=True` to stream massive tensors across the PCIe bus without stalling the Tensor Cores.
* **No-Sync Gradient Accumulation:** The PyTorch `model.no_sync()` context is used to accumulate 32 micro-batches (effective batch size 64) before allowing DDP to talk across the network, slashing NCCL communication overhead by 96%.
* **Memory-Efficient Attention:** Automatically uses `torch.nn.functional.scaled_dot_product_attention` to bypass VRAM bottlenecks.
* **Gradient Checkpointing:** Re-computes forward passes to save ~50% activation VRAM. Implemented securely using `use_reentrant=False` for maximum PyTorch >= 2.0 stability.
* **Mixed Precision & TF32:** Native `torch.autocast` (FP16) combined with `torch.amp.GradScaler` safely navigates 16-bit math. TF32 is explicitly enabled (`allow_tf32=True`) for free hardware speedups on supported devices.
* **CuDNN Auto-Tuning:** Uses `torch.backends.cudnn.benchmark = True` to dynamically profile and select the fastest matrix multiplication algorithms on the fly.
* **Fused AdamW:** Offloads the entire optimizer step into a single fused CUDA kernel (`fused=True`).

### 🛡️ Fault-Tolerant Session Management
Training in preemptible cloud instances (like Kaggle or Colab) requires resilience to strict session timeouts. This engine is built to survive continuous interrupt-and-resume cycles:
* **Graceful Interruptions:** Running `touch pause.flag` triggers a barrier-synchronized DDP exit. Rank 0 detects the file, uses `dist.broadcast` to alert Rank 1, and both GPUs finish their current micro-batch, save an atomic checkpoint, and exit safely without corrupting the gradient tracker.
* **Instant Fast-Forwarding:** Resuming a session doesn't mean re-reading 4 billion tokens. The engine uses a custom `FastForwardSampler` built on `itertools.islice` to instantly slice the dataset and resume on the exact micro-batch you left off on.
* **Deadlock Prevention:** `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are natively supported, with DDP `dist.barrier()` wrappers utilizing a strict 30-minute NCCL timeout to prevent silent ring-fracturing.

---

## 📂 Project Structure
```text
Axiom-V2/
├── axiom_model/
│   ├── configs/          # YAML configs for model size and hyperparams (e.g. 500M.yaml)
│   ├── core/             # Neural network architecture (attention.py, model.py, ffn.py)
│   ├── training/         # DDP Engine components (trainer, dataloader, checkpoint, profiler)
│   └── utils/            # Reproducibility and seed enforcement
├── data_pipeline/        # Scripts to download, clean, and pack the binary memmap dataset
├── sft_pipeline/         # Phase 4 Instruction tuning scripts (ChatML format)
├── train_ddp.py          # The core multi-GPU pretraining engine
└── train_sft.py          # The multi-GPU fine-tuning engine
```

---

## 🚀 Quick Start (Cloud / Kaggle)

1. **Clone & Update**
```bash
git clone https://github.com/Harshkumar2306/Axiom-V2.git
cd Axiom-V2
```

2. **Ignite the Engine**
```bash
!PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 train_ddp.py \
    --config axiom_model/configs/500M.yaml \
    --train_data /path/to/train.bin \
    --val_data /path/to/val.bin
```

3. **Graceful Pause (Before session limits hit)**
Open a terminal in the root directory and run:
```bash
touch pause.flag
```
The engine will safely save `./checkpoints/latest.pt` and shut down.

4. **Resume**
Restart your environment and just run the ignition command again. The engine will auto-detect `latest.pt` and fast-forward instantly.

---

## 🗺️ The Roadmap
- **Phase 1: Build Custom DDP Engine** — Engineered a fully custom PyTorch DistributedDataParallel loop optimized specifically to bypass strict `/dev/shm` RAM limits and maximize T4 GPU compute.
- **Phase 2: Build Dataset (4.5B Tokens)** — Curated, filtered, and tokenized a 4.5 Billion token mix (Web, Code, Science) using `cl100k_base` and packed into binary format for instant Memmap streaming.
- **Phase 3: Pretrain Foundation Model** — (In Progress) Currently executing the 35,000-step pretraining run using asynchronous dataloaders and graceful `pause.flag` resumption logic.
- **Phase 4: Supervised Fine-Tuning (SFT)** — Transitioning the base model into an instruction-following assistant using high-quality ChatML formatted conversational datasets.
- **Phase 5: Direct Preference Optimization (DPO)** — Aligning model outputs with human preferences to reduce hallucinations and enforce logical, helpful responses.
- **Phase 6: Deployment & Ecosystem** — Hooking the trained network up to a localized backend server with Retrieval-Augmented Generation (RAG) capabilities.
