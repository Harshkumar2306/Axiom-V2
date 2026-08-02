<div align="center">

# 🚀 Axiom V2: Custom 476M Foundation Model

**A hyper-optimized 476M Parameter Language Model and Distributed Pretraining Engine built from scratch.**

</div>

## 📖 Overview
Axiom V2 is a from-scratch, pre-trained language model project designed to prove that world-class AI engineering can be executed in resource-constrained environments. By combining the architectural blueprint of Meta's Llama-3 with a custom, highly aggressive PyTorch `DistributedDataParallel` engine, this project successfully trains a 476M parameter model on free Kaggle T4 GPUs, achieving physical hardware limits of ~17.5% Model FLOPs Utilization (MFU).

---

## 🧠 Architecture (Llama-3 Blueprint)
Axiom V2 mirrors the exact architectural choices of Llama-3 to achieve maximum intelligence density in a 500M parameter footprint.

* **Parameters:** 476,000,000
* **Vocabulary:** 100,277 (OpenAI `cl100k_base` / tiktoken)
* **Context Window:** 2048 Tokens
* **Layers:** 24
* **Attention:** Grouped-Query Attention (GQA) with 16 Q-heads and 4 KV-heads (4:1 compression ratio)
* **Activations:** SwiGLU (Swish-Gated Linear Unit)
* **Normalization:** RMSNorm (with explicit FP32 upcasting to prevent NaN explosions)
* **Embeddings:** Rotary Positional Embeddings (RoPE)

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

## ⚡ The Custom DDP Engine
The Axiom V2 Pretraining Engine (`train_ddp.py`) is a custom PyTorch loop engineered to extract the absolute physical speed limit out of free Kaggle hardware. 

### Hyper-Optimizations
* **Memory-Mapped Data Loader:** The dataset is read directly from disk via `numpy.memmap`, bypassing Kaggle's strict `/dev/shm` RAM limits.
* **Asynchronous CPU Streaming:** Uses `non_blocking=True` and `pin_memory=True` to stream massive tensors across the PCIe bus without stalling the Tensor Cores.
* **No-Sync Gradient Accumulation:** The PyTorch `no_sync()` context is used to accumulate 32 micro-batches before allowing DDP to talk across the network, slashing NCCL communication overhead by 96%.
* **Memory-Efficient Attention:** Automatically uses xFormers / PyTorch `scaled_dot_product_attention` to bypass VRAM bottlenecks.
* **Mixed Precision:** Native `torch.autocast` (FP16) combined with `torch.amp.GradScaler` safely navigates 16-bit math.
* **CuDNN Auto-Tuning:** Uses `torch.backends.cudnn.benchmark = True` to dynamically profile and select the fastest matrix multiplication algorithms on the fly.
* **Fused AdamW:** Offloads the optimizer step into a single fused CUDA kernel.

### 🛡️ Kaggle Survival Features
Training on Kaggle means dealing with strict 12-hour session limits. This engine is built to survive continuous interrupt-and-resume cycles:
* **Graceful Interruptions:** Running `!touch pause.flag` triggers a barrier-synchronized DDP exit. Both GPUs finish their current micro-batch, save an atomic checkpoint, and exit safely without corrupting the gradient tracker.
* **Instant Fast-Forwarding:** Resuming a session doesn't mean re-reading 4 billion tokens. The engine uses a custom `FastForwardSampler` to instantly slice the dataset and resume on the exact micro-batch you left off on.
* **Deadlock Prevention:** `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are natively supported, with DDP barriers wrapped in local GPU ranks to prevent silent ring-fracturing.

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

## 🚀 Quick Start (Kaggle)

1. **Clone & Update**
```bash
%cd /kaggle/working
git clone https://github.com/Harshkumar2306/Axiom-V2.git
%cd Axiom-V2
```

2. **Ignite the Engine**
```bash
!PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 train_ddp.py \
    --config axiom_model/configs/500M.yaml \
    --train_data /kaggle/input/datasets/hrsh0o23/axiom-v2-4-5b-dataset/train.bin \
    --val_data /kaggle/input/datasets/hrsh0o23/axiom-v2-4-5b-dataset/val.bin
```

3. **Graceful Pause (Before your 12-hour limit hits)**
Open a new Kaggle terminal and run:
```bash
touch /kaggle/working/Axiom-V2/pause.flag
```
The engine will safely save `./checkpoints/latest.pt` and shut down.

4. **Resume**
Restart your Kaggle kernel and just run the ignition command again. The engine will auto-detect `latest.pt` and fast-forward instantly.

---

## 🗺️ The Roadmap
- [x] **Phase 1:** Build Engine
- [x] **Phase 2:** Build Dataset (4.5B Tokens)
- [x] **Phase 3:** Pretrain Foundation Model 
- [ ] **Phase 4:** Supervised Fine-Tuning (SFT)
- [ ] **Phase 5:** Direct Preference Optimization (DPO)
- [ ] **Phase 6:** Evaluation & RAG Deployment
