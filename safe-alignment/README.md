<div align="center">

## GD²PO — Helpfulness–Safety Alignment

</div>

## 📖 Overview

This repository provides training scripts for safe alignment of LLMs using **GD²PO** (Group-Dynamic Reward-Decoupled Policy Optimization). Given a policy model and two reward models (helpfulness + safety), GD²PO detects and filters rollouts with cross-reward advantage conflicts before policy updates.

---

## 🚀 Getting Started

### Installation

Requires CUDA-compatible GPUs (8× recommended for 7B models).

```bash
# 1. Create conda environment
conda create -n gd2po_safe python=3.12.11 -y
conda activate gd2po_safe

# 2. Install PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install project dependencies
pip install -r requirements_safe.txt

# 4. Install FlashAttention
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### Model Preparation

Download reward models from ModelScope:

```bash
pip install modelscope

# Useful reward model
modelscope download --model Artessay/Qwen2.5-7B-SafeRLHF-RM --local_dir /path/to/Qwen2.5-7B-SafeRLHF-RM

# Harmless reward model (cost model)
modelscope download --model Artessay/Qwen2.5-7B-SafeRLHF-CM --local_dir /path/to/Qwen2.5-7B-SafeRLHF-CM
```

> Links: [Qwen2.5-7B-SafeRLHF-RM](https://modelscope.cn/models/Artessay/Qwen2.5-7B-SafeRLHF-RM) · [Qwen2.5-7B-SafeRLHF-CM](https://modelscope.cn/models/Artessay/Qwen2.5-7B-SafeRLHF-CM)


## 🏋️ Training

Only three paths are required — everything else has sensible defaults:

```bash
POLICY_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
RM_MODEL_PATH=/path/to/Qwen2.5-7B-SafeRLHF-RM \
CM_MODEL_PATH=/path/to/Qwen2.5-7B-SafeRLHF-CM \
bash scripts/run_gd2po_hard.sh
```

### All methods

```bash
# GDPO baseline (no conflict filtering)
bash scripts/run_gdpo.sh

# GD²PO-Hard (sign-based filtering)
bash scripts/run_gd2po_hard.sh

# GD²PO-SNR (SNR-like filtering, default τ=0.8)
bash scripts/run_gd2po_snr.sh

# GD²PO-SNR with custom threshold
SNR_THRESHOLD=0.5 bash scripts/run_gd2po_snr.sh
```



## 📊 Evaluation

Evaluate a trained checkpoint on the validation sets and print reward metrics:

```bash
POLICY_MODEL_PATH=/path/to/trained_checkpoint \
RM_MODEL_PATH=/path/to/Qwen2.5-7B-SafeRLHF-RM \
CM_MODEL_PATH=/path/to/Qwen2.5-7B-SafeRLHF-CM \
bash scripts/eval.sh
```

This runs print `mean@1` scores for each dataset (Alpaca, HH-RLHF, PKU-SafeRLHF) across useful / harmless / combined dimensions.


## 📁 Project Structure

```
├── scripts/
│   ├── run_safe_alignment.sh   # Shared config (auto-sourced)
│   ├── run_gdpo.sh             # GDPO baseline
│   ├── run_gd2po_hard.sh        # GD²PO-Hard
│   └── run_gd2po_snr.sh        # GD²PO-SNR
├── verl/
│   ├── trainer/ppo/
│   │   ├── ray_trainer.py      # Core GD²PO algorithm implementation
│   │   ├── core_algos.py       # Advantage estimators
│   │   └── metric_utils.py     # Validation metrics
│   └── workers/                # Distributed model workers
└── dataset/                    # Training & validation data (parquet)
```

---

## 🙏 Acknowledgements

This codebase is built upon [verl](https://github.com/volcengine/verl) and inspired by [GDPO](https://github.com/NVlabs/GDPO). Reward models are provided by [Amo](https://github.com/Artessay/Amo). We thank all the authors for their excellent open-source contributions.
