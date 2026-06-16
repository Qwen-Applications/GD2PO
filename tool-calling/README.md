<div align="center">

## GD²PO — Tool Calling

</div>

### 📖 Overview

This repository provides training and evaluation scripts for tool-calling post-training of LLMs using **GD²PO** (Group-Dynamic Reward-Decoupled Policy Optimization). Given a policy model and multiple reward signals (correctness, length, and optionally format), GD²PO detects and filters rollouts with cross-reward advantage conflicts before policy updates.

## 🚀 Getting Started

### 🔧 Installation

```bash
# 1. Create conda environment
conda create -n gd2po_tool python=3.10.12 -y
conda activate gd2po_tool

# 2. Install dependencies
pip install -r requirements_tool.txt

# 3. Install FlashAttention
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## 🏋️ Training

### Dataset

The tool-calling training data is based on the processed RLLA-4K data released by ToolRL:

https://github.com/qiancheng0/ToolRL/tree/main/dataset/rlla_4k

Download the parquet files before training and place them under `dataset/rlla_4K/`:

```bash
mkdir -p dataset/rlla_4K
wget -O dataset/rlla_4K/train.parquet https://raw.githubusercontent.com/qiancheng0/ToolRL/main/dataset/rlla_4k/train.parquet
wget -O dataset/rlla_4K/test.parquet https://raw.githubusercontent.com/qiancheng0/ToolRL/main/dataset/rlla_4k/test.parquet
```

Training scripts are organized by reward setting. The model family (`qwen` or `llama`) is auto-inferred from the model path, or can be specified explicitly.

### Two-reward setting (correctness + length)

```bash
# GD²PO-Hard
bash scripts/correctness_length/train_hard.sh /path/to/model

# GD²PO-SNR (default tau=0.8)
bash scripts/correctness_length/train_snr.sh /path/to/model

# GD²PO-SNR with custom tau
bash scripts/correctness_length/train_snr.sh /path/to/model 0.5

# GDPO baseline (no conflict filtering)
bash scripts/correctness_length/train_gdpo.sh /path/to/model
```

### Three-reward setting (correctness + length + format)

```bash
# GD²PO-Hard
bash scripts/correctness_length_format/train_hard.sh /path/to/model

# GD²PO-SNR (default tau=0.8)
bash scripts/correctness_length_format/train_snr.sh /path/to/model
```

> 💡 **Tip**: To explicitly specify the model family (e.g., for non-standard model paths), pass it as an extra argument:
> ```bash
> bash scripts/correctness_length/train_hard.sh /path/to/model qwen
> bash scripts/correctness_length/train_snr.sh /path/to/model 0.8 qwen
> ```

## 📊 Evaluation

Evaluate a trained checkpoint on the API-Bank benchmark:

```bash
bash scripts/eval_api_bank.sh /path/to/checkpoint /path/to/output
```

## 📁 Project Structure

```
├── scripts/                          # Training & evaluation scripts
│   ├── correctness_length/           #   Two-reward setting (correctness + length)
│   ├── correctness_length_format/    #   Three-reward setting (correctness + length + format)
│   └── eval_api_bank.sh              #   API-Bank evaluation pipeline
├── verl/                             # Core framework
│   ├── trainer/ppo/                  #   GD²PO core algorithm
│   ├── utils/                        #   Reward functions & utilities
│   ├── workers/                      #   Rollout & reward workers
│   └── ...
├── API_Bank/                         # Evaluation toolkit
├── dataset/                          # Training and test data
└── requirements_tool.txt             # Python dependencies
```

## 🙏 Acknowledgements

This codebase is built upon [verl](https://github.com/volcengine/verl), [ToolRL](https://github.com/qiancheng0/ToolRL), and [GDPO](https://github.com/NVlabs/GDPO). We thank all teams for their excellent open-source contributions.
