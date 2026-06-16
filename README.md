<div align="center">

## GD²PO: Group-Dynamic Reward-Decoupled Policy Optimization for Mitigating Multi-Reward RL Conflicts

<!-- Badges -->
<a><img 
     src="https://img.shields.io/badge/Qwen-Applications-4433FF?style=for-the-badge&logo=data:image/png;base64,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" 
     alt="Qwen"></a>
<a href="https://arxiv.org/abs/2606.16771"><img src="https://img.shields.io/badge/arXiv-2606.16771-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
<a href="https://github.com/Qwen-Applications/GD2PO"><img src="https://img.shields.io/badge/Github-Code-black?style=for-the-badge&logo=github" alt="Github"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge" alt="License"></a>

<p align="center">
  <i><b> <img src="https://img.alicdn.com/imgextra/i2/O1CN01FPcQDy1WTPjPX6IH9_!!6000000002789-2-tps-96-96.png" width="16px"  style="vertical-align: middle;"> Qwen Large Model Application Team, Alibaba</b></i>
</p>

</div>

### 📖 Overview

In this project, we provide an implementation of **GD²PO** (*Group-Dynamic Reward-Decoupled Policy Optimization*), a conflict-aware multi-reward policy optimization method for LLM post-training. When multiple reward signals are aggregated, the same rollout may have positive advantages on some dimensions but negative on others, causing signals to cancel out. GD²PO addresses this by filtering conflicting rollouts before aggregation and reweighting each query's update strength based on reward consensus.

<div align="center">
<img src="assets/framework.jpg" alt="Framework Overview" width="600"/>
</div>

We validate GD²PO on two multi-reward post-training tasks:

| Task | Directory | Description |
|------|-----------|-------------|
| **Tool Calling** | [`tool-calling/`](tool-calling/) | Multi-reward optimization with correctness, length, and format rewards |
| **Helpfulness–Safety Alignment** | [`safe-alignment/`](safe-alignment/) | Helpfulness–safety alignment with dual reward models |

---

### 🚀 Getting Started

Each task is self-contained with its own dependencies, data, and training scripts. Please enter the corresponding directory and follow its README:

#### Tool Calling

```bash
cd tool-calling
# See tool-calling/README.md for installation and usage
bash scripts/correctness_length/train_gd2po_hard.sh /path/to/model
```

#### Safe Alignment

```bash
cd safe-alignment
# See safe-alignment/README.md for installation and usage
POLICY_MODEL_PATH=/path/to/model \
RM_MODEL_PATH=/path/to/reward_model \
CM_MODEL_PATH=/path/to/cost_model \
bash scripts/run_gd2po_hard.sh
```

<!-- > 💡 **Important**: Always `cd` into the task directory before running scripts, so that Python resolves the correct `verl` module. -->

---

### 📁 Project Structure

```
├── tool-calling/                # Tool-calling task
│   ├── scripts/                 #   Training & evaluation scripts
│   ├── verl/                    #   Core framework (GD²PO implementation)
│   ├── API_Bank/                #   Evaluation toolkit
│   ├── dataset/                 #   Training and test data
│   └── README.md
├── safe-alignment/              # Safe alignment task
│   ├── scripts/                 #   Training & evaluation scripts
│   ├── verl/                    #   Core framework (GD²PO implementation)
│   ├── dataset/                 #   Training and validation data
│   └── README.md
└── README.md                    # This file
```

---

### 🙏 Acknowledgements

This codebase is built upon [verl](https://github.com/volcengine/verl), [GDPO](https://github.com/NVlabs/GDPO), [ToolRL](https://github.com/qiancheng0/ToolRL), and [Amo](https://github.com/Artessay/Amo). We thank all teams for their excellent open-source contributions.

---

### 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@misc{liu2026gd2pomitigatingmultirewardconflicts,
      title={GD$^2$PO: Mitigating Multi-Reward Conflicts via Group-Dynamic reward-Decoupled Policy Optimization}, 
      author={Haotian Liu and Yihao Liu and Jingwei Ni and Siyuan Huang and Xinpeng Liu and Pengyu Cheng and Jiajun Song and Ruijin Ding and Junfeng Li and Zhechao Yu and Mengyu Zhou and Hongteng Xu and Xiaoxi Jiang and Guanjun Jiang},
      year={2026},
      eprint={2606.16771},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.16771}, 
}
```
