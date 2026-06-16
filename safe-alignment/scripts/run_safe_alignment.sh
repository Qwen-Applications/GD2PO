#!/usr/bin/env bash
# Common configuration for safe alignment training.
# Sourced by method-specific scripts (run_hard.sh, run_snr.sh).
#
# Required env vars:
#   POLICY_MODEL_PATH  - path to the policy model
#   RM_MODEL_PATH      - path to the useful reward model
#   CM_MODEL_PATH      - path to the harmless (cost/safety) model
#
# Optional env vars:
#   TRAIN_DATA / VAL_DATA  - data paths (defaults provided)
#   N_GPUS / OUTPUT_DIR / EXP_NAME / ROLLOUT_TP_SIZE

# --- Required paths ---
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:?Please set POLICY_MODEL_PATH}"
RM_MODEL_PATH="${RM_MODEL_PATH:?Please set RM_MODEL_PATH}"
CM_MODEL_PATH="${CM_MODEL_PATH:?Please set CM_MODEL_PATH}"

# --- Optional settings ---
# Default DATA_ROOT points to the dataset/ folder in the project root
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJ_ROOT}/dataset}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_ROOT}/alpaca_prompt_only/train.parquet}"
VAL_DATA="${VAL_DATA:-['${DATA_ROOT}/alpaca_prompt_only/test.parquet','${DATA_ROOT}/anthropic_hh_rlhf/test.parquet','${DATA_ROOT}/pku-saferlhf/test.parquet']}"
N_GPUS="${N_GPUS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
EXP_NAME="${EXP_NAME:-safe_alignment}"
