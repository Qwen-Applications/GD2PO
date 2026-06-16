#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

BASE_MODEL="${1:?Usage: $0 <model_path> [model_family]}"
MODEL_FAMILY="${2:-}"
EXP_NAME="gd2po-hard-correctness_length"

# Infer model family (qwen/llama) from model path if not explicitly provided
if [[ -z "${MODEL_FAMILY}" ]]; then
    BASE_MODEL_LOWER="$(echo "${BASE_MODEL}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${BASE_MODEL_LOWER}" == *"qwen"* ]]; then
        MODEL_FAMILY="qwen"
    elif [[ "${BASE_MODEL_LOWER}" == *"llama"* ]]; then
        MODEL_FAMILY="llama"
    else
        echo "ERROR: Cannot infer model family from '${BASE_MODEL}'." >&2
        echo "Please specify model_family as the second argument: $0 <model_path> <qwen|llama>" >&2
        exit 1
    fi
fi

export EXPERIMENT_NAME="${MODEL_FAMILY}"
export WITHCORRECT=1 WITHFORMAT=0 WITHLENGTH=1
export RAY_USAGE_STATS_ENABLED=0 RAY_DISABLE_DOCKER_CPU_WARNING=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

DATA_DIR="${REPO_ROOT}/dataset/rlla_4k"
OUTPUT_DIR="${REPO_ROOT}/open_source_runs"

echo "── ${EXP_NAME} | model=${BASE_MODEL} ──"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo_filter_group_ratio \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=512 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE:-1}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=ConflictAwareRolloutFiltering \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS:-8}" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints/${EXP_NAME}" \
    trainer.rollout_data_dir="${OUTPUT_DIR}/rollout_data/${EXP_NAME}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/valid_data/${EXP_NAME}"
