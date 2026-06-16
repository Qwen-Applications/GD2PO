#!/usr/bin/env bash
# Evaluate a trained checkpoint on the validation sets (val-only mode).
#
# Usage:
#   POLICY_MODEL_PATH=/path/to/checkpoint \
#   RM_MODEL_PATH=/path/to/reward_model \
#   CM_MODEL_PATH=/path/to/cost_model \
#   bash scripts/eval.sh
#
# Optional:
#   OUTPUT_DIR (default: ./outputs/eval)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_safe_alignment.sh"

EXP_NAME="${EXP_NAME:-eval}"
EVAL_OUTPUT="${OUTPUT_DIR}/eval/${EXP_NAME}"
TENSORBOARD_DIR="tensorboard_log/safe_alignment/${EXP_NAME}"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gdpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.load_contents="['model','extra']" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model_useful.enable=True \
    reward_model_useful.strategy=fsdp \
    reward_model_useful.model.path="${RM_MODEL_PATH}" \
    reward_model_useful.model.input_tokenizer="${POLICY_MODEL_PATH}" \
    reward_model_useful.micro_batch_size_per_gpu=1 \
    reward_model_useful.max_length=2048 \
    +reward_model_useful.model.use_align_anything_reward_model=True \
    +reward_model_useful.model.use_align_anything_formatter=True \
    reward_model_useful.model.fsdp_config.param_offload=True \
    reward_model_harmless.enable=True \
    reward_model_harmless.strategy=fsdp \
    reward_model_harmless.model.path="${CM_MODEL_PATH}" \
    reward_model_harmless.model.input_tokenizer="${POLICY_MODEL_PATH}" \
    reward_model_harmless.micro_batch_size_per_gpu=1 \
    reward_model_harmless.max_length=2048 \
    +reward_model_harmless.model.use_align_anything_reward_model=True \
    +reward_model_harmless.model.use_align_anything_formatter=True \
    reward_model_harmless.model.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name=safe_alignment \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${EVAL_OUTPUT}/checkpoints"

# Print validation mean@1 metrics from the TensorBoard logs
echo ""
echo "=== Validation mean@1 ==="
python3 "${SCRIPT_DIR}/eval_metric.py" --events "${TENSORBOARD_DIR}"
