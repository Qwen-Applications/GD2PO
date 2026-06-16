#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path> <output_root> [run_tag] [seed]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
API_BANK_DIR="${API_BANK_DIR:-${REPO_ROOT}/API_Bank}"

model_path="$1"
output_root="$2"
run_tag="${3:-$(basename "${model_path}")}"
seed="${4:-42}"

if [[ ! -d "${API_BANK_DIR}" ]]; then
    echo "API_BANK_DIR=${API_BANK_DIR} does not exist." >&2
    exit 1
fi

export WORLD_SIZE="${WORLD_SIZE:-4}"
tensor_parallel_size="${TENSOR_PARALLEL_SIZE:-${WORLD_SIZE}}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}"
levels="${LEVELS:-1,2,3}"
max_tokens="${MAX_TOKENS:-4096}"
max_model_len="${MAX_MODEL_LEN:-4096}"

run_dir="${output_root}/${run_tag}"
gen_root="${run_dir}/generate"
eval_root="${run_dir}/eval"
mkdir -p "${gen_root}" "${eval_root}"

(
    cd "${API_BANK_DIR}"
    python generate_deterministic.py \
        --model_paths "${model_path}" \
        --output_root "${gen_root}" \
        --seed "${seed}" \
        --tensor_parallel_size "${tensor_parallel_size}" \
        --gpu_memory_utilization "${gpu_memory_utilization}" \
        --levels "${levels}" \
        --max_tokens "${max_tokens}" \
        --max_model_len "${max_model_len}"
)

gen_model_name="${model_path//\//_}"
if [[ "${model_path}" == *"verl_example_rlla/"* ]]; then
    gen_model_name="${model_path##*verl_example_rlla/}"
    gen_model_name="${gen_model_name//\//_}"
fi

src_result="${gen_root}/${gen_model_name}/seed_${seed}/result.json"
if [[ ! -f "${src_result}" ]]; then
    echo "Missing deterministic result: ${src_result}" >&2
    exit 1
fi

final_eval_dir="${eval_root}/${run_tag}"
mkdir -p "${final_eval_dir}"
cp -f "${src_result}" "${final_eval_dir}/result.json"

eval_tmpdir="$(mktemp -d /tmp/conflict_aware_api_bank_eval_XXXXXX)"
trap 'rm -rf "${eval_tmpdir}"' EXIT

eval_tmp_model_dir="${eval_tmpdir}/model"
mkdir -p "${eval_tmp_model_dir}"
cp -f "${src_result}" "${eval_tmp_model_dir}/result.json"

abs_eval_tmp_model_dir="$(cd "${eval_tmp_model_dir}" && pwd)"
encoded_model_name="${abs_eval_tmp_model_dir#/}"
encoded_model_name="${encoded_model_name//\//_}"
encoded_score_dir="${eval_tmpdir}/${encoded_model_name}"

mkdir -p "${encoded_score_dir}"
cp -f "${src_result}" "${encoded_score_dir}/result.json"
rm -f "${encoded_score_dir}/score_reward.json"

export API_BANK_SCORE_ROOT="${eval_tmpdir}"

(
    cd "${API_BANK_DIR}"
    python evaluate_reward.py --model_paths "${abs_eval_tmp_model_dir}"
)

for fname in result.json score_reward.json; do
    if [[ -f "${encoded_score_dir}/${fname}" ]]; then
        cp -f "${encoded_score_dir}/${fname}" "${final_eval_dir}/${fname}"
    fi
done

python "${API_BANK_DIR}/aggregate_leaderboard.py" \
    --score-root "${eval_root}" \
    --output "${eval_root}/leaderboard.json"

echo "Evaluation finished: ${final_eval_dir}"
