import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import vllm
from tqdm import tqdm
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_paths",
        type=str,
        required=True,
        help="Comma-separated model paths",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="PATH_TO_YOUR_SCORE_ROOT_DETERMINISTIC",
        help="Root directory for deterministic generation outputs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3",
        help="Comma-separated levels to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing result.json if present",
    )
    parser.add_argument(
        "--verify_twice",
        action="store_true",
        help="Run the full generation twice and compare outputs",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length",
    )
    return parser.parse_args()


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def normalize_model_paths(model_paths_str: str):
    return [path.strip() for path in model_paths_str.split(",") if path.strip()]


def normalize_levels(levels_str: str):
    return [level.strip() for level in levels_str.split(",") if level.strip()]


def build_model_name(model_path: str):
    if "verl_example_rlla/" in model_path:
        return model_path.split("verl_example_rlla/")[-1].replace("/", "_")
    return model_path.replace("/", "_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_datasets(levels):
    datasets = {}
    for level in levels:
        data_path = Path(f"./level-{level}-api_processed.json")
        datasets[level] = load_json(data_path)
    return datasets


def build_llm(args, model_path: str):
    return LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
    )


def build_sampling_params(args):
    # Use greedy decoding instead of near-zero-temperature sampling.
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        seed=args.seed,
    )


def extract_text_from_result(result):
    if hasattr(result, "outputs") and len(result.outputs) > 0:
        return result.outputs[0].text.strip()
    if hasattr(result, "choices") and len(result.choices) > 0:
        return result.choices[0].message.content.strip()
    return result[0].outputs[0].text.strip()


def parse_assistant_output(assistant_output: str):
    thought = ""
    tool_calls = []

    if "<think>" in assistant_output and "</think>" in assistant_output:
        thought = assistant_output.split("<think>", 1)[-1].split("</think>", 1)[0].strip()

    if "<tool_call>" in assistant_output and "</tool_call>" in assistant_output:
        tool_block = assistant_output.split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0].strip()
        for line in tool_block.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tool_calls.append(json.loads(line))
            except Exception:
                pass

    return thought, tool_calls


def generate_one_sample(llm, sampling_params, data):
    messages = [
        {"role": "system", "content": data["system"]},
        {"role": "user", "content": data["user"]},
    ]
    result = llm.chat(messages, sampling_params=sampling_params)
    assistant_output = extract_text_from_result(result)
    thought, tool_calls = parse_assistant_output(assistant_output)

    return {
        "data": data,
        "raw_output": assistant_output,
        "thought": thought,
        "tool_calls": tool_calls,
    }


def write_run_meta(save_dir: Path, args, model_path: str):
    run_meta = {
        "model_path": model_path,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "levels": args.levels,
        "WORLD_SIZE_env": os.getenv("WORLD_SIZE"),
        "torch_version": torch.__version__,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "timestamp": int(time.time()),
    }
    dump_json(run_meta, save_dir / "run_meta.json")


def run_generation_for_model(llm, sampling_params, datasets, levels, save_dir: Path, resume: bool):
    result_path = save_dir / "result.json"
    error_path = save_dir / "error.json"

    results = load_json(result_path) if resume and result_path.exists() else {}
    errors = load_json(error_path) if resume and error_path.exists() else {}

    log = {"success": 0, "fail": 0, "exist": 0}

    for level in levels:
        datas = datasets[level]
        for idx, data in tqdm(enumerate(datas), total=len(datas), desc=f"Level {level}"):
            gold = f"Level{level}_{idx}"

            if resume and gold in results:
                log["exist"] += 1
                continue

            try:
                record = generate_one_sample(llm, sampling_params, data)
                results[gold] = record
                log["success"] += 1
            except Exception as e:
                errors[gold] = {
                    "data": data,
                    "error": str(e),
                }
                log["fail"] += 1

            dump_json(results, result_path)
            dump_json(errors, error_path)

    dump_json(results, result_path)
    dump_json(errors, error_path)
    return results, errors, log


def verify_repeatability(run1_results: dict, run2_results: dict, save_dir: Path):
    common_keys = sorted(set(run1_results.keys()) & set(run2_results.keys()))
    mismatches = []

    for key in common_keys:
        output1 = run1_results[key].get("raw_output", "")
        output2 = run2_results[key].get("raw_output", "")
        if output1 != output2:
            mismatches.append(
                {
                    "id": key,
                    "run1_raw_output": output1,
                    "run2_raw_output": output2,
                }
            )

    report = {
        "total_compared": len(common_keys),
        "matched": len(common_keys) - len(mismatches),
        "mismatched": len(mismatches),
        "match_rate": 0.0 if not common_keys else (len(common_keys) - len(mismatches)) / len(common_keys),
        "mismatch_samples": mismatches,
    }
    dump_json(report, save_dir / "verify_report.json")
    return report


def run_and_save(args, model_path: str, datasets, levels, save_dir: Path, resume: bool):
    print("Loading model with vLLM...")
    llm = build_llm(args, model_path)
    print(f"Successfully loaded model from {model_path}")
    print(f"vLLM version: {getattr(vllm, '__version__', 'unknown')}")

    sampling_params = build_sampling_params(args)
    results, errors, log = run_generation_for_model(
        llm=llm,
        sampling_params=sampling_params,
        datasets=datasets,
        levels=levels,
        save_dir=save_dir,
        resume=resume,
    )
    print(log)
    return results, errors, log


def main():
    args = parse_args()
    set_global_seed(args.seed)

    model_paths = normalize_model_paths(args.model_paths)
    levels = normalize_levels(args.levels)
    datasets = load_datasets(levels)

    for model_path in model_paths:
        model_name = build_model_name(model_path)
        model_save_dir = ensure_dir(Path(args.output_root) / model_name / f"seed_{args.seed}")
        write_run_meta(model_save_dir, args, model_path)

        if args.verify_twice:
            run1_dir = ensure_dir(model_save_dir / "run1")
            run2_dir = ensure_dir(model_save_dir / "run2")

            run1_results, _, _ = run_and_save(
                args=args,
                model_path=model_path,
                datasets=datasets,
                levels=levels,
                save_dir=run1_dir,
                resume=args.resume,
            )

            set_global_seed(args.seed)
            run2_results, _, _ = run_and_save(
                args=args,
                model_path=model_path,
                datasets=datasets,
                levels=levels,
                save_dir=run2_dir,
                resume=False,
            )

            report = verify_repeatability(run1_results, run2_results, model_save_dir)
            print(report)
        else:
            run_and_save(
                args=args,
                model_path=model_path,
                datasets=datasets,
                levels=levels,
                save_dir=model_save_dir,
                resume=args.resume,
            )

        print("All done for", model_path)


if __name__ == "__main__":
    main()
