"""
Aggregate per-model API-Bank scores into ONE leaderboard.json.

Reads, for every <model_name> sub-directory found anywhere under --score-root:
  - score_reward.json  (from evaluate_reward.py -> correctness + format + length)

Writes a single leaderboard.json sorted by overall_acc desc.

Output schema (per model):
{
  "<encoded_model_name>": {
    "correct_lv1": int,  "correct_lv2": int,  "correct_lv3": int,
    "total_lv1":   int,  "total_lv2":   int,  "total_lv3":   int,
    "lv1_acc":     float, "lv2_acc":    float, "lv3_acc":    float, "overall_acc": float,

    "format_lv1":      int,  "format_lv2":     int,  "format_lv3":     int,
    "format_lv1_acc":  float, "format_lv2_acc": float, "format_lv3_acc": float, "overall_format_acc": float,

    "length_avg_lv1":  float, "length_avg_lv2": float, "length_avg_lv3": float, "overall_length_avg": float,
    "think_word_count_avg_lv1": float, "think_word_count_avg_lv2": float, "think_word_count_avg_lv3": float,
    "overall_think_word_count_avg": float,

    "reward_avg_lv1":  float, "reward_avg_lv2": float, "reward_avg_lv3": float, "overall_reward_avg": float
  },
  ...
}
"""
import argparse
import json
import os
from typing import Optional


LEVELS = ("lv1", "lv2", "lv3")

SCORE_FILE_NAMES = ("score_reward.json",)


def get_level_key(sample_key: str) -> Optional[str]:
    if sample_key.startswith("Level1"):
        return "lv1"
    if sample_key.startswith("Level2"):
        return "lv2"
    if sample_key.startswith("Level3"):
        return "lv3"
    return None


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_or_none(value, ndigits: int):
    if value is None:
        return None
    return round(value, ndigits)


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def find_model_dirs(score_root: str):
    """Yield every directory under score_root that contains at least one of the
    expected score*.json files."""
    for root, _dirs, files in os.walk(score_root):
        if any(name in files for name in SCORE_FILE_NAMES):
            yield root


def aggregate_one_model(model_dir: str) -> Optional[dict]:
    """Read up to three score*.json files in model_dir and return a flat record
    that matches the schema described in the module docstring. Returns None when
    none of the expected files exist."""
    reward_path = os.path.join(model_dir, "score_reward.json")

    if not os.path.exists(reward_path):
        return None

    # Counters / accumulators per level.
    correct = {lv: 0 for lv in LEVELS}
    total_correct = {lv: 0 for lv in LEVELS}

    format_pass = {lv: 0 for lv in LEVELS}
    total_format = {lv: 0 for lv in LEVELS}

    length_sum = {lv: 0.0 for lv in LEVELS}
    length_count = {lv: 0 for lv in LEVELS}

    think_sum = {lv: 0.0 for lv in LEVELS}
    think_count = {lv: 0 for lv in LEVELS}

    reward_sum = {lv: 0.0 for lv in LEVELS}
    reward_count = {lv: 0 for lv in LEVELS}

    # ---- unified score file (score_reward.json) ----
    # evaluate_reward.py writes correctness / format / length / think_word_count
    # into a single per-sample json.
    rwd_scores = load_json(reward_path)
    for key, item in rwd_scores.items():
        lv = get_level_key(key)
        if lv is None:
            continue

        total_correct[lv] += 1
        if item.get("score") == 1:
            correct[lv] += 1

        total_format[lv] += 1
        if item.get("format_score") == 1:
            format_pass[lv] += 1

        length_score = item.get("length_score")
        if isinstance(length_score, (int, float)):
            length_value = float(length_score)
            length_sum[lv] += length_value
            length_count[lv] += 1
            reward_sum[lv] += length_value
            reward_count[lv] += 1

        think_word_count = item.get("think_word_count")
        if isinstance(think_word_count, (int, float)):
            think_sum[lv] += float(think_word_count)
            think_count[lv] += 1

    record: dict = {}

    # --- correctness ---
    for lv in LEVELS:
        record[f"correct_{lv}"] = correct[lv]
    for lv in LEVELS:
        record[f"total_{lv}"] = total_correct[lv]
    for lv in LEVELS:
        acc = safe_div(correct[lv], total_correct[lv])
        record[f"{lv}_acc"] = round_or_none((acc * 100) if acc is not None else None, 2)

    overall_correct_num = sum(correct.values())
    overall_correct_den = sum(total_correct.values())
    overall_acc = safe_div(overall_correct_num, overall_correct_den)
    record["overall_acc"] = round_or_none((overall_acc * 100) if overall_acc is not None else None, 2)

    # --- format ---
    for lv in LEVELS:
        record[f"format_{lv}"] = format_pass[lv]
    for lv in LEVELS:
        acc = safe_div(format_pass[lv], total_format[lv])
        record[f"format_{lv}_acc"] = round_or_none((acc * 100) if acc is not None else None, 2)

    overall_format_num = sum(format_pass.values())
    overall_format_den = sum(total_format.values())
    overall_format = safe_div(overall_format_num, overall_format_den)
    record["overall_format_acc"] = round_or_none((overall_format * 100) if overall_format is not None else None, 2)

    # --- length / think word count ---
    for lv in LEVELS:
        avg = safe_div(length_sum[lv], length_count[lv])
        record[f"length_avg_{lv}"] = round_or_none(avg, 4)
    overall_len_avg = safe_div(sum(length_sum.values()), sum(length_count.values()))
    record["overall_length_avg"] = round_or_none(overall_len_avg, 4)

    for lv in LEVELS:
        avg = safe_div(think_sum[lv], think_count[lv])
        record[f"think_word_count_avg_{lv}"] = round_or_none(avg, 4)
    overall_think_avg = safe_div(sum(think_sum.values()), sum(think_count.values()))
    record["overall_think_word_count_avg"] = round_or_none(overall_think_avg, 4)

    # --- reward ---
    for lv in LEVELS:
        avg = safe_div(reward_sum[lv], reward_count[lv])
        record[f"reward_avg_{lv}"] = round_or_none(avg, 4)
    overall_reward_avg = safe_div(sum(reward_sum.values()), sum(reward_count.values()))
    record["overall_reward_avg"] = round_or_none(overall_reward_avg, 4)

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-root",
        required=True,
        help="Root directory to scan recursively for per-model score*.json files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the aggregated leaderboard.json.",
    )
    parser.add_argument(
        "--key-mode",
        choices=("dir", "basename"),
        default="dir",
        help="How to derive each entry's key. 'dir' uses the encoded directory "
             "name (matches evaluate*.py output, e.g. root_xxx_actor_merge_model). "
             "'basename' uses just the leaf basename.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    score_root = os.path.abspath(args.score_root)
    if not os.path.isdir(score_root):
        print(f"[aggregate] ERROR: score-root does not exist: {score_root}")
        return 1

    leaderboard: dict = {}
    skipped: list = []

    for model_dir in find_model_dirs(score_root):
        # Skip the score_root itself if its files were dumped at the top level.
        if os.path.samefile(model_dir, score_root):
            continue

        record = aggregate_one_model(model_dir)
        if record is None:
            continue

        if args.key_mode == "basename":
            key = os.path.basename(model_dir.rstrip("/"))
        else:
            key = os.path.basename(model_dir.rstrip("/"))

        # Avoid clobbering: if the same basename appears twice, fall back to the
        # rel path under score_root for the second one.
        if key in leaderboard:
            rel = os.path.relpath(model_dir, score_root).replace("/", "_")
            skipped.append((key, rel))
            key = rel

        leaderboard[key] = record

    # Sort by overall_acc desc; entries without an overall_acc go to the bottom.
    def sort_key(item):
        v = item[1].get("overall_acc")
        return (-1 if v is None else v)

    leaderboard = dict(sorted(leaderboard.items(), key=sort_key, reverse=True))

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=4, ensure_ascii=False)

    print(f"[aggregate] wrote {len(leaderboard)} entries -> {output_path}")
    if skipped:
        print(f"[aggregate] disambiguated {len(skipped)} duplicate basename(s): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
