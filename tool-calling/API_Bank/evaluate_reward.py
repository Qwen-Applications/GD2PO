import json
import os


def build_model_name(model_path: str) -> str:
    return os.path.abspath(model_path).strip("/").replace("/", "_")


def normalize_answer(answer):
    if isinstance(answer, list) and answer:
        return answer[0]
    return answer


def compute_correctness_score(tool_calls, answer):
    answer = normalize_answer(answer)
    if not isinstance(answer, dict):
        return None

    answer_name = answer.get("name")
    answer_parameters = answer.get("parameters")
    if not isinstance(answer_name, str) or not isinstance(answer_parameters, dict):
        return None

    try:
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                tool_call = json.loads(tool_call)
            predict = tool_call

            if "name" not in predict or "parameters" not in predict:
                name = answer_name
                parameters = predict
            else:
                name = predict["name"]
                parameters = predict["parameters"]

            if name == answer_name and parameters == answer_parameters:
                return 1
    except Exception as e:
        print("Error parsing tool calls:", e)

    return 0


def validate_output_format(raw_output: str):
    if not isinstance(raw_output, str) or not raw_output.strip():
        return 0, ["empty_output"]

    text = raw_output.strip()
    errors = []

    for tag_name in ("think", "tool_call", "response"):
        open_count = text.count(f"<{tag_name}>")
        close_count = text.count(f"</{tag_name}>")
        if open_count != close_count:
            errors.append(f"unbalanced_{tag_name}_tags")
        if open_count > 1:
            errors.append(f"multiple_{tag_name}_blocks")

    if errors:
        return 0, sorted(set(errors))

    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start < 0 or think_end < 0:
        return 0, ["missing_think"]
    if think_end <= think_start:
        return 0, ["invalid_think_order"]

    think_close_pos = think_end + len("</think>")
    tool_start = text.find("<tool_call>")
    tool_end = text.find("</tool_call>")
    response_start = text.find("<response>")
    response_end = text.find("</response>")

    has_tool_call = tool_start >= 0
    has_response = response_start >= 0

    if not has_tool_call and not has_response:
        return 0, ["missing_tool_call_and_response"]

    if has_tool_call:
        if tool_end <= tool_start:
            errors.append("invalid_tool_call_order")
        if tool_start < think_close_pos:
            errors.append("tool_call_before_think_end")

    if has_response:
        if response_end <= response_start:
            errors.append("invalid_response_order")
        if response_start < think_close_pos:
            errors.append("response_before_think_end")

    if has_tool_call and has_response and response_start < tool_end:
        errors.append("response_before_tool_call_end")

    if has_response and has_tool_call and response_end < tool_start:
        errors.append("tool_call_after_response")

    return int(len(errors) == 0), sorted(set(errors))


def extract_think_content(raw_output: str):
    if not isinstance(raw_output, str):
        return None

    think_start = raw_output.find("<think>")
    think_end = raw_output.find("</think>")
    if think_start < 0 or think_end < 0 or think_end <= think_start:
        return None

    think_start += len("<think>")
    return raw_output[think_start:think_end].strip()


def compute_length_score(raw_output: str):
    max_possible_reward = float(os.getenv("LENGTH_MAX_POSSIBLE_REWARD", "1.0"))
    min_possible_reward = float(os.getenv("LENGTH_MIN_POSSIBLE_REWARD", "0.0"))
    step = float(os.getenv("LENGTH_SCORE_STEP", "0"))

    if os.getenv("SCHEDULELENGTH", "0") == "1":
        max_reward_len = (640 - 384) * step / 105 + 384
    else:
        max_reward_len = 512

    think_content = extract_think_content(raw_output)
    if think_content is None:
        return min_possible_reward, 0

    reward = round(len(think_content.split()) / max_reward_len, 2)
    if reward > 1.0:
        reward = 1.0

    final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
    return final_reward, len(think_content.split())


def get_level_key(sample_key: str):
    if sample_key.startswith("Level1"):
        return "lv1"
    if sample_key.startswith("Level2"):
        return "lv2"
    if sample_key.startswith("Level3"):
        return "lv3"
    return None


def compute_ratio(numerator: int, denominator: int):
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100, 2)


def compute_mean(total: float, count: int):
    if count <= 0:
        return None
    return round(total / count, 4)


def metric_str(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def build_empty_record():
    return {
        "correct_lv1": 0,
        "correct_lv2": 0,
        "correct_lv3": 0,
        "format_lv1": 0,
        "format_lv2": 0,
        "format_lv3": 0,
        "length_sum_lv1": 0.0,
        "length_sum_lv2": 0.0,
        "length_sum_lv3": 0.0,
        "total_lv1": 0,
        "total_lv2": 0,
        "total_lv3": 0,
        "scored_total_lv1": 0,
        "scored_total_lv2": 0,
        "scored_total_lv3": 0,
    }


def sort_key(item):
    record = item[1]
    overall_acc = record["overall_acc"]
    overall_format = record["overall_format"]
    overall_length = record["overall_length"]
    return (
        overall_acc if overall_acc is not None else -1,
        overall_format if overall_format is not None else -1,
        overall_length if overall_length is not None else -1,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, default=None, help="Comma-separated model paths")
    args = parser.parse_args()

    args.model_paths = args.model_paths.split(",") if args.model_paths else []
    args.model_paths = [path.strip() for path in args.model_paths if path.strip()]
    score_root = os.getenv("API_BANK_SCORE_ROOT", "PATH_TO_YOUR_SCORE_ROOT")

    for model_path in args.model_paths:
        model_name = build_model_name(model_path)
        save_path = os.path.join(score_root, model_name)
        os.makedirs(save_path, exist_ok=True)
        result_save_path = os.path.join(save_path, "result.json")
        score_save_path = os.path.join(save_path, "score_reward.json")

        if os.path.exists(result_save_path):
            results = json.load(open(result_save_path, "r", encoding="utf-8"))
        else:
            print(f"Skip evaluation because result file does not exist: {result_save_path}")
            continue

        if os.path.exists(score_save_path):
            scores = json.load(open(score_save_path, "r", encoding="utf-8"))
        else:
            scores = {}

        for key, result in results.items():
            sample_data = result.get("data", {})
            tool_calls = result.get("tool_calls", [])
            answer = sample_data.get("answer") if isinstance(sample_data, dict) else None
            raw_output = result.get("raw_output", "")

            score = compute_correctness_score(tool_calls, answer)
            format_score, format_errors = validate_output_format(raw_output)
            length_score, think_word_count = compute_length_score(raw_output)

            result["score"] = score
            result["correct_score"] = score
            result["format_score"] = format_score
            result["length_score"] = length_score
            result["think_word_count"] = think_word_count
            result["format_errors"] = format_errors
            scores[key] = result

        with open(score_save_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=4, ensure_ascii=False)

        print("All done for", model_path)

    leader_board = {}
    for dir_name in os.listdir(score_root):
        score_path = os.path.join(score_root, dir_name, "score_reward.json")
        if not os.path.exists(score_path):
            continue

        record = build_empty_record()
        scores = json.load(open(score_path, "r", encoding="utf-8"))

        for key, result in scores.items():
            level_key = get_level_key(key)
            if level_key is None:
                continue

            record[f"total_{level_key}"] += 1
            record[f"length_sum_{level_key}"] += float(result.get("length_score", 0.0))

            if result.get("format_score", 0) == 1:
                record[f"format_{level_key}"] += 1

            if result.get("score") is not None:
                record[f"scored_total_{level_key}"] += 1
                if result.get("score", 0) == 1:
                    record[f"correct_{level_key}"] += 1

        total_all = record["total_lv1"] + record["total_lv2"] + record["total_lv3"]
        scored_total_all = record["scored_total_lv1"] + record["scored_total_lv2"] + record["scored_total_lv3"]
        correct_all = record["correct_lv1"] + record["correct_lv2"] + record["correct_lv3"]
        format_all = record["format_lv1"] + record["format_lv2"] + record["format_lv3"]
        length_all = record["length_sum_lv1"] + record["length_sum_lv2"] + record["length_sum_lv3"]

        record["lv1_acc"] = compute_ratio(record["correct_lv1"], record["scored_total_lv1"])
        record["lv2_acc"] = compute_ratio(record["correct_lv2"], record["scored_total_lv2"])
        record["lv3_acc"] = compute_ratio(record["correct_lv3"], record["scored_total_lv3"])
        record["overall_acc"] = compute_ratio(correct_all, scored_total_all)

        record["lv1_format"] = compute_ratio(record["format_lv1"], record["total_lv1"])
        record["lv2_format"] = compute_ratio(record["format_lv2"], record["total_lv2"])
        record["lv3_format"] = compute_ratio(record["format_lv3"], record["total_lv3"])
        record["overall_format"] = compute_ratio(format_all, total_all)

        record["lv1_length"] = compute_mean(record["length_sum_lv1"], record["total_lv1"])
        record["lv2_length"] = compute_mean(record["length_sum_lv2"], record["total_lv2"])
        record["lv3_length"] = compute_mean(record["length_sum_lv3"], record["total_lv3"])
        record["overall_length"] = compute_mean(length_all, total_all)

        leader_board[dir_name] = record

    sorted_leader_board = dict(sorted(leader_board.items(), key=sort_key, reverse=True))
    for key, value in sorted_leader_board.items():
        print(
            f"{key}: acc={metric_str(value['overall_acc'], '%')} "
            f"(lv1={metric_str(value['lv1_acc'], '%')}, lv2={metric_str(value['lv2_acc'], '%')}, lv3={metric_str(value['lv3_acc'], '%')}), "
            f"format={metric_str(value['overall_format'], '%')} "
            f"(lv1={metric_str(value['lv1_format'], '%')}, lv2={metric_str(value['lv2_format'], '%')}, lv3={metric_str(value['lv3_format'], '%')}), "
            f"length={metric_str(value['overall_length'])} "
            f"(lv1={metric_str(value['lv1_length'])}, lv2={metric_str(value['lv2_length'])}, lv3={metric_str(value['lv3_length'])})"
        )

    leaderboard_path = os.path.join(score_root, "leaderboard_reward.json")
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(sorted_leader_board, f, indent=4, ensure_ascii=False)
        print("Leaderboard saved to", leaderboard_path)
