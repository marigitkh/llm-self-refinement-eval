import os
import argparse
from data import load_jsonl

def format_stats(result_dir: str) -> str:
    """
    Given a directory for one model/dataset run, loads
    the initial and post-hint inference JSONL files,
    computes accuracy metrics, and returns a formatted
    stats block.
    """
    # Derive model and dataset names from directory path
    model_name   = os.path.basename(os.path.dirname(result_dir))
    dataset_name = os.path.basename(result_dir)
    label = f"{model_name}/{dataset_name}"

    # Define expected file paths
    init_path = os.path.join(result_dir, "initial_inference.jsonl")
    post_path = os.path.join(result_dir, "post_hint_inference.jsonl")

    # If no initial results, skip
    if not os.path.exists(init_path):
        return None

    # Load initial inference records
    initial = load_jsonl(init_path)
    total = len(initial)
    if total == 0:
        return None  # nothing to report

    # Count correct vs incorrect initially
    correct_initial = sum(r.get("is_correct", False) for r in initial)
    wrong_initial   = total - correct_initial

    # Load post-hint results if available, else assume zero gains
    post_correct = 0
    if os.path.exists(post_path):
        post = load_jsonl(post_path)
        post_correct = sum(r.get("is_correct", False) for r in post)

    # Compute metrics
    before_acc = correct_initial / total
    after_correct = correct_initial + post_correct
    after_acc = after_correct / total
    delta_pct = (after_acc - before_acc) * 100

    # Build the formatted output lines
    lines = [
        f"Model/Dataset: {label}",
        f"  Total Questions Evaluated: {total}",
        f"  Incorrect Initially: {wrong_initial}",
        f"  Accuracy Before Hints: {correct_initial}/{total} = {before_acc:.2%}",
        (
            f"  Net Gain on Wrong Subset: {post_correct}/{wrong_initial} = "
            f"{(post_correct/wrong_initial):.2%}"
            if wrong_initial > 0
            else "  No initially wrong questions to improve"
        ),
        f"  Accuracy After Hints: {after_correct}/{total} = {after_acc:.2%}",
        f"  Δ Accuracy: {delta_pct:+.2f}%"
    ]
    return "\n".join(lines)


def main():
    # Set up CLI arguments
    parser = argparse.ArgumentParser(
        description="Compute and print accuracy stats for each model/dataset."
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="results",
        help="Top-level results folder containing model subdirectories"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="statistics.txt",
        help="File path to write the combined summary"
    )
    args = parser.parse_args()

    stats_blocks = []

    # Walk through each model and dataset folder
    for model in sorted(os.listdir(args.parent_dir)):
        model_dir = os.path.join(args.parent_dir, model)
        if not os.path.isdir(model_dir):
            continue

        for dataset in sorted(os.listdir(model_dir)):
            ds_dir = os.path.join(model_dir, dataset)
            if not os.path.isdir(ds_dir):
                continue

            block = format_stats(ds_dir)
            if block:
                print(block + "\n")
                stats_blocks.append(block)

    # Write all blocks to the output file
    with open(args.output_file, "w") as fout:
        fout.write("\n\n".join(stats_blocks))

    print(f"Wrote summaries for {len(stats_blocks)} model/dataset pairs to '{args.output_file}'")


if __name__ == "__main__":
    main()
