import os
import argparse
from src.data import load_jsonl

def format_stats(result_dir):
    model = os.path.basename(os.path.dirname(result_dir))
    dataset = os.path.basename(result_dir)
    label = f"{model}/{dataset}"

    # paths
    init_path = os.path.join(result_dir, "initial_inference.jsonl")
    post_path = os.path.join(result_dir, "post_hint_inference.jsonl")

    if not os.path.exists(init_path):
        return None

    initial = load_jsonl(init_path)
    init_tot  = len(initial)
    if init_tot == 0:
        return None

    init_corr = sum(r.get("is_correct", False) for r in initial)
    init_wrong = init_tot - init_corr

    # load post-hint (only the initially wrong ones)
    post_corr = 0
    if os.path.exists(post_path):
        post = load_jsonl(post_path)
        post_corr = sum(r.get("is_correct", False) for r in post)

    # build the two-line block
    lines = [
        f"Model/Dataset: {label}",
        f"  Initial Accuracy: {init_corr}/{init_tot} = {init_corr/init_tot:.2%}"
    ]
    # only report post-hint if there were incorrect answers
    if init_wrong > 0:
        lines.append(
            f"  Post-Hint Accuracy on those initially wrong: "
            f"{post_corr}/{init_wrong} = {post_corr/init_wrong:.2%}"
        )

    return "\n".join(lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute initial and post-hint accuracies for each model/dataset."
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default="results",
        help="Parent directory containing model subfolders"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="summary.txt",
        help="Where to write the TXT summary"
    )
    args = parser.parse_args()

    blocks = []
    for model in sorted(os.listdir(args.parent_dir)):
        model_dir = os.path.join(args.parent_dir, model)
        if not os.path.isdir(model_dir):
            continue

        for dataset in sorted(os.listdir(model_dir)):
            ds_dir = os.path.join(model_dir, dataset)
            if not os.path.isdir(ds_dir):
                continue

            blk = format_stats(ds_dir)
            if blk:
                print(blk, "\n")
                blocks.append(blk)

    with open(args.output_file, "w") as fout:
        fout.write("\n\n".join(blocks))

    print(f"Wrote summaries for {len(blocks)} model/dataset pairs to {args.output_file}")
