import os
import argparse
from data import load_jsonl

def compute_and_print_stats(result_dir):
    # build full paths
    initial_path   = os.path.join(result_dir, "initial_results.jsonl")
    corrected_path = os.path.join(result_dir, "corrected_results.jsonl")

    # skip if files aren’t there
    if not os.path.exists(initial_path):
        print(f"  ✗ no initial_results.jsonl in {result_dir}")
        return
    if not os.path.exists(corrected_path):
        print(f"  ✗ no corrected_results.jsonl in {result_dir}")
        return

    # load the data
    initial   = load_jsonl(initial_path)
    corrected = load_jsonl(corrected_path)

    # compute stats
    init_tot     = len(initial)
    init_corr    = sum(r["is_correct"] for r in initial)
    corr_tot     = len(corrected)
    corr_corr    = sum(r["is_correct"] for r in corrected)

    # print
    print(f"Directory: {os.path.basename(result_dir)}")
    print(f"  Initial Accuracy: {init_corr}/{init_tot} = {init_corr/init_tot:.2%}")
    if corr_tot == 0:
        print("  No incorrect samples were found. Post-Hint evaluation skipped.")
    else:
        print(
            f"  Post-Hint Accuracy on those initially wrong: "
            f"{corr_corr}/{corr_tot} = {corr_corr/corr_tot:.2%}"
        )
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute initial and post-hint accuracies from JSONL result files in each sub-directory."
    )
    parser.add_argument(
        "--parent_dir",
        type=str,
        default=".",
        help="Parent directory containing one folder per experiment (each with initial_results.jsonl & corrected_results.jsonl)"
    )
    args = parser.parse_args()

    # loop through every subfolder
    for sub in sorted(os.listdir(args.parent_dir)):
        subpath = os.path.join(args.parent_dir, sub)
        if os.path.isdir(subpath):
            compute_and_print_stats(subpath)
