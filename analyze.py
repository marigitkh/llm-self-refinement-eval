import os
from data import load_jsonl

if __name__ == "__main__":
    initial = load_jsonl("./results/initial_results.jsonl")
    corrected = load_jsonl("./results/corrected_results.jsonl")

    initial_total = len(initial)
    initial_correct = sum(r["is_correct"] for r in initial)

    corrected_total = len(corrected)
    corrected_correct = sum(r["is_correct"] for r in corrected)

    print("Initial Accuracy:", f"{initial_correct}/{initial_total} = {initial_correct/initial_total:.2%}")

    if corrected_total == 0:
        print("No incorrect samples were found. Post-Hint evaluation skipped.")
    else:
        print("Post-Hint Accuracy of the Subset - Initially Wrognly Answered:", f"{corrected_correct}/{corrected_total} = {corrected_correct/corrected_total:.2%}")
