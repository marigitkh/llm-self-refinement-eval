import os
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import load_jsonl, save_jsonl
from inference import solve_questions, generate_hints

def main():
    # --- Parse command-line arguments ---
    parser = ArgumentParser(
        description="Compute initial and post-hint inference pipelines."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="HuggingFace path or local dir for the pretrained model"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="JSONL file with the list of questions to solve"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write inference and hint outputs"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="If set, only process this many examples from the input"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and optionally truncate dataset ---
    all_data = load_jsonl(args.input_path)
    data = all_data[: args.max_samples] if args.max_samples else all_data

    # --- Load model & tokenizer once ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",       # automatically spread on available GPUs/CPU
        torch_dtype="auto"       # choose best dtype (fp16/32) for device
    )
    model.eval()

    # --- Step 1: Initial inference on all questions ---
    print("1) Running initial inference...")
    initial_results = solve_questions(data, model, tokenizer)
    save_jsonl(
        initial_results,
        os.path.join(args.output_dir, "initial_inference.jsonl")
    )

    # Filter out the ones answered correctly
    wrong_only = [r for r in initial_results if not r.get("is_correct", False)]

    # --- Step 2: Generate hints for wrong answers ---
    print("2) Generating hints for incorrectly answered questions...")
    hints = generate_hints(wrong_only, model, tokenizer)
    save_jsonl(
        hints,
        os.path.join(args.output_dir, "hints.jsonl")
    )

    # --- Step 3: Post-hint inference with injected hints ---
    print("3) Running post-hint inference...")
    post_results = solve_questions(
        hints,
        model,
        tokenizer,
        inject_hint=True
    )
    save_jsonl(
        post_results,
        os.path.join(args.output_dir, "post_hint_inference.jsonl")
    )

    print(f"\nDone! Outputs saved in directory: {args.output_dir}")

if __name__ == "__main__":
    main()