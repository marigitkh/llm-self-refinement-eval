import os
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import extract_answer, extract_chain_of_thought, generate_prompt, generate_hint_prompt, is_valid_hint
from src.data import load_jsonl, save_jsonl
from src.inference import solve_questions, generate_hints

if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load initial dataset
    if args.max_samples:
        data = load_jsonl(args.input_path)[:args.max_samples]
    else:
        data = load_jsonl(args.input_path)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype="auto")
    model.eval()

    print("Running initial inference...")
    initial_results = solve_questions(data, model, tokenizer)
    save_jsonl(initial_results, os.path.join(args.output_dir, "initial_results.jsonl"))

    wrong = [r for r in initial_results if not r["is_correct"]]
    save_jsonl(wrong, os.path.join(args.output_dir, "wrong_only.jsonl"))

    print("Generating hints...")
    feedback = generate_hints(wrong, model, tokenizer)
    save_jsonl(feedback, os.path.join(args.output_dir, "hints.jsonl"))

    print("Running corrected inference...")
    corrected_results = solve_questions(feedback, model, tokenizer, inject_hint=True)
    save_jsonl(corrected_results, os.path.join(args.output_dir, "corrected_results.jsonl"))

    print("Done!")
