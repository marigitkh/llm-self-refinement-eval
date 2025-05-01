# Capstone Project: Do LLMs Know What/Where and Why They Lack?

This project evaluates the self-refinement abilities of large language models (LLMs) by:
- Solving math questions
- Identifying incorrect answers
- Generating hint sentences
- Re-solving the questions with injected hints
- Measuring the improvement in accuracy

## Project Structure

utils.py         # Prompt generation, extraction, validation functions  
data.py          # Loading and saving JSONL data  
inference.py     # Core logic: solving questions, generating hints (pure functions)  
run.py           # Main script to run the full pipeline (using argparse)  
analyze.py       # Script to calculate and display accuracy improvements  
results/         # Folder where output files will be saved  

## How to Run

1. Install dependencies:

   pip install transformers torch

2. Run the main inference and self-correction pipeline:

   python run.py --model_path "google/gemma-2-2b-it" --input_path "data/asdiv.jsonl" --output_dir "./results" --max_samples 300

3. Analyze the accuracy before and after hint injection:

   python analyze.py

## Outputs

- initial_results.jsonl — Model's first-pass answers  
- wrong_only.jsonl      — Subset of questions answered incorrectly  
- hints.jsonl           — Hint sentences generated for incorrect answers  
- corrected_results.jsonl — Model's re-answers after receiving hints  

---
