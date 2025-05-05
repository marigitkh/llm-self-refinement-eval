# Capstone Project: Do LLMs Know What/Where and Why They Lack?

This codebase was developed as a part of Bachelor’s capstone project in Data Science at American University of Armenia to investigate the self-refinement capabilities of large language models (LLMs).

The project evaluates the self-refinement abilities of large language models (LLMs) by:
- Solving math questions
- Identifying incorrect answers
- Generating hint sentences given the correct answer
- Re-solving the questions with injected hints
- Measuring the improvement in accuracy

## Project Structure

```
src/utils.py         # Prompt generation, extraction, validation functions
src/data.py          # Loading and saving JSONL data
src/inference.py     # Core logic: solving questions, generating hints (pure functions)
scripts/run.py       # Main script to run the full pipeline (using argparse)
scripts/statistics.py# Script to calculate and display accuracy improvements
results/             # Folder where output files will be saved
```

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the full pipeline

This will generate initial answers, hints, and post-hint answers:

```
python scripts/run.py \
  --model_path <HF_MODEL_NAME> \
  --input_path data/<DATASET>.jsonl \
  --output_dir results/<MODEL_NAME>/<DATASET>
```

3. Analyze accuracy improvements
Summarize initial vs. post-hint accuracy across all model/dataset folders:

```
python scripts/statistics.py \
  --parent_dir results \
  --output_file results/statistics.txt
```

**Note:** Always run these commands from the project root so that the src/ package is on Python’s import path.

## Data, Prompts & Results

### Data
Stores raw JSONL datasets for arithmetic reasoning benchmarks.
- `gsm8k.jsonl`
- `asdiv.jsonl`

These files serve as inputs to the pipeline via the `--input_path` argument.

---

### Prompts
Stores text-based prompt templates for model interactions:  
- `answer_prompt.txt` — template for initial answer generation  
- `hint_prompt.txt` — template for “answer-free” hint generation  

Editing these files adjusts how questions and hints are framed.

---

### Results  
For each LLM/dataset pair the following results are saved:
- `initial_results.jsonl` — Model's first-pass answers
- `wrong_only.jsonl` — Subset of questions answered incorrectly
- `hints.jsonl` — Hint sentences generated for incorrect answers
- `corrected_results.jsonl` — Model's re-answers after receiving hints
- `statistics.txt` — Summary file reporting accuracy before and after hint injection
  
Specifically, `gsm8k.jsonl` and `asdiv.jsonl` were tested on `gemma-2-2b-it` and `phi-4-mini-instruct` in the scope of the research.

---
