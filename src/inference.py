import torch
from typing import List, Dict, Any
from utils import (
    extract_answer,
    extract_chain_of_thought,
    generate_prompt,
    generate_hint_prompt,
    is_valid_hint
)

def solve_questions(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 512,
    inject_hint: bool = False
) -> List[Dict[str, Any]]:
    """
    Run model inference on a list of questions, optionally injecting hints.

    - For each item, build a prompt (with hint if inject_hint=True).
    - Generate an initial completion and parse out answer and CoT.
    - If no answer found, do a second “answer-forcing” pass.
    - Record prediction, extracted answer, correctness flag, and CoT.
    """
    results = []

    for item in data:
        # 1) Build the prompt text, including hint if requested
        hint_text = item.get("hint_sentence") if inject_hint else None
        prompt = generate_prompt(item["question"], hint=hint_text)

        # 2) Tokenize and move to the model’s device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].size(1)

        # 3) Generate continuation
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )[0]
        # Only the newly generated tokens
        gen_ids = output_ids[prompt_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 4) Extract numeric answer and chain-of-thought
        answer = extract_answer(decoded)
        cot = extract_chain_of_thought(decoded)

        # 5) Fallback pass if no answer was parsed
        if not answer:
            fallback_prompt = f"{prompt}\nAnswer:"
            fb_inputs = tokenizer(fallback_prompt, return_tensors="pt", padding=True).to(model.device)
            fb_len = fb_inputs["input_ids"].size(1)
            fb_output = model.generate(
                **fb_inputs,
                max_new_tokens=10,
                do_sample=False  # deterministic
            )[0]
            fb_decoded = tokenizer.decode(fb_output[fb_len:], skip_special_tokens=True).strip()
            answer = extract_answer(fb_decoded)

        # 6) Mark correctness and assemble result record
        is_corr = (answer == item["answer"])
        results.append({
            **item,
            "prediction":       decoded,
            "predicted_answer": answer,
            "is_correct":       is_corr,
            "chain_of_thought": cot
        })

    return results


def generate_hints(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    num_attempts: int = 3
) -> List[Dict[str, Any]]:
    """
    For each incorrectly answered item, generate a hint prompt and produce a hint.

    - Tries up to num_attempts, first deterministically then with sampling.
    - Validates that the hint does not contain the true answer or the word "correct".
    - Attaches the final hint string to each item.
    """
    hint_items: List[Dict[str, Any]] = []

    for item in data:
        # 1) Build the hint-generation prompt with question, CoT, and true answer
        prefix = generate_hint_prompt(
            question=item["question"],
            predicted_answer=item.get("predicted_answer", ""),
            chain_of_thought=item.get("chain_of_thought", ""),
            correct_answer=item["answer"]
        )
        inputs = tokenizer(prefix, return_tensors="pt", padding=True).to(model.device)
        prefix_len = inputs["input_ids"].size(1)

        hint_sentence = ""
        # 2) Attempt generation multiple times
        for attempt in range(num_attempts):
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            # Use sampling on subsequent tries
            if attempt > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)

            out_ids = model.generate(**inputs, **gen_kwargs)[0]
            decoded = tokenizer.decode(out_ids[prefix_len:], skip_special_tokens=True).strip()

            # 3) Validate the hint
            if is_valid_hint(decoded, item["answer"]) and "correct" not in decoded.lower():
                hint_sentence = decoded
                break
            # Otherwise, keep the last generated text as fallback
            hint_sentence = decoded

        # 4) Append hint to the original item
        hint_items.append({**item, "hint_sentence": hint_sentence})

    return hint_items