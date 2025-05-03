import torch
from utils import extract_answer, extract_chain_of_thought, generate_prompt, generate_hint_prompt, is_valid_hint
from typing import List, Dict, Any

# Solve math questions with optional hint injection and answer forcing fallback
def solve_questions(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    inject_hint: bool = False
) -> List[Dict[str, Any]]:
    # List to collect results
    results = []
    for item in data:
        # Build the prompt (with or without hint)
        prompt = generate_prompt(
            item["question"],
            hint=item.get("hint_sentence") if inject_hint else None
        )
        # Tokenize and move inputs to model device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs["input_ids"].size(1)

        # Primary generation pass
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )[0]
        gen_ids = output_ids[prompt_len:]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Extract answer and chain-of-thought
        answer = extract_answer(decoded)
        cot = extract_chain_of_thought(decoded)

        # Fallback: answer forcing if no answer detected
        if not answer:
            # Append 'Answer:' cue and generate a short continuation
            fallback = f"{prompt}\nAnswer:"
            fb_inputs = tokenizer(fallback, return_tensors="pt", padding=True).to(model.device)
            fb_prompt_len = fb_inputs["input_ids"].size(1)
            fb_ids = model.generate(
                **fb_inputs,
                max_new_tokens=10,
                do_sample=False
            )[0]
            fb_gen = fb_ids[fb_prompt_len:]
            fb_decoded = tokenizer.decode(fb_gen, skip_special_tokens=True).strip()
            answer = extract_answer(fb_decoded)

        # Determine correctness
        is_corr = (answer == item["answer"])
        results.append({
            **item,
            "predicted_answer": answer,
            "is_correct": is_corr,
            "chain_of_thought": cot
        })

    return results


# Generate hint sentences for wrongly answered items with validation
def generate_hints(
    data: List[Dict[str, Any]],
    model,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    num_attempts: int = 3
) -> List[Dict[str, Any]]:
    feedback_items = []
    for item in data:
        # Build the hint-generation prompt
        prefix = generate_hint_prompt(
            item["question"],
            item.get("predicted_answer", ""),
            item.get("chain_of_thought", ""),
            item["answer"]
        )
        inputs = tokenizer(prefix, return_tensors="pt", padding=True).to(model.device)
        prefix_len = inputs["input_ids"].size(1)

        hint = ""
        # Try up to `num_attempts` to get a valid hint
        for attempt in range(num_attempts):
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if attempt > 0:
                # Use sampling after the first attempt
                gen_kwargs.update(do_sample=True, temperature=temperature)

            out_ids = model.generate(**inputs, **gen_kwargs)[0]
            decoded = tokenizer.decode(out_ids[prefix_len:], skip_special_tokens=True).strip()

            if is_valid_hint(decoded, item["answer"]):
                hint = decoded
                break
            hint = decoded

        feedback_items.append({**item, "hint_sentence": hint})

    return feedback_items
