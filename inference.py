import torch
from utils import extract_answer, extract_chain_of_thought, generate_prompt, generate_hint_prompt, is_valid_hint
from typing import List, Dict, Any


def solve_questions(data: List[Dict[str, Any]], model, tokenizer, max_new_tokens: int = 128, inject_hint: bool = False) -> List[Dict[str, Any]]:
    # Run inference to solve math questions, optionally using injected hints
    results = []
    for item in data:
        prompt = generate_prompt(item["question"], hint=item.get("hint_sentence") if inject_hint else None)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        prompt_len = inputs["input_ids"].size(1)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
        gen_ids = output_ids[prompt_len:]

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        answer = extract_answer(text)
        cot = extract_chain_of_thought(text)

        try:
            is_correct = float(answer) == float(item["answer"])
        except ValueError:
            is_correct = False

        results.append({
            **item,
            "prediction": text,
            "predicted_answer": answer,
            "is_correct": is_correct,
            "chain_of_thought": cot,
        })
    return results


def generate_hints(wrong_results: List[Dict[str, Any]], model, tokenizer, max_new_tokens: int = 64, temperature: float = 0.7, max_attempts: int = 3) -> List[Dict[str, Any]]:
    # Generate hint sentences for wrongly answered questions
    feedback_items = []

    for item in wrong_results:
        prompt = generate_hint_prompt(
            question=item["question"],
            predicted_answer=item["predicted_answer"],
            chain_of_thought=item["chain_of_thought"],
            correct_answer=item["answer"],
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
        prefix_len = inputs["input_ids"].size(1)

        hint = ""
        for attempt in range(max_attempts):
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if attempt > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature)

            output_ids = model.generate(**inputs, **gen_kwargs)[0]
            decoded = tokenizer.decode(output_ids[prefix_len:], skip_special_tokens=True).strip()

            if is_valid_hint(decoded, item["answer"]):
                hint = decoded
                break
            hint = decoded

        feedback_items.append({**item, "hint_sentence": hint})

    return feedback_items
