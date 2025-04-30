import re
from typing import List, Dict, Any

# Precompiled regex to extract chain of thought segments
_COT_PATTERN = re.compile(r"<cot_start>(.*?)<cot_end>", re.DOTALL)


def extract_chain_of_thought(text: str) -> str:
    # Extract chain of thought text between <cot_start> and <cot_end>
    segments = _COT_PATTERN.findall(text)
    return " ".join(seg.strip() for seg in segments) if segments else ""


def extract_answer(text: str) -> str:
    # Extract the numerical answer following the 'Answer:' tag
    for line in text.splitlines():
        if line.strip().lower().startswith("answer:"):
            part = line.split(":", 1)[1]
            m = re.search(r"-?\d+(?:[.,]\\d+)?", part)
            if not m:
                return ""
            return m.group(0).replace(",", ".")
    return ""


def generate_prompt(question: str, hint: str = None) -> str:
    # Load the prompt template for solving math questions
    with open("prompts/answer_prompt.txt", "r") as f:
        base_prompt = f.read()

    prompt = f"{base_prompt}\nNow solve this:\nQuestion: {question}\n"
    if hint:
        prompt += f"Additional Context: {hint}\n"
    return prompt


def generate_hint_prompt(question: str, predicted_answer: str, chain_of_thought: str, correct_answer: str) -> str:
    # Load the prompt template for generating a hint
    with open("prompts/hint_prompt.txt", "r") as f:
        base_hint_prompt = f.read()

    return base_hint_prompt.format(
        question=question,
        predicted_answer=predicted_answer,
        chain_of_thought=chain_of_thought,
        correct_answer=correct_answer
    )


def is_valid_hint(hint: str, correct_answer: str) -> bool:
    # Validate that the hint doesn't contain digits or leak the answer
    if re.search(r"\\d", hint):
        return False
    if str(correct_answer) in hint:
        return False
    return True
