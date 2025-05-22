import re
from typing import List, Dict, Any

# Compile once: capture everything between CoT tags 
_COT_PATTERN = re.compile(r"<cot_start>(.*?)<cot_end>", re.DOTALL)


def extract_chain_of_thought(text: str) -> str:
    """
    Pull out all chain-of-thought segments marked by <cot_start>…<cot_end>.
    Returns the concatenated inner text, or an empty string if none found.
    """
    segments = _COT_PATTERN.findall(text)
    # Strip whitespace and join multiple segments with a space
    return " ".join(seg.strip() for seg in segments) if segments else ""


def extract_answer(text: str) -> str:
    """
    Find the final numeric answer in a model’s output.
    Looks for “Answer: <number>” (allows commas and decimals).
    Returns the last match with commas removed, or empty if none.
    """
    pattern = r"Answer:\s*([-+]?[0-9,]+(?:\.\d+)?)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return ""
    # Use the last reported answer (in case the model repeated)
    raw = matches[-1]
    return raw.replace(",", "")


def generate_prompt(question: str, hint: str = None) -> str:
    """
    Build the full prompt for initial or hinted inference.
    Reads a static template, appends the question, and injects a hint if provided.
    """
    with open("prompts/answer_prompt.txt", "r") as f:
        template = f.read()

    prompt = (
        f"{template}\n"
        f"Now solve this:\n"
        f"Question: {question}\n"
    )
    if hint:
        prompt += f"Additional Context (Hint): {hint}\n"
    return prompt


def generate_hint_prompt(
    question: str,
    predicted_answer: str,
    chain_of_thought: str,
    correct_answer: str
) -> str:
    """
    Create the prompt for hint generation.
    """
    with open("prompts/hint_prompt.txt", "r") as f:
        hint_template = f.read()

    return hint_template.format(
        question=question,
        predicted_answer=predicted_answer,
        chain_of_thought=chain_of_thought,
        correct_answer=correct_answer
    )


def is_valid_hint(hint: str, correct_answer: str) -> bool:
    """
    Ensure the generated hint does not directly include the correct answer.
    Returns False if the exact answer string appears in the hint.
    """
    return str(correct_answer) not in hint