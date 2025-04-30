import json
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    # Load a JSONL file into a list of dictionaries
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    # Save a list of dictionaries to a JSONL file
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
