import json
from typing import List, Dict, Any

def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL (JSON Lines) file and return a list of records.
    """
    records: List[Dict[str, Any]] = []
    with open(filepath, "r") as f:
        for line in f:
            # Parse each line as JSON and append to the list
            records.append(json.loads(line))
    return records


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    """
    with open(filepath, "w") as f:
        for item in data:
            # Serialize the dict to JSON and write with a trailing newline
            f.write(json.dumps(item) + "\n")