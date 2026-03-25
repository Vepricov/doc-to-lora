"""
Prepare Alpaca dataset for D2L SFT training.

Creates 3 datasets in D2L format (context, prompt, response):
  1. alpaca_with_input.jsonl  — instruction→context, input→prompt (10k)
  2. alpaca_no_input.jsonl    — instruction→context, generic→prompt (10k)
  3. alpaca_combined.jsonl    — both merged (20k)

Usage:
    uv run scripts/ih_challenge/prepare_alpaca_sft.py
"""

import json
import os
import random

INPUT_PATH = "data/raw_datasets/alpaca/train.jsonl"
OUTPUT_DIR = "data/raw_datasets/alpaca"
SAMPLES_PER_TYPE = 10000
SEED = 42

GENERIC_PROMPT = "Please respond to the instruction."


def main():
    random.seed(SEED)

    with open(INPUT_PATH) as f:
        entries = [json.loads(line) for line in f]

    with_input = [e for e in entries if e.get("input", "").strip()]
    no_input = [e for e in entries if not e.get("input", "").strip()]

    random.shuffle(with_input)
    random.shuffle(no_input)

    with_input = with_input[:SAMPLES_PER_TYPE]
    no_input = no_input[:SAMPLES_PER_TYPE]

    # Type 1: instruction → context, input → prompt
    type1 = []
    for e in with_input:
        type1.append({
            "context": e["instruction"],
            "prompt": e["input"],
            "response": e["output"],
        })

    # Type 2: instruction → context, generic → prompt
    type2 = []
    for e in no_input:
        type2.append({
            "context": e["instruction"],
            "prompt": GENERIC_PROMPT,
            "response": e["output"],
        })

    # Write
    combined = type1 + type2
    random.shuffle(combined)

    datasets = {
        "alpaca_with_input.jsonl": type1,
        "alpaca_no_input.jsonl": type2,
        "alpaca_combined.jsonl": combined,
    }

    for filename, data in datasets.items():
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"{filename}: {len(data)} samples")


if __name__ == "__main__":
    main()
