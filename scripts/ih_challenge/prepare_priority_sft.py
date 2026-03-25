"""
Prepare Priority SFT dataset — mixed modes for LoRA instruction hierarchy.

Mode 3 (Knowledge): SQuAD context → LoRA, question → prompt, answer → target
Mode 4 (Neutral): Alpaca instruction → LoRA, input → prompt, response → target

Output: single shuffled JSONL with equal proportions.

Usage:
    uv run scripts/ih_challenge/prepare_priority_sft.py
"""

import json
import random

import pandas as pd

SQUAD_PATH = "data/raw_datasets/squad/plain_text/train-00000-of-00001.parquet"
ALPACA_PATH = "data/raw_datasets/alpaca/alpaca_combined.jsonl"
OUTPUT_PATH = "data/raw_datasets/priority_sft/train.jsonl"
DEV_OUTPUT_PATH = "data/raw_datasets/priority_sft/dev.jsonl"

SAMPLES_PER_MODE = 10000
DEV_SAMPLES_PER_MODE = 250
SEED = 42

# Same templates D2L uses for SQuAD training
CLOSED_QA_TEMPLATES = [
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "Answer without any explanation.\n\nQuestion: {input}",
    "Based on the provided text, what is the answer to the following question? Provide only the answer.\n\nQuestion: {input}",
    "Extract the answer to the question from the text. Be concise. Do not explain.\n\nQuestion: {input}",
    "What is the answer to this question, based on the context? Respond with the answer only.\n\nQuestion: {input}",
]


def prepare_squad(rng: random.Random, n_train: int, n_dev: int) -> tuple[list[dict], list[dict]]:
    """Convert SQuAD to Priority SFT format (mode=knowledge)."""
    df = pd.read_parquet(SQUAD_PATH)
    rows = df.to_dict("records")
    rng.shuffle(rows)

    def convert(rows_slice):
        out = []
        for row in rows_slice:
            answer_text = row["answers"]["text"][0]
            template = rng.choice(CLOSED_QA_TEMPLATES)
            prompt = template.format(input=row["question"])
            out.append({
                "context": row["context"],
                "prompt": prompt,
                "response": answer_text,
                "mode": "knowledge",
            })
        return out

    train = convert(rows[:n_train])
    dev = convert(rows[n_train : n_train + n_dev])
    return train, dev


def prepare_alpaca(rng: random.Random, n_train: int, n_dev: int) -> tuple[list[dict], list[dict]]:
    """Load existing Alpaca combined (mode=neutral)."""
    with open(ALPACA_PATH) as f:
        entries = [json.loads(line) for line in f]
    rng.shuffle(entries)

    def convert(entries_slice):
        out = []
        for e in entries_slice:
            out.append({
                "context": e["context"],
                "prompt": e["prompt"],
                "response": e["response"],
                "mode": "neutral",
            })
        return out

    train = convert(entries[:n_train])
    dev = convert(entries[n_train : n_train + n_dev])
    return train, dev


def main():
    rng = random.Random(SEED)

    squad_train, squad_dev = prepare_squad(rng, SAMPLES_PER_MODE, DEV_SAMPLES_PER_MODE)
    alpaca_train, alpaca_dev = prepare_alpaca(rng, SAMPLES_PER_MODE, DEV_SAMPLES_PER_MODE)

    train = squad_train + alpaca_train
    dev = squad_dev + alpaca_dev
    rng.shuffle(train)
    rng.shuffle(dev)

    import os
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    for path, data in [(OUTPUT_PATH, train), (DEV_OUTPUT_PATH, dev)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    train_modes = {}
    for e in train:
        train_modes[e["mode"]] = train_modes.get(e["mode"], 0) + 1
    dev_modes = {}
    for e in dev:
        dev_modes[e["mode"]] = dev_modes.get(e["mode"], 0) + 1

    print(f"Train: {len(train)} samples — {train_modes}")
    print(f"Dev: {len(dev)} samples — {dev_modes}")
    print(f"Saved to {OUTPUT_PATH} and {DEV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
