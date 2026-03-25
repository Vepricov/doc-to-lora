"""
Assign unique IDs to all IH-Challenge entries across splits.

IDs are prefixed by split to avoid intersections:
  sc_NNNN  (single-constraint)
  mc_NNNN  (multi-constraint)
  ic_NNNNN (input-conditioned)
  ar_NNNN  (anti-overrefusal)

Overwrites original files in place.

Usage:
    uv run scripts/ih_challenge/assign_ids.py
"""

import json
import os

DATA_DIR = "data/benchmarks/ih-challenge"

SPLITS = [
    ("single-constraint.jsonl", "sc"),
    ("multi-constraint.jsonl", "mc"),
    ("input-conditioned.jsonl", "ic"),
    ("anti-overrefusal.jsonl", "ar"),
]


def main():
    for filename, prefix in SPLITS:
        path = os.path.join(DATA_DIR, filename)
        entries = []
        with open(path) as f:
            for line in f:
                entries.append(json.loads(line))

        # Determine zero-padding width
        width = len(str(len(entries) - 1))

        for idx, entry in enumerate(entries):
            entry["id"] = f"{prefix}_{idx:0{width}d}"

        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"{filename}: {len(entries)} entries, IDs {entries[0]['id']} .. {entries[-1]['id']}")


if __name__ == "__main__":
    main()
