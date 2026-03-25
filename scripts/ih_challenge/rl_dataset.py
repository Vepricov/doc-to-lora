"""
IH-Challenge RL dataset loader.

Loads task skeletons from IH-Challenge benchmarks for online RL training.
Each sample provides instruction, attacker_problem, grader_code, and metadata.
Attacks are generated online by the attacker LLM during training.
"""

import json
import logging
import os

from torch.utils.data import Dataset

log = logging.getLogger(__name__)

SPLIT_FILES = {
    "single-constraint": "single-constraint.jsonl",
    "multi-constraint": "multi-constraint.jsonl",
    "input-conditioned": "input-conditioned.jsonl",
    "anti-overrefusal": "anti-overrefusal.jsonl",
}


def extract_instruction(defender_template: list[dict]) -> tuple[str, str]:
    """Extract the high-priority instruction and the attack placeholder role."""
    instruction_parts = []
    attack_role = None
    for msg in defender_template:
        if "<<<REPLACE THIS WITH YOUR ATTACK>>>" in msg["content"]:
            attack_role = msg["role"]
        else:
            instruction_parts.append(msg["content"])
    return "\n\n".join(instruction_parts), attack_role


class IHChallengeRLDataset(Dataset):
    """
    Dataset of IH-Challenge task skeletons for RL training.

    Each sample provides:
    - instruction: high-priority text → goes to hypernetwork → LoRA
    - attacker_problem: task description for the attacker LLM
    - attack_role: which role the attack fills (user/developer)
    - grader_code: Python grader source for reward computation
    - task_type, task_id: metadata for logging
    """

    def __init__(
        self,
        data_dir: str,
        splits: list[str] | None = None,
        task_types: list[str] | None = None,
        max_per_type: int = 0,
    ):
        self.entries = []
        splits = splits or list(SPLIT_FILES.keys())

        type_counts: dict[str, int] = {}

        for split_name in splits:
            path = os.path.join(data_dir, SPLIT_FILES[split_name])
            if not os.path.exists(path):
                log.warning(f"Split file not found: {path}")
                continue

            with open(path) as f:
                for line in f:
                    entry = json.loads(line)
                    meta = entry["metadata"]
                    task_type = meta["task_type"]

                    if task_types and task_type not in task_types:
                        continue

                    if max_per_type > 0:
                        count = type_counts.get(task_type, 0)
                        if count >= max_per_type:
                            continue
                        type_counts[task_type] = count + 1

                    instruction, attack_role = extract_instruction(
                        entry["defender_problem_template"]
                    )

                    self.entries.append({
                        "task_id": entry.get("id", f"{split_name}_{len(self.entries)}"),
                        "instruction": instruction,
                        "attacker_problem": entry["attacker_problem"],
                        "attack_role": attack_role or "user",
                        "grader_code": meta["grader_code_python"],
                        "task_type": task_type,
                        "source_split": split_name,
                    })

        log.info(f"Loaded {len(self.entries)} task skeletons from {splits}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> dict:
        return self.entries[idx]

    @staticmethod
    def collate_fn(batch: list[dict]) -> list[dict]:
        """Pass through as list — RL training processes samples individually."""
        return batch
