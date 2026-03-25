"""
SFT dataset for D2L instruction-following training.

Loads (context, prompt, response) triplets from JSONL files.
Supports mode filtering (conflict, knowledge, neutral, compatible).
Each sample is tokenized into:
  - ctx_ids: instruction tokenized for the hypernetwork (→ LoRA)
  - input_ids: prompt + response tokenized as chat (→ base model input)
  - labels: -100 for prompt tokens, token ids for response tokens (→ CE loss target)
"""

import json
import logging
from collections import Counter

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class IHChallengeSFTDataset(Dataset):
    """
    Dataset of (context, prompt, response) triplets for SFT.

    context → tokenized for hypernetwork → LoRA
    prompt + response → tokenized as chat → CE loss on response tokens

    Supports filtering by mode (conflict/knowledge/neutral/compatible).
    Preserves all entry fields as metadata for eval breakdown.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        ctx_tokenizer,
        max_samples: int = 0,
        max_response_tokens: int = 512,
        modes: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.max_response_tokens = max_response_tokens
        self.entries = []

        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                # Skip entries without gold response
                if entry.get("response") in (None, "", "__GRADER_EVAL__"):
                    continue
                # Filter by mode if specified
                if modes is not None and entry.get("mode") not in modes:
                    continue
                self.entries.append(entry)

        if max_samples > 0:
            self.entries = self.entries[:max_samples]

        mode_counts = Counter(e.get("mode", "unknown") for e in self.entries)
        log.info(f"Loaded {len(self.entries)} SFT samples from {data_path} (modes: {dict(mode_counts)})")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> dict:
        entry = self.entries[idx]
        instruction = entry["context"]
        attack = entry["prompt"]
        gold_response = entry["response"]

        # 1. Tokenize instruction for hypernetwork
        ctx_inputs = self.ctx_tokenizer.apply_chat_template(
            [[{"role": "system", "content": ""}, {"role": "user", "content": instruction.strip()}]],
            tokenize=True, add_generation_prompt=True, return_attention_mask=False,
            padding=False, truncation=False, add_special_tokens=False, return_dict=True,
        )
        ctx_ids = torch.tensor([ctx_inputs["input_ids"][0]], dtype=torch.long)  # [1, ctx_len]

        # 2. Tokenize attack (prompt) with chat template — get the prompt portion
        prompt_messages = [{"role": "user", "content": attack}]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True,
        )

        # 3. Tokenize response
        response_ids = self.tokenizer.encode(
            gold_response, add_special_tokens=False,
        )
        # Truncate response if too long
        if len(response_ids) > self.max_response_tokens:
            response_ids = response_ids[:self.max_response_tokens]
        # Add EOS
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        # 4. Build input_ids = prompt + response, labels = -100 for prompt, token ids for response
        full_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "ctx_ids": ctx_ids,
            "ctx_attn_mask": torch.ones_like(ctx_ids),
            "input_ids": input_ids.unsqueeze(0),      # [1, seq_len] — model expects batch dim
            "attention_mask": attention_mask.unsqueeze(0),
            "labels": labels.unsqueeze(0),
        }
