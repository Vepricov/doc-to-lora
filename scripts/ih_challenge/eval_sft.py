"""
Evaluate D2L SFT checkpoints on a dev set via generation.

For each sample: encodes instruction into LoRA, generates response from prompt.
Metrics:
  - conflict mode: runs Python grader (grader_code field), reports pass rate
    with breakdown by task_type, dev_split (holdout/in_distribution), level
  - knowledge/neutral: computes ROUGE-L

Usage:
    uv run scripts/ih_challenge/eval_sft.py \
        --checkpoint_path train_outputs/sft_priority/checkpoint-5000/pytorch_model.bin \
        --data_path data/raw_datasets/priority_sft/dev_v2.jsonl

    # Eval only conflict mode
    uv run scripts/ih_challenge/eval_sft.py \
        --checkpoint_path ... --data_path ... --modes conflict
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer
from sft_dataset import IHChallengeSFTDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--max_samples", type=int, default=0, help="0 = all")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--modes", default=None,
                   help="Comma-separated modes to eval (e.g. conflict,knowledge). Default: all")
    p.add_argument("--output_path", default=None, help="Auto-generated if not set")
    return p.parse_args()


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L F1 (word-level LCS)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / m if m > 0 else 0.0
    recall = lcs_len / n if n > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, reference: str) -> float:
    """Token-level F1 score."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def run_grader(grader_code: str, prompt: str, response: str) -> bool:
    """Execute IH-Challenge grader code and return pass/fail."""
    try:
        namespace = {}
        exec(grader_code, namespace)
        return bool(namespace["grade_output_correct"](prompt, response))
    except Exception as e:
        log.warning(f"Grader error: {e}")
        return False


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto output path
    if args.output_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint_path)
        run_dir = os.path.dirname(ckpt_dir)
        data_name = os.path.splitext(os.path.basename(args.data_path))[0]
        ckpt_name = os.path.basename(ckpt_dir)
        args.output_path = os.path.join(run_dir, f"eval-{ckpt_name}-{data_name}.jsonl")

    # Load model
    log.info(f"Loading: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_flash_attn=True, use_sequence_packing=False,
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    # Tokenizers
    base_model_name = model.base_model.config.name_or_path
    ctx_encoder_name = getattr(
        state_dict.get("ctx_encoder_args"), "ctx_encoder_model_name_or_path", None
    ) or base_model_name
    tokenizer = get_tokenizer(base_model_name, train=False)
    ctx_tokenizer = get_tokenizer(ctx_encoder_name, train=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    modes = args.modes.split(",") if args.modes else None
    dataset = IHChallengeSFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        ctx_tokenizer=ctx_tokenizer,
        max_samples=args.max_samples,
        modes=modes,
    )

    log.info(f"Evaluating {len(dataset)} samples -> {args.output_path}")
    results = []
    t0 = time.time()

    # Accumulators for per-mode metrics
    rouge_by_mode = defaultdict(list)
    f1_by_mode = defaultdict(list)
    grader_by_mode = defaultdict(list)  # conflict only
    grader_by_type = defaultdict(list)
    grader_by_split = defaultdict(list)
    grader_by_level = defaultdict(list)

    for i in range(len(dataset)):
        entry = dataset.entries[i]
        sample = dataset[i]
        mode = entry.get("mode", "unknown")

        ctx_ids = sample["ctx_ids"].to(device)
        ctx_attn_mask = sample["ctx_attn_mask"].to(device)

        # Build prompt-only input_ids
        prompt_messages = [{"role": "user", "content": entry["prompt"]}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                n_ctx_chunks=torch.tensor([ctx_ids.shape[0]], device=device),
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        generated_ids = output_ids[0][len(prompt_ids):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Metrics
        result = {
            "idx": i,
            "mode": mode,
            "instruction": entry["context"],
            "prompt": entry["prompt"],
            "gold_response": entry["response"],
            "generated": generated_text,
        }

        if mode == "conflict" and entry.get("grader_code"):
            passed = run_grader(entry["grader_code"], entry["prompt"], generated_text)
            result["grader_pass"] = passed
            result["task_type"] = entry.get("task_type", "")
            result["dev_split"] = entry.get("dev_split", "")
            result["level"] = entry.get("level", 0)

            grader_by_mode[mode].append(passed)
            grader_by_type[entry.get("task_type", "unknown")].append(passed)
            grader_by_split[entry.get("dev_split", "unknown")].append(passed)
            grader_by_level[entry.get("level", 0)].append(passed)
        else:
            rouge = compute_rouge_l(generated_text, entry["response"])
            f1 = compute_f1(generated_text, entry["response"])
            result["rouge_l"] = round(rouge, 4)
            result["f1"] = round(f1, 4)
            rouge_by_mode[mode].append(rouge)
            f1_by_mode[mode].append(f1)

        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            log.info(f"[{i+1}/{len(dataset)}] ({elapsed:.0f}s, {elapsed/(i+1):.1f}s/sample)")

    # Save results
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Summary ---
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"RESULTS — {len(results)} samples, {elapsed:.0f}s total")
    log.info(f"{'='*60}")

    # ROUGE-L and F1 by mode
    for mode in sorted(rouge_by_mode):
        avg_rouge = sum(rouge_by_mode[mode]) / len(rouge_by_mode[mode])
        avg_f1 = sum(f1_by_mode[mode]) / len(f1_by_mode[mode])
        log.info(f"  [{mode}] ROUGE-L: {avg_rouge:.4f}, F1: {avg_f1:.4f} (n={len(rouge_by_mode[mode])})")

    # Grader by mode
    for mode, passes in sorted(grader_by_mode.items()):
        rate = sum(passes) / len(passes) * 100
        log.info(f"  [{mode}] Grader pass: {rate:.1f}% ({sum(passes)}/{len(passes)})")

    # Grader by task_type
    if grader_by_type:
        log.info(f"\n  Conflict — by task_type:")
        for tt in sorted(grader_by_type, key=lambda t: sum(grader_by_type[t])/len(grader_by_type[t])):
            passes = grader_by_type[tt]
            rate = sum(passes) / len(passes) * 100
            log.info(f"    {tt:<25} {rate:>5.1f}% ({sum(passes)}/{len(passes)})")

    # Grader by dev_split
    if grader_by_split:
        log.info(f"\n  Conflict — by dev_split:")
        for ds in sorted(grader_by_split):
            passes = grader_by_split[ds]
            rate = sum(passes) / len(passes) * 100
            log.info(f"    {ds:<25} {rate:>5.1f}% ({sum(passes)}/{len(passes)})")

    # Grader by level
    if grader_by_level:
        log.info(f"\n  Conflict — by level:")
        for lv in sorted(grader_by_level):
            passes = grader_by_level[lv]
            rate = sum(passes) / len(passes) * 100
            log.info(f"    L{lv:<24} {rate:>5.1f}% ({sum(passes)}/{len(passes)})")

    log.info(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
