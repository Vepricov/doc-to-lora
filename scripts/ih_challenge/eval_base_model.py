"""
Run IH-Challenge eval on base models (no D2L) with instruction in prompt.

For comparison: same tasks but instruction goes into context window as part of
the prompt, not through LoRA. Shows what the base model can do with in-context
instructions vs what D2L adds via LoRA.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/ih_challenge/eval_base_model.py \
        --model google/gemma-2-2b-it \
        --dataset data/raw_datasets/ih_challenge/baseline.jsonl \
        --output results/ih_challenge_base_gemma2b.jsonl
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--dataset", default="data/raw_datasets/ih_challenge/baseline.jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Load dataset
    samples = []
    with open(args.dataset) as f:
        for line in f:
            samples.append(json.loads(line))
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples")

    # Generate
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []

    for i, sample in enumerate(samples):
        instruction = sample["context"]
        query = sample["prompt"]

        # Format as chat: system=instruction, user=query
        messages = [
            {"role": "user", "content": f"{instruction}\n\n{query}"},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        # Decode only the generated part
        generated_ids = output_ids[0][input_ids["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        results.append({
            "label": sample.get("response", ""),
            "input": input_text,
            "generated": generated_text,
            "context": instruction,
            "input_ids_len": input_ids["input_ids"].shape[1],
            "ctx_ids_len": 0,
            "rougeL.f1": 0.0,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['metadata']['task_type']}: {generated_text[:80]}")

    # Write output in same format as D2L eval
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
