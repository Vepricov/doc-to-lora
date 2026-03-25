"""
Generate Mode 2 (compatible) data for Priority SFT.

Takes unused Alpaca samples (not in alpaca_combined or dev) and asks an LLM
to generate an additional compatible instruction + target response.

Two variants (10k each, 20k total):
  A) Extra constraint — adds a format/style/length constraint to the existing task
  B) Second task — adds an independent second instruction to fulfill alongside

The original instruction goes to LoRA (context), the additional instruction
goes to the prompt channel. The model should learn to follow BOTH channels
when they don't conflict.

Uses OpenAI-compatible API (works with OpenRouter, local vLLM, etc.).

Usage:
    uv run scripts/ih_challenge/generate_compatible.py \
        --base_url https://openrouter.ai/api/v1 \
        --model openai/gpt-4o-mini

    # Quick test
    uv run scripts/ih_challenge/generate_compatible.py \
        --base_url https://openrouter.ai/api/v1 \
        --model openai/gpt-4o-mini \
        --max_samples 10
"""

import argparse
import asyncio
import json
import logging
import os
import random

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FULL_ALPACA = "data/raw_datasets/alpaca/train.jsonl"
COMBINED_ALPACA = "data/raw_datasets/alpaca/alpaca_combined.jsonl"
DEV_ALPACA = "data/raw_datasets/alpaca/dev.jsonl"
OUTPUT_DIR = "data/raw_datasets/priority_sft"

SEED = 123  # different from prepare_alpaca_sft (42) to avoid ordering artifacts
SAMPLES_PER_VARIANT = 10000

# ---------------------------------------------------------------------------
# Prompts — Variant A: extra constraint
# ---------------------------------------------------------------------------

VARIANT_A_SYSTEM = """\
You are a helpful data-generation assistant. Your job is to take an existing \
instruction and produce:
1. An additional COMPATIBLE constraint (format, style, length, keyword, etc.) \
that does NOT conflict with the original instruction.
2. A response that follows BOTH the original instruction AND your new constraint.

Output valid JSON with two keys:
  "additional_instruction": the extra constraint (1-2 sentences)
  "response": the combined response (following both)

Output ONLY the JSON object. No markdown fences, no explanation."""

VARIANT_A_USER = """\
ORIGINAL INSTRUCTION:
{instruction}

INPUT (may be empty):
{input}

ORIGINAL RESPONSE (for reference — your response should address the same task \
but also satisfy your new constraint):
{output}"""

VARIANT_A_FEWSHOT = [
    {
        "role": "user",
        "content": (
            "ORIGINAL INSTRUCTION:\n"
            "Explain what photosynthesis is.\n\n"
            "INPUT (may be empty):\n\n\n"
            "ORIGINAL RESPONSE (for reference — your response should address "
            "the same task but also satisfy your new constraint):\n"
            "Photosynthesis is the process by which plants convert sunlight "
            "into energy, using carbon dioxide and water to produce glucose "
            "and oxygen."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "additional_instruction":
                "Use exactly 3 bullet points and include the word 'chlorophyll'.",
            "response":
                "- Photosynthesis is the process by which plants use sunlight "
                "to make food from carbon dioxide and water.\n"
                "- The key pigment responsible for capturing light energy is "
                "chlorophyll, found in the chloroplasts of plant cells.\n"
                "- The end products are glucose, which the plant uses for "
                "energy, and oxygen, which is released into the atmosphere.",
        }),
    },
    {
        "role": "user",
        "content": (
            "ORIGINAL INSTRUCTION:\n"
            "Classify the following item as either a fruit or a vegetable.\n\n"
            "INPUT (may be empty):\n"
            "Tomato\n\n"
            "ORIGINAL RESPONSE (for reference — your response should address "
            "the same task but also satisfy your new constraint):\n"
            "Tomato is a fruit."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "additional_instruction":
                "Reply in no more than 15 words and mention the botanical reason.",
            "response":
                "Tomato is botanically a fruit because it develops from "
                "a flower's fertilized ovary.",
        }),
    },
]

# ---------------------------------------------------------------------------
# Prompts — Variant B: second independent task
# ---------------------------------------------------------------------------

VARIANT_B_SYSTEM = """\
You are a helpful data-generation assistant. Your job is to take an existing \
instruction and produce:
1. A SECOND, independent task that the model should also complete alongside \
the original instruction. The second task should be compatible (not conflicting) \
but genuinely different — not just a constraint on the first task.
2. A response that completes BOTH the original instruction AND the second task.

Output valid JSON with two keys:
  "additional_instruction": the second task (1-2 sentences)
  "response": the combined response (completing both tasks)

Output ONLY the JSON object. No markdown fences, no explanation."""

VARIANT_B_USER = """\
ORIGINAL INSTRUCTION:
{instruction}

INPUT (may be empty):
{input}

ORIGINAL RESPONSE (for reference — your response should address the same task \
plus your second task):
{output}"""

VARIANT_B_FEWSHOT = [
    {
        "role": "user",
        "content": (
            "ORIGINAL INSTRUCTION:\n"
            "Write a short greeting for a company newsletter.\n\n"
            "INPUT (may be empty):\n\n\n"
            "ORIGINAL RESPONSE (for reference — your response should address "
            "the same task plus your second task):\n"
            "Hello team! We hope everyone is having a great start to the month."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "additional_instruction":
                "Also write a one-sentence motivational quote to include "
                "after the greeting.",
            "response":
                "Hello team! We hope everyone is having a great start to the "
                "month. Here is your quote of the day: \"Success is not final, "
                "failure is not fatal — it is the courage to continue that "
                "counts.\"",
        }),
    },
    {
        "role": "user",
        "content": (
            "ORIGINAL INSTRUCTION:\n"
            "Convert the temperature from Fahrenheit to Celsius.\n\n"
            "INPUT (may be empty):\n"
            "98.6°F\n\n"
            "ORIGINAL RESPONSE (for reference — your response should address "
            "the same task plus your second task):\n"
            "98.6°F is equal to 37°C."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "additional_instruction":
                "After converting, explain whether this temperature is "
                "considered normal for a human body.",
            "response":
                "98.6°F is equal to 37°C. This is considered the average "
                "normal body temperature for a healthy adult, though it can "
                "vary slightly throughout the day.",
        }),
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_unused_alpaca(seed: int) -> list[dict]:
    """Load Alpaca samples not used in alpaca_combined.jsonl or dev.jsonl."""
    with open(FULL_ALPACA) as f:
        full = [json.loads(l) for l in f]

    # Build fingerprints of used samples
    used_fps = set()
    for path in [COMBINED_ALPACA, DEV_ALPACA]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                fp = (e["context"].strip(), e["response"].strip())
                used_fps.add(fp)

    remaining = []
    for e in full:
        fp = (e["instruction"].strip(), e["output"].strip())
        if fp not in used_fps:
            remaining.append(e)

    rng = random.Random(seed)
    rng.shuffle(remaining)
    return remaining


def load_completed_ids(output_path: str) -> set[str]:
    """Read existing output and return set of completed IDs."""
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

async def generate_one(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    variant: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """Call LLM to generate additional instruction + response."""
    if variant == "A":
        system_prompt = VARIANT_A_SYSTEM
        user_template = VARIANT_A_USER
        fewshot = VARIANT_A_FEWSHOT
    else:
        system_prompt = VARIANT_B_SYSTEM
        user_template = VARIANT_B_USER
        fewshot = VARIANT_B_FEWSHOT

    user_msg = user_template.format(
        instruction=entry["instruction"],
        input=entry.get("input", ""),
        output=entry["output"],
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *fewshot,
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                )
            text = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            parsed = json.loads(text)
            if "additional_instruction" not in parsed or "response" not in parsed:
                raise ValueError(f"Missing keys in response: {list(parsed.keys())}")
            return parsed

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                log.warning(
                    f"Parse error (attempt {attempt + 1}): {e}. "
                    f"Raw: {text[:200]}... Retrying."
                )
                await asyncio.sleep(1)
            else:
                log.error(f"Failed to parse after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                log.error(f"API call failed after {max_retries} attempts: {e}")
                raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args):
    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ATTACKER_API_KEY")
    )
    if not api_key:
        log.error("No API key. Set --api_key or OPENROUTER_API_KEY env var.")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Load unused Alpaca
    remaining = load_unused_alpaca(SEED)
    log.info(f"Loaded {len(remaining)} unused Alpaca samples")

    n_per_variant = args.samples_per_variant
    if args.max_samples > 0:
        n_per_variant = args.max_samples // 2

    variant_a_samples = remaining[:n_per_variant]
    variant_b_samples = remaining[n_per_variant : n_per_variant * 2]
    log.info(f"Variant A (constraint): {len(variant_a_samples)} samples")
    log.info(f"Variant B (second task): {len(variant_b_samples)} samples")

    # Assign IDs
    tasks = []
    for i, entry in enumerate(variant_a_samples):
        tasks.append({"id": f"compat_a_{i:05d}", "variant": "A", "entry": entry})
    for i, entry in enumerate(variant_b_samples):
        tasks.append({"id": f"compat_b_{i:05d}", "variant": "B", "entry": entry})

    # Resume
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = args.output or os.path.join(OUTPUT_DIR, "compatible.jsonl")
    completed_ids = load_completed_ids(output_path)
    if completed_ids:
        log.info(f"Resuming: {len(completed_ids)} already completed")

    pending = [t for t in tasks if t["id"] not in completed_ids]
    log.info(f"Pending: {len(pending)} samples to process")

    if not pending:
        log.info("Nothing to do.")
        return

    # Process in batches
    batch_size = args.max_concurrent * 2
    total_generated = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]

        coros = [
            generate_one(
                client=client,
                model=args.model,
                entry=t["entry"],
                variant=t["variant"],
                semaphore=semaphore,
            )
            for t in batch
        ]

        results = await asyncio.gather(*coros, return_exceptions=True)

        with open(output_path, "a") as f:
            for task_item, result in zip(batch, results):
                if isinstance(result, Exception):
                    log.error(f"Failed {task_item['id']}: {result}")
                    continue

                entry = task_item["entry"]
                # context = original instruction (→ LoRA)
                # prompt = original input + additional instruction (→ prompt channel)
                original_input = entry.get("input", "").strip()
                additional = result["additional_instruction"]

                if original_input:
                    prompt = f"{original_input}\n\nAdditional instruction: {additional}"
                else:
                    prompt = additional

                output = {
                    "id": task_item["id"],
                    "variant": task_item["variant"],
                    "context": entry["instruction"],
                    "prompt": prompt,
                    "response": result["response"],
                    "mode": "compatible",
                }
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
                total_generated += 1

        done = batch_start + len(batch)
        log.info(f"Progress: {done}/{len(pending)} | generated: {total_generated}")

    log.info(f"Done. Total generated: {total_generated}. Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Mode 2 (compatible) data for Priority SFT."
    )
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", required=True, help="Model name (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--api_key", default=None, help="API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument(
        "--samples_per_variant", type=int, default=SAMPLES_PER_VARIANT,
        help=f"Samples per variant (default: {SAMPLES_PER_VARIANT})",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="Total max samples (0 = use samples_per_variant)")
    parser.add_argument("--output", default=None, help="Output path (default: priority_sft/compatible.jsonl)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
