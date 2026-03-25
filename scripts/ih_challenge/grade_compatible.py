"""Grade compatible mode data using LLM-as-judge.

Checks each (context, prompt, response) triple for:
1. Does the response follow the main instruction (context)?
2. Does the response follow the additional instruction (prompt)?
3. Are there arithmetic errors?
4. Are there factual errors or inaccuracies?

Outputs pass/fail + reason. Async, resumable.

Usage:
    uv run scripts/ih_challenge/grade_compatible.py \
        --base_url https://openrouter.ai/api/v1 \
        --model openai/gpt-4o-mini \
        --input data/raw_datasets/priority_sft/compatible.jsonl

    # Quick test
    uv run scripts/ih_challenge/grade_compatible.py \
        --base_url https://openrouter.ai/api/v1 \
        --model openai/gpt-4o-mini \
        --input data/raw_datasets/priority_sft/compatible.jsonl \
        --max_samples 100
"""

import argparse
import asyncio
import json
import logging
import os

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are a strict quality judge for training data. You will be given:
- INSTRUCTION: the main task
- ADDITIONAL INSTRUCTION: an extra requirement in the prompt
- RESPONSE: the model's answer

Evaluate the response on these criteria:
1. INSTRUCTION_FOLLOWED: Does the response correctly address the main instruction?
2. ADDITIONAL_FOLLOWED: Does the response follow the additional instruction from the prompt?
3. ARITHMETIC: Are all calculations and numbers correct? (Mark OK if no math is involved.)
4. FACTUAL: Are there any factual errors or obvious inaccuracies?

Respond with EXACTLY this JSON format, nothing else:
{"pass": true/false, "reason": "brief explanation if fail, 'ok' if pass"}

Set "pass" to false if ANY criterion fails."""

JUDGE_USER = """\
INSTRUCTION:
{context}

ADDITIONAL INSTRUCTION:
{prompt}

RESPONSE:
{response}"""


def load_completed_ids(output_path: str) -> set[str]:
    completed = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


async def judge_one(
    client: AsyncOpenAI,
    model: str,
    entry: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    user_prompt = JUDGE_USER.format(
        context=entry["context"],
        prompt=entry["prompt"],
        response=entry["response"],
    )

    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                        max_tokens=200,
                    ),
                    timeout=60,
                )
            text = resp.choices[0].message.content.strip()
            # Parse JSON
            result = json.loads(text)
            return {
                "id": entry["id"],
                "judge_pass": result.get("pass", False),
                "judge_reason": result.get("reason", ""),
            }
        except json.JSONDecodeError:
            # Try to extract pass/fail from text
            return {
                "id": entry["id"],
                "judge_pass": False,
                "judge_reason": f"unparseable: {text[:200]}",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                log.warning(f"API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                log.error(f"Failed {entry['id']} after {max_retries} attempts: {e}")
                return {
                    "id": entry["id"],
                    "judge_pass": False,
                    "judge_reason": f"api_error: {e}",
                }


async def run(args):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("No API key. Set --api_key or OPENROUTER_API_KEY env var.")
        return

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Load input
    entries = []
    with open(args.input) as f:
        for line in f:
            entries.append(json.loads(line))
    if args.max_samples > 0:
        entries = entries[:args.max_samples]
    log.info(f"Loaded {len(entries)} samples from {args.input}")

    # Resume
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    completed_ids = load_completed_ids(args.output)
    if completed_ids:
        log.info(f"Resuming: {len(completed_ids)} already graded")

    pending = [e for e in entries if e["id"] not in completed_ids]
    log.info(f"Pending: {len(pending)} samples")

    batch_size = args.max_concurrent * 2
    total_pass = 0
    total_fail = 0

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]

        results = await asyncio.gather(*[
            judge_one(client, args.model, e, semaphore) for e in batch
        ])

        with open(args.output, "a") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if r["judge_pass"]:
                    total_pass += 1
                else:
                    total_fail += 1

        done = batch_start + len(batch)
        total = total_pass + total_fail
        rate = total_pass / total * 100 if total > 0 else 0
        log.info(f"Progress: {done}/{len(pending)} | pass: {total_pass}/{total} ({rate:.1f}%)")

    log.info(f"Done. Pass: {total_pass}, Fail: {total_fail}")


def main():
    parser = argparse.ArgumentParser(description="Grade compatible mode data with LLM judge")
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--max_concurrent", type=int, default=20)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.replace(".jsonl", "_grades.jsonl")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
