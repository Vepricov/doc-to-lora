"""
Run IH-Challenge graders on D2L generated responses.

Takes:
  1. Generated text jsonl from D2L eval (has 'input', 'generated', 'context')
  2. Original IH-Challenge dataset jsonl (has 'metadata.grader_code', 'metadata.task_type', 'prompt')

Outputs per-sample pass/fail and aggregated stats by task_type.

Usage:
    uv run scripts/ih_challenge/run_grader.py \
        --generated <path_to_generated_text.jsonl> \
        --dataset <path_to_ih_dataset.jsonl>
"""

import argparse
import json
import sys
from collections import defaultdict


def load_grader(grader_code: str):
    """Compile grader code and return the grade_output_correct function."""
    namespace = {}
    exec(grader_code, namespace)
    return namespace["grade_output_correct"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True,
                        help="Path to D2L generated_text.jsonl")
    parser.add_argument("--dataset", required=True,
                        help="Path to IH-Challenge dataset jsonl (with grader codes)")
    parser.add_argument("--output", default=None,
                        help="Path to write detailed results jsonl (optional)")
    args = parser.parse_args()

    # Load generated responses
    generated = []
    with open(args.generated) as f:
        for line in f:
            generated.append(json.loads(line))

    # Load dataset (graders + metadata)
    dataset = []
    with open(args.dataset) as f:
        for line in f:
            dataset.append(json.loads(line))

    # Match generated samples to dataset entries.
    # Generated 'context' field contains chat-template-wrapped version of ds 'context'.
    # ds["context"] is a substring of gen["context"], so we match by checking inclusion.
    # Similarly, ds["prompt"] is a substring of gen["input"].
    # We require BOTH to match to avoid ambiguity.

    ds_used = set()
    results = []
    stats = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0})
    unmatched = 0

    for gi, gen in enumerate(generated):
        gen_input = gen["input"]
        gen_context = gen.get("context", "")

        # Find matching dataset entry: ds context must be substring of gen context,
        # AND ds prompt must be substring of gen input.
        ds_idx = None
        for di, ds_candidate in enumerate(dataset):
            if di in ds_used:
                continue
            if (ds_candidate["context"].strip() in gen_context
                    and ds_candidate["prompt"].strip() in gen_input):
                ds_idx = di
                break

        if ds_idx is not None:
            ds_used.add(ds_idx)
        else:
            unmatched += 1
            print(f"  [{gi}] Could not match to dataset entry, skipping", file=sys.stderr)
            continue

        ds = dataset[ds_idx]
        meta = ds["metadata"]
        task_type = meta["task_type"]
        grader_code = meta["grader_code"]

        input_text = ds["prompt"]
        assistant_response = gen["generated"]

        try:
            grader = load_grader(grader_code)
            passed = grader(input_text, assistant_response)
            status = "pass" if passed else "fail"
        except Exception as e:
            status = "error"
            print(f"  [{gi}] Grader error for {task_type}: {e}", file=sys.stderr)

        stats[task_type][status] += 1
        results.append({
            "idx": gi,
            "ds_idx": ds_idx,
            "task_type": task_type,
            "status": status,
            "prompt": input_text,
            "context": ds["context"][:200],
            "generated": assistant_response[:300],
            "privileged_level": meta["privileged_level"],
            "attack_level": meta["attack_level"],
        })

    # Print results table
    print(f"\n{'task_type':<25} {'pass':>5} {'fail':>5} {'error':>5} {'pass_rate':>10}")
    print("-" * 55)

    total_pass = 0
    total_fail = 0
    total_error = 0

    for task_type in sorted(stats.keys()):
        s = stats[task_type]
        total = s["pass"] + s["fail"] + s["error"]
        rate = s["pass"] / total if total > 0 else 0
        print(f"{task_type:<25} {s['pass']:>5} {s['fail']:>5} {s['error']:>5} {rate:>9.1%}")
        total_pass += s["pass"]
        total_fail += s["fail"]
        total_error += s["error"]

    total = total_pass + total_fail + total_error
    overall_rate = total_pass / total if total > 0 else 0
    print("-" * 55)
    print(f"{'TOTAL':<25} {total_pass:>5} {total_fail:>5} {total_error:>5} {overall_rate:>9.1%}")

    if unmatched:
        print(f"\nWARNING: {unmatched} generated samples could not be matched to dataset")

    # Write detailed results
    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
