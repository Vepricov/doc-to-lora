"""
RL training script for D2L adversarial robustness.

Trains the D2L hypernetwork using GRPO/REINFORCE on IH-Challenge tasks.
Online attacker (LLM via API) generates adversarial attacks via propose-evaluate-revise.
Python graders provide binary reward signal.

Usage:
    uv run scripts/ih_challenge/train_rl.py \
        --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --attacker_base_url https://openrouter.ai/api/v1 \
        --attacker_model openai/gpt-4o-mini \
        --output_dir train_outputs/rl_runs/qwen_4b_ih
"""

import argparse
import asyncio
import logging
import os
import random
import sys

import torch
from openai import AsyncOpenAI
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Add project root and scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer

from rl_dataset import IHChallengeRLDataset
from rl_trainer import RLTrainer
from experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="RL training for D2L adversarial robustness")

    # Model
    parser.add_argument("--checkpoint_path", required=True, help="D2L checkpoint")

    # Data
    parser.add_argument("--data_dir", default="data/benchmarks/ih-challenge")
    parser.add_argument("--splits", nargs="+", default=["single-constraint"])
    parser.add_argument("--task_types", nargs="+", default=None)
    parser.add_argument("--max_per_type", type=int, default=0)

    # Attacker
    parser.add_argument("--attacker_base_url", required=True)
    parser.add_argument("--attacker_model", default="openai/gpt-4o-mini")
    parser.add_argument("--attacker_api_key", default=None)
    parser.add_argument("--max_rounds", type=int, default=3)

    # RL
    parser.add_argument("--rl_method", choices=["grpo", "reinforce"], default="grpo")
    parser.add_argument("--num_responses", type=int, default=4, help="K responses per sample")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Training
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--kl_coef", type=float, default=0.0, help="KL preservation weight (0 = disabled)")

    # Output
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logger", choices=["comet", "wandb", "none"], default="comet")
    parser.add_argument("--project_name", default="d2l-rl")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def save_checkpoint(model, optimizer, step, output_dir):
    """Save hypernet checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)

    state = model.state_dict()
    torch.save(state, os.path.join(path, "pytorch_model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    log.info(f"Saved checkpoint to {path}")


async def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Logger
    exp_logger = ExperimentLogger(backend=args.logger, project=args.project_name, config=vars(args))

    # 1. Load D2L model from checkpoint
    log.info(f"Loading checkpoint: {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=True, use_flash_attn=True, use_sequence_packing=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).to(torch.bfloat16)

    # Ensure base_model and ctx_encoder are frozen
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.hypernet.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    log.info(f"Trainable parameters: {num_params:,} (hypernet only)")

    # 2. Tokenizers
    base_model_name = model.base_model.config.name_or_path
    ctx_encoder_name = state_dict.get("ctx_encoder_args", argparse.Namespace(
        ctx_encoder_model_name_or_path=None,
    )).ctx_encoder_model_name_or_path or base_model_name

    tokenizer = get_tokenizer(base_model_name, train=False)
    ctx_tokenizer = get_tokenizer(ctx_encoder_name, train=False)

    # 3. Dataset
    dataset = IHChallengeRLDataset(
        data_dir=args.data_dir,
        splits=args.splits,
        task_types=args.task_types,
        max_per_type=args.max_per_type,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        collate_fn=IHChallengeRLDataset.collate_fn,
    )
    log.info(f"Dataset: {len(dataset)} task skeletons")

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, args.num_steps,
    )

    # 5. Attacker client
    api_key = args.attacker_api_key or os.environ.get("OPENROUTER_API_KEY") or "x"
    attacker_client = AsyncOpenAI(base_url=args.attacker_base_url, api_key=api_key)

    # 6. RL Trainer
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        ctx_tokenizer=ctx_tokenizer,
        rl_method=args.rl_method,
        num_responses=args.num_responses,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        clip_eps=args.clip_eps,
        attacker_client=attacker_client,
        attacker_model=args.attacker_model,
        max_rounds=args.max_rounds,
    )

    # 7. Training loop
    log.info(f"Starting RL training: {args.num_steps} steps, method={args.rl_method}")
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    data_iter = iter(dataloader)
    optimizer.zero_grad()

    while step < args.num_steps:
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Attacker rollout for each task in batch
        tasks_with_attacks = []
        for task in batch:
            rollout = await trainer.attacker_rollout(task)
            task_with_attack = {**task, "attack": rollout["attack"]}
            tasks_with_attacks.append(task_with_attack)

            # Log rollout info
            n_rounds = len(rollout["traces"])
            attack_succeeded = any(not t["defender_passed"] for t in rollout["traces"])
            log.debug(
                f"  Rollout: {task['task_type']} | rounds={n_rounds} | "
                f"attack_succeeded={attack_succeeded}"
            )

        # GRPO/REINFORCE update
        model.hypernet.train()
        model.base_model.eval()
        model.ctx_encoder.eval()
        metrics = trainer.train_step(tasks_with_attacks)
        loss = metrics["loss"]

        # Accumulate gradients
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Logging
        if step % args.log_steps == 0:
            log.info(
                f"Step {step}/{args.num_steps} | "
                f"loss={metrics['rl_loss']:.4f} | "
                f"reward={metrics['mean_reward']:.3f} | "
                f"pass_rate={metrics['pass_rate']:.1%} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            exp_logger.log({
                "train/loss": metrics["rl_loss"],
                "train/mean_reward": metrics["mean_reward"],
                "train/pass_rate": metrics["pass_rate"],
                "train/reward_std": metrics["reward_std"],
                "train/lr": scheduler.get_last_lr()[0],
            }, step=step)

            # Log generations for debugging (terminal + experiment tracker)
            for sample in metrics.get("samples", []):
                lines = [
                    f"\n{'='*80}",
                    f"[{sample['task_type']}]",
                    f"Instruction: {sample['instruction']}",
                    f"Attack: {sample['attack']}",
                ]
                for i, resp in enumerate(sample["responses"]):
                    reward_str = "PASS" if resp["reward"] > 0.5 else "FAIL"
                    lines.append(f"--- Response {i} [{reward_str}] ---")
                    lines.append(resp["text"])
                lines.append(f"{'='*80}")
                text_block = "\n".join(lines)
                log.info(text_block)
                exp_logger.log_text(
                    text_block,
                    step=step,
                    metadata={"task_type": sample["task_type"]},
                )

        # Save checkpoint
        if (step + 1) % args.save_steps == 0:
            save_checkpoint(model, optimizer, step + 1, args.output_dir)

        step += 1

    # Final save
    save_checkpoint(model, optimizer, step, args.output_dir)
    log.info("Training complete.")
    exp_logger.finish()


if __name__ == "__main__":
    asyncio.run(main())
