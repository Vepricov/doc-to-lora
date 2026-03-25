"""
SFT training script for D2L instruction-following.

Fine-tunes the D2L hypernetwork on (instruction, query, gold_response) triplets.
Teaches the hypernetwork to generate LoRA weights that encode behavioral
instructions (not just knowledge).

Supports multi-GPU via accelerate.

Usage:
    # Multi-GPU (4 GPUs)
    uv run accelerate launch --config_file accelerate_config.yaml \
        scripts/ih_challenge/train_sft.py \
        --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --data_path data/raw_datasets/alpaca/alpaca_combined.jsonl \
        --output_dir train_outputs/sft_runs/qwen_4b_alpaca

    # Single GPU smoke test
    uv run scripts/ih_challenge/train_sft.py \
        --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --data_path data/raw_datasets/ih_challenge/qwen_test/defenses_static.jsonl \
        --output_dir train_outputs/sft_smoke_test \
        --num_steps 5 --max_samples 10
"""

import argparse
import glob
import logging
import os
import random
import re
import shutil
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Add project root and scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.dirname(__file__))

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer

from sft_dataset import IHChallengeSFTDataset
from experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training for D2L instruction-following")

    # Model
    parser.add_argument("--checkpoint_path", required=True, help="D2L checkpoint to fine-tune")

    # Data
    parser.add_argument("--data_path", required=True, help="Path to SFT jsonl (context, prompt, response)")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated modes to include (e.g. conflict,knowledge,neutral). Default: all")

    # Training
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=0, help="Override num_epochs if > 0")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_response_tokens", type=int, default=512)

    # Output
    parser.add_argument("--output_dir", default=None, help="Auto-generated with timestamp if not set")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--logger", choices=["comet", "wandb", "none"], default="comet")
    parser.add_argument("--project_name", default="d2l-sft")

    # Checkpoint management
    parser.add_argument("--keep_last_n", type=int, default=1,
                        help="Keep only last N checkpoints (0 = keep all)")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def save_checkpoint(model, optimizer, step, output_dir, keep_last_n=1):
    """Save hypernet checkpoint and optionally remove old ones."""
    path = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    state = model.state_dict()
    torch.save(state, os.path.join(path, "pytorch_model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
    log.info(f"Saved checkpoint to {path}")

    # Cleanup old checkpoints
    if keep_last_n > 0:
        ckpt_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda d: int(re.search(r"checkpoint-(\d+)$", d).group(1)),
        )
        for old in ckpt_dirs[:-keep_last_n]:
            shutil.rmtree(old)
            log.info(f"Removed old checkpoint: {old}")


def compute_ce_loss(logits, labels):
    """Compute cross-entropy loss on response tokens only (labels=-100 for prompt)."""
    # Upcast to float32 for precision (model outputs bf16)
    logits = logits.float()
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


def main():
    args = parse_args()

    # Auto-generate output_dir with timestamp if not specified
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("train_outputs", f"sft_{ts}")

    # Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )
    is_main = accelerator.is_main_process

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Logger (only on main process)
    if is_main:
        exp_logger = ExperimentLogger(backend=args.logger, project=args.project_name, config=vars(args))
    else:
        exp_logger = ExperimentLogger(backend="none", project=args.project_name)

    # 1. Load D2L model
    if is_main:
        log.info(f"Loading checkpoint: {args.checkpoint_path}")
    # Build args.yaml: start from source checkpoint's args (needed by run_eval.py
    # for model config), then add SFT-specific fields so we know what data was used.
    os.makedirs(args.output_dir, exist_ok=True)
    src_args_yaml = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint_path)), "args.yaml")
    dst_args_yaml = os.path.join(args.output_dir, "args.yaml")
    if is_main and not os.path.exists(dst_args_yaml):
        import yaml
        base_args = {}
        if os.path.exists(src_args_yaml):
            with open(src_args_yaml) as f:
                try:
                    base_args = yaml.safe_load(f) or {}
                except yaml.constructor.ConstructorError:
                    f.seek(0)
                    base_args = yaml.unsafe_load(f) or {}
                    base_args = {k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                                 for k, v in base_args.items()}
        # Add SFT-specific fields
        base_args["sft_checkpoint_path"] = args.checkpoint_path
        base_args["sft_data_path"] = args.data_path
        base_args["sft_max_samples"] = args.max_samples
        base_args["sft_num_epochs"] = args.num_epochs
        base_args["sft_num_steps"] = args.num_steps
        base_args["sft_lr"] = args.lr
        base_args["sft_warmup_steps"] = args.warmup_steps
        base_args["sft_gradient_accumulation_steps"] = args.gradient_accumulation_steps
        base_args["sft_max_response_tokens"] = args.max_response_tokens
        base_args["sft_keep_last_n"] = args.keep_last_n
        with open(dst_args_yaml, "w") as f:
            yaml.dump(base_args, f, default_flow_style=False, sort_keys=True)
        log.info(f"Saved args.yaml with SFT config to {dst_args_yaml}")

    state_dict = torch.load(args.checkpoint_path, weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=True, use_flash_attn=True, use_sequence_packing=False,
    )
    model = model.to(accelerator.device).to(torch.bfloat16)

    # Freeze base_model and ctx_encoder
    for p in model.base_model.parameters():
        p.requires_grad = False
    for p in model.ctx_encoder.parameters():
        p.requires_grad = False

    # Only train hypernet
    model.hypernet.train()
    model.base_model.eval()
    model.ctx_encoder.eval()

    trainable_params = [p for p in model.hypernet.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in trainable_params)
    if is_main:
        log.info(f"Trainable parameters: {num_params:,} (hypernet only)")

    # 2. Tokenizers
    base_model_name = model.base_model.config.name_or_path
    ctx_encoder_name = getattr(
        state_dict.get("ctx_encoder_args"), "ctx_encoder_model_name_or_path", None
    ) or base_model_name

    tokenizer = get_tokenizer(base_model_name, train=False)
    ctx_tokenizer = get_tokenizer(ctx_encoder_name, train=False)

    # 3. Dataset
    modes = args.modes.split(",") if args.modes else None
    dataset = IHChallengeSFTDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        ctx_tokenizer=ctx_tokenizer,
        max_samples=args.max_samples,
        max_response_tokens=args.max_response_tokens,
        modes=modes,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Compute total steps from epochs (or use explicit --num_steps override)
    # With multi-GPU, accelerate splits dataloader across GPUs,
    # so each GPU sees len(dataset)/num_gpus samples per epoch.
    steps_per_epoch = len(dataset) // max(accelerator.num_processes, 1)
    num_steps = args.num_steps if args.num_steps > 0 else steps_per_epoch * args.num_epochs
    if is_main:
        log.info(
            f"Dataset: {len(dataset)} samples, {steps_per_epoch} steps/epoch/gpu, "
            f"{num_steps} total steps ({args.num_epochs} epochs)"
        )

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    # Scheduler counts optimizer steps, not micro-steps.
    # accelerator.prepare() wraps scheduler to only step on real optimizer updates.
    num_optimizer_steps = num_steps // args.gradient_accumulation_steps
    warmup_optimizer_steps = args.warmup_steps // args.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_optimizer_steps, num_optimizer_steps,
    )
    if is_main:
        log.info(f"Scheduler: {warmup_optimizer_steps} warmup, {num_optimizer_steps} total optimizer steps")

    # 5. Prepare with accelerate (handles DDP, device placement, grad accumulation)
    # NOTE: scheduler NOT prepared — we step it manually on real optimizer steps only
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader,
    )

    # 6. Training loop
    if is_main:
        log.info(f"Starting SFT training: {num_steps} steps, {accelerator.num_processes} GPUs")
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    data_iter = iter(dataloader)
    running_loss = 0.0

    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Squeeze DataLoader batch dim (batch_size=1)
        ctx_ids = batch["ctx_ids"].squeeze(0)
        ctx_attn_mask = batch["ctx_attn_mask"].squeeze(0)
        input_ids = batch["input_ids"].squeeze(0)
        attention_mask = batch["attention_mask"].squeeze(0)
        labels = batch["labels"].squeeze(0)

        with accelerator.accumulate(model):
            # Forward pass through full D2L pipeline
            outputs = model(
                ctx_ids=ctx_ids,
                ctx_attn_mask=ctx_attn_mask,
                n_ctx_chunks=torch.tensor([ctx_ids.shape[0]], device=accelerator.device),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # CE loss on response tokens
            loss = compute_ce_loss(outputs.logits, labels)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        # Step scheduler only on real optimizer steps (not every micro-step)
        if accelerator.sync_gradients:
            scheduler.step()

        running_loss += loss.item()

        # Logging (main process only)
        if (step + 1) % args.log_steps == 0 and is_main:
            avg_loss = running_loss / args.log_steps
            running_loss = 0.0
            log.info(
                f"Step {step + 1}/{num_steps} | "
                f"loss={avg_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
            exp_logger.log({
                "train/loss": avg_loss,
                "train/lr": scheduler.get_last_lr()[0],
            }, step=step + 1)

        # Checkpoint (main process only)
        if (step + 1) % args.save_steps == 0 and is_main:
            unwrapped = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped, optimizer, step + 1, args.output_dir, args.keep_last_n)

        step += 1

    # Final save
    if is_main:
        unwrapped = accelerator.unwrap_model(model)
        save_checkpoint(unwrapped, optimizer, step, args.output_dir, args.keep_last_n)
        log.info("SFT training complete.")
        exp_logger.finish()


if __name__ == "__main__":
    main()
