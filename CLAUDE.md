# Doc-to-LoRA (D2L)

## Project

Sakana AI repo. A hypernetwork reads a document in a single forward pass and generates LoRA weights that modify the LLM — instead of keeping the document in the context window.

## Environment

- Server: `hardcore-wright-0`
- Python: uv-managed 3.10.20 (`/home/jovyan/research/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/`)
- venv: `.venv/` (created via uv with managed Python so that `Python.h` is available for Triton/DeepSpeed)
- CUDA: 12.1 (nvcc) at `/home/jovyan/davydenko/cuda-12.1`, torch built for cu124 (ships its own libs)
- uv: `/home/jovyan/.local/bin/uv` (v0.10.9)

## Key dependencies

- torch 2.6.0+cu124
- flash-attn 2.7.4.post1 (prebuilt wheel)
- flashinfer 0.2.2+cu124torch2.6
- transformers 4.51.3, deepspeed 0.17.1, peft 0.15.2, vllm 0.8.5.post1

## Repo structure

- `train.py` — training entry point (accelerate launch, 8 GPUs)
- `run_eval.py` — evaluation entry point (generative eval)
- `src/ctx_to_lora/` — main package
  - `modeling/hypernet.py` — core: HyperLoRA, ModulatedPretrainedModel
  - `modeling/lora_merger.py` — LoRA chunk merging (stack or average)
  - `modeling/aggregator.py` — Perceiver aggregator
  - `modeling/ctx_encoder.py` — context encoder
  - `modeling/lora_layer.py` — custom LoRA forward pass
  - `data/` — data processing, packing
  - `configs.py` — all dataclass arguments
  - `eval_utils.py` — evaluation pipeline
- `configs/` — YAML experiment configs
- `scripts/main_exp/` — main experiment (train/eval)
- `scripts/niah/` — needle-in-a-haystack
- `scripts/ih_challenge/` — IH-Challenge eval pipeline
  - `prepare_ih_dataset.py` — converts IH-Challenge to D2L format (probing queries)
  - `run_grader.py` — runs Python graders on D2L generated text, outputs per-task-type stats
  - `assign_ids.py` — assigns unique IDs to IH-Challenge entries across all splits
  - `generate_attacks.py` — LLM-based attack generation (3 difficulty levels, async, resumable)
  - `generate_defenses.py` — generates gold defense responses for SFT training data
  - `generate_compatible.py` — generates Mode 2 (compatible) data: additional constraint (A) or second task (B) via LLM
  - `prepare_alpaca_sft.py` — prepares Alpaca dataset for D2L SFT (with_input / no_input / combined)
  - `train_sft.py` — SFT training (CE loss on gold responses, multi-GPU via accelerate)
  - `train_rl.py` — RL training (REINFORCE with group-normalized advantages, online attacker)
  - `rl_trainer.py` — RLTrainer class (attacker rollout + GRPO/REINFORCE train_step)
  - `rl_dataset.py` — IH-Challenge task skeleton loader for RL
  - `sft_dataset.py` — SFT dataset loader (instruction → LoRA, query → input, response → target)
  - `eval_sft.py` — generative eval of SFT checkpoints on dev set (saves generations + ROUGE-L)
  - `experiment_logger.py` — unified logger (Comet ML / W&B / none)
- `demo/app.py` — Gradio interface
- `data/` — data prep scripts + raw_datasets
  - `benchmarks/ih-challenge/` — IH-Challenge dataset (4 splits, 27k+ task skeletons)
  - `benchmarks/ConInstruct/` — ConInstruct dataset (100 instructions + conflicts)
  - `raw_datasets/ih_challenge/` — D2L-formatted eval data (baseline.jsonl, attacks_static.jsonl, defenses_static.jsonl, defenses_graded.jsonl)
  - `raw_datasets/alpaca/` — Alpaca SFT data (alpaca_with_input.jsonl, alpaca_no_input.jsonl, alpaca_combined.jsonl, dev.jsonl)
  - `raw_datasets/priority_sft/` — Priority SFT v2 dataset (train_v2.jsonl, dev_v2.jsonl, conflict_train.jsonl, conflict_dev.jsonl, conflict_composite.jsonl, compatible.jsonl, compatible_filtered.jsonl)
- `results/` — eval results and dataset stats
  - `ih_challenge_baseline_all_models.md` — IH-Challenge baseline eval (D2L vs base models, 95 probing queries)
  - `sft_alpaca_eval.md` — Alpaca SFT eval (ROUGE-L, F1, paired examples)
  - `qa_benchmark_comparison.md` — QA benchmark (D2L vs RAG vs no-context, 3 datasets × 1500 samples)
  - `defense_grader_failures.md` — defense v1 grader failures (4 examples per 21 task types)
  - `priority_sft_dataset_stats.md` — Priority SFT v2 dataset composition and statistics
- `papers/` — PDFs of related papers (D2L, pRAG, IH-Challenge, ConInstruct)

### External directories

- `/home/jovyan/research/` — shared research storage (datasets backups, shared tools)
  - `datasets/skillsbench/` — SkillsBench repo (87 tasks, 3GB, with skills and pytest verifiers)
  - `datasets/d2l/ih_challenge/` — IH-Challenge data backups
- `/home/jovyan/davydenko/compression_experiments/` — unified eval framework for D2L, RAG, baselines (LLM-as-judge, multiple methods/datasets). Contains `Doc2LoRA` wrapper, eval configs, results.

## Pretrained checkpoints

In `trained_d2l/`:
- `gemma_2b_d2l/checkpoint-20000/`
- `gemma_demo/checkpoint-80000/`
- `mistral_7b_d2l/checkpoint-20000/`
- `qwen_4b_d2l/checkpoint-20000/`

## In-repo documentation

- `paper.md` — critical review of the D2L paper: math (KL-loss, Perceiver, chunking), figures analysis, strengths/weaknesses
- `code_overview.md` — detailed code walkthrough with file references and formulas

## Architecture (key details from code_overview)

- **ctx_encoder**: uses `PerLayerActivations` (main mode) — activations from all layers up to `ctx_encoder_last_layer`, output `[bs, n_layers, seq_len, hidden]`
- **Perceiver**: encoder (num_blocks cross-attn) -> decoder (1 cross-attn), `layer_to_layer=True` — each LLM layer processed independently via rearrange `(n_layers bs) seq_len d`
- **HyperLoRA head**: `EinMix` with `n_layers` axis — layer-specific weights. Output split into A and B, multiplied by `scaler_A`/`scaler_B` (B initialized to zero -> LoRA starts from zero)
- **lora_alpha**: set automatically as `r^(3/2) * 2` in `model_loading.py`
- **LoRA forward**: `lora_forward` (normal) / `lora_forward_packed` (sequence packing) in `lora_layer.py`. Patched via `partial` in `apply_lora_to_layers`
- **Trainer**: `DistillationTrainer` (KL on top-K teacher tokens) or `CrossEntropyTrainer`. `latents_q` excluded from weight decay
- **Data**: chunking at word boundaries, chunk count augmentation via `num_chunk_probs`, sequence packing (greedy algorithm)

## Fixes

- Added `avg_chunk_loras=False` fallback in `hypernet.py:from_state_dict()` for backward compatibility with old checkpoints (~line 497)

## Running

### Generate NIAH data
```bash
uv run data/generate_ctx_magic_number.py
```

### Eval (NIAH example)
```bash
WANDB_MODE=disabled uv run run_eval.py \
  --checkpoint_path trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --datasets ctx_magic_number_32_1024 ctx_magic_number_1024_2048 \
  --max_ctx_chunk_len=1024 --split test --eval_batch_size_gen=4
```

### Eval (QA example)
```bash
WANDB_MODE=disabled uv run run_eval.py \
  --checkpoint_path trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
  --datasets squad --split test --max_test_samples_per_ds 50 --eval_batch_size_gen 1
```

### SFT training (instruction-following)
```bash
# Multi-GPU (4 GPUs), 1 epoch on Alpaca combined (20k samples)
uv run accelerate launch --config_file accelerate_config.yaml \
    scripts/ih_challenge/train_sft.py \
    --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
    --data_path data/raw_datasets/alpaca/alpaca_combined.jsonl \
    --output_dir train_outputs/sft_alpaca_qwen4b \
    --logger comet --project_name d2l-sft

# Single GPU, custom steps
uv run scripts/ih_challenge/train_sft.py \
    --checkpoint_path trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
    --data_path data/raw_datasets/alpaca/alpaca_combined.jsonl \
    --output_dir train_outputs/sft_alpaca_qwen4b \
    --num_steps 500 --logger none
```

### RL training (adversarial robustness)
```bash
# Requires attacker API (OpenRouter or local vLLM)
uv run scripts/ih_challenge/train_rl.py \
    --checkpoint_path train_outputs/sft_alpaca_qwen4b/checkpoint-5000/pytorch_model.bin \
    --attacker_base_url https://openrouter.ai/api/v1 \
    --attacker_model openai/gpt-4o-mini \
    --output_dir train_outputs/rl_runs/qwen_4b_ih \
    --num_steps 2000 --num_responses 4 --max_rounds 3 \
    --logger comet --project_name d2l-rl
```

### Eval (IH-Challenge)
```bash
WANDB_MODE=disabled uv run run_eval.py \
    --checkpoint_path train_outputs/sft_alpaca_qwen4b/checkpoint-5000/pytorch_model.bin \
    --datasets ih_challenge_baseline \
    --split test --eval_batch_size_gen=1

uv run scripts/ih_challenge/run_grader.py \
    --generated <eval_output_dir>/test_ih_challenge_baseline_generated_text.jsonl \
    --dataset data/raw_datasets/ih_challenge/baseline.jsonl
```

### Demo
```bash
uv run demo/app.py
```

## Work direction

Two research threads around parametric knowledge injection (D2L and pRAG):
1. **Adversarial robustness** — instruction hierarchy, LoRA-vs-context prioritization (Experiments 1-3)
2. **Skill-following** — encoding complex procedural instructions into LoRA, skill retrieval (Experiments 4-5)

Both architectures share the same LoRA-vs-context split: instructions are encoded into LoRA weights, adversarial/conflicting input comes through the context window. The difference is how LoRA is produced:
- **D2L**: hypernetwork generates LoRA in a single forward pass (fast inference, trains the hypernetwork)
- **pRAG**: separate LoRA trained per document via SFT on augmented data (slow offline, trains the LoRA directly)

### Research plan

**Experiment 1 — IH-Challenge baseline eval (done):**
- Evaluated D2L and base models on IH-Challenge single-constraint (19 task types, probing queries)
- Key finding: base model with in-context instruction >> D2L LoRA (Gemma 74.7% vs 40.0%, Qwen 81.1% vs 55.8%)
- D2L encodes LoRA content as knowledge (leaks secrets) but doesn't follow behavioral instructions
- Results: `results/ih_challenge_baseline_all_models.md`

**Experiment 2 — Attack augmentation + RL training:**
- Two separate pipelines:
  - **Static generation** (`generate_attacks.py`): generate N=3 attacks per task skeleton at 3 difficulty levels (L1 direct, L2 tricky, L3 sophisticated) using external LLM
  - **RL training** (next step, separate pipeline): IH-Challenge-style propose-evaluate-revise loop with online defender
- Attacker model comparison (smoke test, 114 samples on single-constraint, `--max_per_type 2`):
  - **GPT-OSS-120B via local vLLM**: 81% refusals. Too safety-aligned for red-teaming. Unusable.
  - **GPT-4o-mini via OpenRouter**: 0% refusals, good quality. Cost ~$10 for full 82k run.
  - **Qwen2.5-72B-Instruct via local vLLM**: 0% refusals (1 borderline case = valid attack), best attack quality — more aggressive L1, concrete social engineering L2, advanced L3 (base64 encoding, fake system messages). Free (local).
  - **Decision: use Qwen2.5-72B-Instruct via local vLLM for static generation.** GPT-4o-mini as fallback.
- Reuse IH-Challenge Python graders (verify matching quality)

**Experiment 3 — Constraint prioritization (ConInstruct-style):**
- LoRA ← one set of output constraints (e.g., "don't use word X")
- Context ← conflicting constraint (e.g., "start sentence with phrase containing X")
- Goal: model should prioritize LoRA-channel over context-channel
- Closer to pure instruction-following than adversarial attacks
- Uses ConInstruct conflict types (CC, KK, PP, LL, FF, SS, KP, PC, PS) to systematically test prioritization
- Applies to both D2L and pRAG identically (only LoRA production differs)
- Future: extend with different RL strategies for constraint prioritization training (TBD)

**Experiment 4 — Skill-following via LoRA:**
- New direction: encode complex procedural instructions (skills) into LoRA instead of simple constraints
- LoRA ← skill description (SKILL.md: workflows, SOPs, coding conventions, domain procedures)
- Context ← task instruction (specific problem to solve using the skill)
- Key difference from Experiments 1-3: skills are much longer and more complex than atomic constraints
- Hypothesis: D2L hypernetwork trained on QA/simple instructions won't generalize to skills → need skill-specific SFT
- **Eval: SkillsBench** ([skillsbench.ai](https://www.skillsbench.ai)) — 84 tasks, 11 domains, pytest verifiers (deterministic pass/fail). Paired evaluation: no-skill vs with-skill. Paper: "SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks" (Feb 2026, BenchFlow). GitHub: [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench).
- **Training data sources:** SkillsBench skills (47k), Anthropic skills repo ([anthropics/skills](https://github.com/anthropics/skills)), any structured instructions (SOPs, guidelines)
- **Pipeline:** baseline eval on SkillsBench (D2L with skill→LoRA) → skill-specific SFT → re-eval
- Applies to both D2L (hypernetwork → LoRA) and pRAG (per-skill SFT → LoRA) identically

**Experiment 5 — Skill RAG (retrieval over skill repositories):**
- Separate from parametric injection (D2L/pRAG) but complementary — D2L can use this mechanism, not the other way around
- **Problem:** current paradigm is one task → one manually matched skill. Want to move to: large shared skill corpus → automatic retrieval → solve diverse tasks
- **Vision:** build a searchable repository of skills from multiple sources (SkillsBench 47k, [anthropics/skills](https://github.com/anthropics/skills), OpenAI skills, custom). Given an arbitrary task, retrieve relevant skills and apply them.
- **Challenges:**
  - Skill indexing: how to search? Embedding similarity on SKILL.md, category/domain taxonomy, or hybrid?
  - Skill composition: SkillsBench shows 2-3 skills optimal, 4+ hurts. Need ranking/selection, not just top-K
  - Evaluation: need benchmark where tasks aren't pre-matched with skills. Likely synthetic task generation across categories
- **First steps:**
  1. Aggregate skill corpus from available sources
  2. Generate/collect diverse task corpus (synthetic, across categories)
  3. Baseline: embedding retrieval → top-K skills in-context → solve → eval with deterministic verifiers
  4. Compare: in-context skills vs D2L (skill → LoRA) vs pRAG (skill → trained LoRA)
- **Connection to prompt-following:** skill-following is a form of complex instruction-following. Insights from adversarial robustness (Experiments 1-3) and skill-following (Experiment 4) feed into this

### Skill-related resources

- **SkillsBench** — [skillsbench.ai](https://www.skillsbench.ai), [GitHub](https://github.com/benchflow-ai/skillsbench). 84 tasks, 11 domains, 47k skills, pytest verifiers. Paper: "SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks" (Feb 2026, BenchFlow).
- **Anthropic Skills** — [github.com/anthropics/skills](https://github.com/anthropics/skills). Curated skill repository for Claude Code.
- **OpenAI Skills** — [github.com/openai/skills](https://github.com/openai/skills). Skill repository for OpenAI agents.
- **Memento-Skills** — [github.com/Memento-Teams/Memento-Skills](https://github.com/Memento-Teams/Memento-Skills), [arxiv 2603.18743](https://arxiv.org/abs/2603.18743). Agent-designing agent: autonomously constructs, adapts, and improves task-specific agents through experience. Memory-based RL framework with stateful prompts, reusable skills stored as structured markdown files serve as persistent, evolving memory.

## Related papers

PDFs in `papers/`:

- **Doc-to-LoRA** (`papers/doc-to-lora.pdf`) — [arxiv 2602.15902](https://arxiv.org/abs/2602.15902). Sakana AI. This repo's paper. Hypernetwork (ctx_encoder + Perceiver + HyperLoRA head) reads a document in one forward pass and generates LoRA weights for the LLM. Trained via KL-distillation from teacher (LLM with document in context). Eliminates need for per-document training or long context at inference.
- **Parametric RAG** (`papers/parametric-rag.pdf`) — [arxiv 2501.15915](https://arxiv.org/abs/2501.15915), [code](https://github.com/oneal2000/PRAG). Tsinghua. Injects documents into FFN parameters as LoRA adapters instead of context window. Offline pipeline: (1) Document Augmentation — LLM rewrites document n times + generates m QA pairs → cross-product of (rewrite, Q, A) triplets; (2) Parametric Document Encoding — trains randomly-initialized LoRA (FFN only, W'=W+AB^T) per document on augmented data with standard LM objective. Online: Retrieve-Update-Generate (RUG) — retrieve top-k docs → average-merge their LoRAs → update LLM → generate from query alone (no docs in context). Can combine with in-context RAG for further gains. Key difference from D2L: separate LoRA trained per document (slow offline, no hypernetwork) vs D2L's single forward pass. Models: LLaMA-3.2-1B, Qwen2.5-1.5B, LLaMA-3.1-8B. Benchmarks: 2WikiMultihopQA, HotpotQA, PopQA, ComplexWebQuestions.

### Attack-related papers

- **IH-Challenge** (`papers/ih-challenge.pdf`) — [arxiv 2603.10521](https://arxiv.org/abs/2603.10521), [dataset](https://huggingface.co/datasets/openai/ih-challenge). OpenAI. RL training dataset for instruction hierarchy (IH) robustness. IH policy: system ≻ developer ≻ user ≻ tool (lower-priority instructions honored only when compatible with higher-priority). Task skeletons constructed offline: high-priority instructions + placeholder for attack + Python grader as verifiable reward. Online: frozen attacker LLM fills the placeholder via budgeted propose-evaluate-revise loop with tool access to current defender. 4 task splits: Single-Constraint (IFEval-like: contain-word, hide-PIN, JSON-only, etc.), Multi-Constraint (composition of 2–6 atomic), Input-Conditioned (closed-ended: parse pattern → output in schema), Anti-Overrefusal (benign requests rewritten to look forbidden; refusal = failure). GPT-5-Mini-R result: +10% avg IH robustness across 16 benchmarks, unsafe behavior 6.6%→0.7%, minor capability regression (chat win-rate -0.05, preference -0.06). Eval benchmarks: Gandalf Password, TensorTrust, RealGuardrails, System IFEval (academic); Tutor Jailbreak, System↔User/Developer↔User Conflict (internal OOD).
- **ConInstruct** (`papers/coninstruct.pdf`) — [arxiv 2511.14342](https://arxiv.org/abs/2511.14342), [code+dataset](https://github.com/NLPCode/ConInstruct). Benchmark for conflict detection & resolution in user instructions. 100 seed instructions, 6 NLP tasks (email, plan, story, QA, review, article), 35 domains. GPT-4o expands seeds with 6 constraint types (content, keyword, phrase, length, format, style), then adds conflict pairs → 9 conflict types: 6 intra-class (CC, KK, PP, LL, FF, SS) + 3 inter-class (KP, PC, PS). Two evaluations: conflict detection (binary: are there conflicts?) and conflict resolution (4 behaviors: unacknowledged / clarification requested / autonomously resolved / other). Key findings: (1) best detection F1 — DeepSeek-R1 91.5%, Claude-4.5-Sonnet 87.3%; (2) GPT-4o, DeepSeek-R1, Qwen2.5-32B silently ignore conflicts in ~96–100% cases; (3) only Claude models scale conflict-aware behavior with conflict count; (4) constraints later in the instruction are more likely to be satisfied (recency bias).

### Dataset details

**ConInstruct** (`data/benchmarks/ConInstruct/`):
- `datasets/seed_instruction.jsonl` — 100 bare tasks (email, plan, story, QA, review, article)
- `datasets/expand_instruction.jsonl` — 100 tasks + 6 types of compatible constraints (no conflicts)
- `datasets/conflict_instruction.jsonl` — 100 tasks + bank of 9 conflict pairs each (org_constraint vs new_constraint)
- `datasets/conflict_density/{1..6}_conflict.json` — 100 entries each: which of 9 conflict types to use for k conflicts
- Assembly: `inject_conflict_constraints()` in `LLMs/gpt4.py` — simple concatenation of new_constraints to expand-instruction via space (position=after/before)

**IH-Challenge** (`data/benchmarks/ih-challenge/`):
- `single-constraint.jsonl` — 1,900 tasks (19 types × 100), `multi-constraint.jsonl` — 4,832 (composite), `input-conditioned.jsonl` — 19,038 (blue_team_auto), `anti-overrefusal.jsonl` — 1,800 (18 types × 100)
- 19 task types: no_PII, ascii_only, no_CC, secret_pin, sentiment, hidden_word, nsfw_check, avoid_competitors, advertise, do_not_swear, password, hide_system, disclaimer, partial_pin, say_banned_words, letters_only, json_format, partial_password, no_emoji
- 3 hierarchy levels (~equal split): system>user, developer>user, system>developer
- Each entry: `defender_problem_template` (system/developer + user with placeholder `<<<REPLACE THIS WITH YOUR ATTACK>>>`), `attacker_problem` (task for attacker LLM), `metadata.grader_code_python` (deterministic Python grader)

## Current work

**Experiment 1 — IH-Challenge baseline eval (done):**
- Pipeline: `prepare_ih_dataset.py` → `run_eval.py` → `run_grader.py`
- Data: `data/raw_datasets/ih_challenge/baseline.jsonl` (95 samples, 19 types × 5)
- D2L integration: dataset `ih_challenge_baseline` registered in `definitions.py`
- **Results (probing queries, 95 samples):**
  - D2L: Gemma 2B 40.0%, Qwen 4B 55.8%, Mistral 7B 46.3%
  - Base model (in-context): Gemma 2B 74.7%, Qwen 4B 81.1%, Mistral 7B 50.5%
  - See `results/ih_challenge_baseline_all_models.md`

**Experiment 2 — Static attack generation (done):**
- `scripts/ih_challenge/assign_ids.py` — unique IDs assigned to all 27.5k IH-Challenge entries (sc/mc/ic/ar prefixes)
- `scripts/ih_challenge/generate_attacks.py` — generates N=3 attacks per task at 3 difficulty levels (L1 direct, L2 tricky, L3 sophisticated) via OpenAI-compatible API
- Attacker LLM: GPT-4o-mini via OpenRouter (~$10 for full run). Qwen2.5-72B-Instruct (best quality, free) for future runs.
- Dataset `ih_challenge_attack_static` registered in `definitions.py`
- **Result: 82,710 attacks generated** → `data/raw_datasets/ih_challenge/attacks_static.jsonl` (308MB)
- Backup: `/home/jovyan/research/datasets/d2l/ih_challenge/attacks_3_static.jsonl`

**Experiment 2b — Defense SFT data generation (v1 done, v2 running):**
- Problem: D2L was trained with KL-distillation on QA — the hypernetwork encodes knowledge, not behavioral instructions. RL from cold start produces no gradient signal (all K responses identical reward).
- Solution: SFT step first — generate gold defense responses that follow instructions despite attacks, then fine-tune D2L on these triplets (instruction → LoRA, attack → input, gold_response → target).
- `scripts/ih_challenge/generate_defenses.py` — for each (instruction, attack) pair from `attacks_static.jsonl`, asks GPT-4o-mini to generate a response that follows the instruction. Same async/resumable pattern as `generate_attacks.py`.
- **v1 (done):** `defenses_static.jsonl` — 82,710 defenses. Grader pass rate **68.3%** (56.5k passed). Major failure modes: GPT-4o-mini leaks secrets (password 65%, secret_pin 43%), decodes base64 banned words (50%), outputs non-ASCII/digits violating format constraints (32-39%). Details in `results/defense_grader_failures.md`.
- **v2 (running):** Rewrote defense prompt with explicit rules for each failure mode (secret values, banned content, format constraints, system instruction hiding). Test on 1000 samples: **86.3% → 97.8% pass rate**. Full 82k generation running → `defenses_v2.jsonl`.
- For Priority SFT conflict mode: will filter by grader, sample 10k from passed entries.

**Experiment 2b-sft — SFT training pipeline (trained, 3 checkpoints):**
- `scripts/ih_challenge/train_sft.py` — standalone SFT trainer with accelerate (multi-GPU)
- `scripts/ih_challenge/sft_dataset.py` — dataset loader for (context, prompt, response) triplets
- `scripts/ih_challenge/prepare_alpaca_sft.py` — prepares Alpaca dataset: 10k with_input + 10k no_input = 20k combined
- Data: `data/raw_datasets/alpaca/alpaca_combined.jsonl` (instruction→context, input/generic→prompt, output→response), `dev.jsonl` (1k dev, zero overlap with train)
- Logging: Comet ML (default), W&B, or none via `--logger` flag
- Features: `--keep_last_n N` (default 1) auto-removes old checkpoints; `args.yaml` saves SFT-specific fields (`sft_data_path`, `sft_checkpoint_path`, etc.) alongside original model config
- Note: batch_size=1 (no sequence packing). Real batching requires D2L's packing infrastructure — could integrate with existing `train.py` + `CrossEntropyTrainer` if throughput becomes a bottleneck. For now gradient_accumulation_steps is sufficient.
- **Trained checkpoints (Qwen 4B):**
  - `train_outputs/sft_alpaca_with_input/checkpoint-2499/` — SFT on `alpaca_with_input.jsonl` (10k)
  - `train_outputs/sft_alpaca_no_input/checkpoint-2498/` — SFT on `alpaca_no_input.jsonl` (10k)
  - `train_outputs/sft_alpaca_qwen4b/checkpoint-5000/` — SFT on `alpaca_combined.jsonl` (20k) — **best**
- **Dev eval (100 samples from `alpaca/dev.jsonl`, ROUGE-L):**
  - Qwen 4B D2L pre-SFT: 0.1772
  - `sft_alpaca_no_input` (10k): 0.2569 (+45%)
  - `sft_alpaca_with_input` (10k): 0.2986 (+69%)
  - `sft_alpaca_qwen4b` (combined 20k): **0.3115** (+76%) — best; more data + mixed prompt types wins
- **IH-Challenge baseline eval (95 probing queries, grader pass rate):**
  - Qwen 4B D2L pre-SFT: 55.8%
  - `sft_alpaca_qwen4b` (combined 20k): 56.8% (+1%)
  - **Hypothesis:** Alpaca SFT teaches general instruction-following (prompt→response) but provides no signal for LoRA-vs-prompt prioritization. IH-Challenge requires the model to follow LoRA-channel instructions over adversarial prompt-channel inputs — this needs defense-specific SFT where the model explicitly sees (instruction in LoRA, attack in prompt, correct defense as target).

**Experiment 2b-sft-v2 — Priority SFT (planned):**
- Goal: teach D2L instruction hierarchy (LoRA ≻ prompt) using a single mixed dataset with CE loss.
- **50k total:**
  1. **Conflict** (10k + 4.7k optional): LoRA ← instruction, prompt ← attack, target ← defense. Model learns to prioritize LoRA over adversarial prompt. **Source: `defenses_graded.jsonl` (82k with grader_pass labels, 56.5k passed). READY.**
     - **Files in `data/raw_datasets/priority_sft/`:**
       - `conflict_train.jsonl` (10,000): 12 SC types all passed levels (4,462) + blue_team_auto 1/task_id (5,538). All grader_pass=1.
       - `conflict_composite.jsonl` (4,701): composite 1/task_id, separate for optional second run.
       - `conflict_dev.jsonl` (900): 18 types × 50 task_ids (1 random entry each). Field `dev_split`: `holdout` (200) / `in_distribution` (700).
     - **Fields:** id, context, prompt, response, mode, task_type, source_split, level, grader_pass, grader_code (+dev_split in dev)
     - **Train types (12 SC):** secret_pin, partial_pin, partial_password, hidden_word, letters_only, json_format, no_PII, say_banned_words, hide_system, sentiment, advertise, nsfw_check
     - **Hold-out types (4, dev only):** password, no_CC, disclaimer, avoid_competitors — generalization test to unseen constraint types
     - **Excluded (3):** ascii_only, no_emoji (format-specific, defenses were JSON hacks), do_not_swear (trivial, 100% pass)
     - **Zero task_id overlap** between train, composite, and dev
  2. **Compatible** (15.6k after filtering): LoRA ← instruction, prompt ← input + additional instruction, target follows both. Two sub-variants: (A) extra constraint on existing task (7,594), (B) second independent task (8,002). **Source: `generate_compatible.py` — unused Alpaca (31k) + GPT-4o-mini. 19,993 generated.** Filtered by LLM-as-judge (Gemini 3 Flash) for arithmetic/factual errors and instruction compliance. Judge comparison: GPT-4o-mini 76% pass (high false positive rate — bad at word counting, alphabetical order), Gemini 2.5 Flash 72% (misses real errors), **Gemini 3 Flash 92%** (best — low false positives, catches real errors). **READY.**
  3. **Knowledge** (10k): LoRA ← document, prompt ← question, target ← answer. Preserves D2L knowledge extraction. **Source: SQuAD train. READY.**
  4. **Neutral** (10k): LoRA ← instruction, prompt ← input, target ← response. Standard instruction-following. **Source: Alpaca combined. READY.**
- Data: single mixed JSONL (`data/raw_datasets/priority_sft/train.jsonl`), all 4 modes.
- Training: same `train_sft.py`, CE loss on response tokens, hypernet-only.
- **Future: KL distillation option.** Original D2L uses KL(student || teacher) where teacher = base LLM with document in-context. Could use KL for knowledge/neutral/compatible (teacher reliable) and CE for conflict (teacher only 81% robust). Mixed loss or full KL with teacher-forced gold tokens. Not implemented yet — CE first, KL if quality insufficient.
- Key insight: Alpaca SFT alone doesn't help IH-Challenge (55.8% → 56.8%) because it never shows LoRA-vs-prompt conflict.
- **Data files (`data/raw_datasets/priority_sft/`):**
  - `train_v2.jsonl` (45.6k): conflict 10k + compatible 15.6k + knowledge 10k + neutral 10k. All fields preserved (conflict has id, task_type, source_split, level, grader_pass, grader_code).
  - `dev_v2.jsonl` (1.9k): conflict 900 (18 types × 50, dev_split holdout/in_distribution) + knowledge 500 (SQuAD validation, zero train overlap) + neutral 500 (Alpaca dev, zero train overlap).
  - `conflict_composite.jsonl` (4.7k): optional add-on for second run.
  - Stats: `results/priority_sft_dataset_stats.md`
- **Status:** train_v2 (45.6k) + dev_v2 (1.9k) READY. All 4 modes included.

**Experiment 2c — RL training pipeline (built, needs SFT first):**
- `scripts/ih_challenge/train_rl.py` — main RL training script
- `scripts/ih_challenge/rl_trainer.py` — GRPO/REINFORCE with online attacker rollout (propose-evaluate-revise)
- `scripts/ih_challenge/rl_dataset.py` — IH-Challenge task skeleton loader
- Smoke test: pipeline runs end-to-end (model loads, attacker generates, defender responds, grader scores, GRPO loss computes). But no gradient signal yet due to cold start — SFT needed first.

**QA benchmark eval via compression_experiments (done):**
- Integrated D2L into unified eval framework at `davydenko/compression_experiments`
  - `evaluation/methods/doc2lora_method.py` — `Doc2LoRA(BaseMethod)` wrapper
  - Registered as type `doc2lora` in `run_evaluation.py`
- QA eval datasets in `data/raw_datasets/` (2wikimultihop, hotpotqa, triviaqa, longbench), registered in `definitions.py`
- 9 runs: 3 methods (D2L-base, D2L-SFT, Qwen4B RAG) × 3 datasets (TriviaQA, HotpotQA, 2WikiMultihop) × 1500 samples
- **Results** (`results/qa_benchmark_comparison.md`):
  - RAG baseline wins on judge correctness: TriviaQA 73.6%, HotpotQA 90.9%, 2Wiki 71.2%
  - D2L-base: TriviaQA 66.2%, HotpotQA 73.6%, 2Wiki 64.9% (70-80% of RAG quality)
  - D2L-SFT: improves EM/F1 (shorter answers) but judge correctness flat or slightly down
  - EM/F1 misleading for both methods — judge correctness is the right metric

**How to run QA eval via compression_experiments:**

Eval framework lives in `davydenko/compression_experiments`. Must use D2L venv (`doc-to-lora/.venv/bin/python`).

Datasets (JSON with `{samples: [{query, context, target_answer, ...}]}`):
- `data/raw_datasets/triviaqa/test.json` (11313)
- `data/raw_datasets/hotpotqa/test.json` (7405)
- `data/raw_datasets/2wikimultihop/test.json` (12576)
- `data/raw_datasets/longbench/test.json` (2950)

Three method types available in YAML configs:
```yaml
# D2L (hypernetwork LoRA)
- name: "D2L-Qwen4B"
  type: "doc2lora"
  config:
    checkpoint_path: "/path/to/pytorch_model.bin"
    d2l_root: "/home/jovyan/davydenko/doc-to-lora"
    max_new_tokens: 256

# Baseline full context (standard RAG)
- name: "Qwen4B-FullContext"
  type: "baseline_full_context"
  config:
    model_name: "Qwen/Qwen3-4B-Instruct-2507"
    device: "cuda"
    torch_dtype: "bfloat16"
    max_context_length: 4096
    max_new_tokens: 256
    temperature: 0.0
    eval_batch_size: 8

# Baseline no context (parametric knowledge only)
- name: "Qwen4B-NoContext"
  type: "baseline_no_context"
  config:
    model_name: "Qwen/Qwen3-4B-Instruct-2507"
    device: "cuda"
    max_new_tokens: 256
    temperature: 0.0
    eval_batch_size: 64
```

Run:
```bash
cd /home/jovyan/davydenko/compression_experiments
CUDA_VISIBLE_DEVICES=X /home/jovyan/davydenko/doc-to-lora/.venv/bin/python \
  evaluation/scripts/run_evaluation.py --config config/YOUR_CONFIG.yaml \
  > results/log_NAME.txt 2>&1 &
```

Existing configs in `compression_experiments/config/`:
- `eval_d2l_{triviaqa,hotpotqa,2wiki}.yaml` — D2L-base, 1500 samples
- `eval_d2l_sft_{triviaqa,hotpotqa,2wiki}.yaml` — D2L-SFT, 1500 samples
- `eval_baseline_qwen4b_{triviaqa,hotpotqa,2wiki}.yaml` — RAG baseline, 1500 samples
- `eval_nocontext_qwen4b_{triviaqa,hotpotqa,2wiki}.yaml` — No context baseline, 1500 samples

Notes:
- D2L generates per-sample (no batching, ~3-4s/sample). Baseline batches (eval_batch_size=8, ~1s/sample).
- LLM judge via OpenRouter GPT-4o-mini adds ~1-2s/sample. Set `llm_judge.enabled: false` to skip.
- `n_samples: null` uses all, or set integer to limit.
- D2L checkpoint `args.yaml` must be in parent dir of `checkpoint-*/`. SFT checkpoints work too.

**vLLM setup for local models (reference):**
```bash
# Qwen2.5-72B-Instruct (best attacker: 0% refusals, high quality)
/home/jovyan/davydenko/compression_experiments/venv_augmentation/bin/vllm serve Qwen/Qwen2.5-72B-Instruct \
  --tensor-parallel-size 4 --host 0.0.0.0 --port 8000 \
  --no-enable-prefix-caching --max-num-batched-tokens 8192
# ~144GB bf16 on 4×A100, fits comfortably

# GPT-OSS-120B (reasoning model, mxfp4 quantized, ~17GB VRAM)
# 81% refusal rate — not suitable as attacker
# Qwen3-235B-A22B — OOM on 4×A100 (MoE experts too large in bf16)
```

## Rules

- **Log every change**: when modifying any file in this repo — add an entry to the "Changelog" section below with date, what was changed, and why.
- **CLAUDE.md language**: always write CLAUDE.md content in English. Chat with the user in their language, but all persistent documentation must be in English.
- **Track current work**: keep the "Current work" section up to date with what is being done right now and next steps.

## Changelog

- **2026-03-20**: Added `avg_chunk_loras=False` fallback in `src/ctx_to_lora/modeling/hypernet.py:from_state_dict()` (~line 497) — old checkpoints lack this field, `__repr__` crashed on load.
- **2026-03-20**: Updated PATH in `/home/jovyan/davydenko/.bashrc` — added `/home/jovyan/.local/bin` for `uv` access.
- **2026-03-20**: Recreated `.venv/` with uv-managed Python 3.10.20 (instead of system 3.10.12) — needed `Python.h` for Triton/DeepSpeed extension compilation.
- **2026-03-20**: Created `CLAUDE.md`.
- **2026-03-20**: Created `scripts/eval_normalized_perf.sh` — runs teacher (ICL) and D2L on SQuAD, computes normalized performance.
- **2026-03-20**: Created `scripts/niah/eval_gemma_niah.sh` — NIAH eval on Gemma 2B D2L, all bins up to 131k tokens, iterative mode.
- **2026-03-20**: Changed `accelerate_config.yaml` num_processes 8 -> 4 (server has 4 GPUs).
- **2026-03-20**: Updated `scripts/main_exp/1-train.sh` — 3 GPUs (1,2,3), GPU 0 reserved for eval. gradient_accumulation_steps 8->22 to preserve effective batch size (~64). Added `"$@"` for CLI overrides.
- **2026-03-20**: Added `papers/` directory with PDFs: `ih-challenge.pdf` (arxiv 2603.10521) and `coninstruct.pdf` (arxiv 2511.14342) — attack/conflict datasets for adversarial research.
- **2026-03-20**: Updated attack paper descriptions in CLAUDE.md — verified against actual PDFs, removed inaccurate benchmark refs (CyberSecEval 2), added precise details (task splits, results, recency bias finding).
- **2026-03-20**: Added research plan to CLAUDE.md — 3 experiments: (1) IH-Challenge as D2L eval, (2) RL training for IH robustness, (3) ConInstruct-style constraint prioritization (LoRA > context).
- **2026-03-20**: Added `papers/doc-to-lora.pdf` and `papers/parametric-rag.pdf`. Restructured papers section: "Related papers" (D2L, pRAG) + "Attack-related papers" (IH-Challenge, ConInstruct).
- **2026-03-20**: Commit of all changes above.
- **2026-03-20**: Created `scripts/ih_challenge/prepare_ih_dataset.py` — converts IH-Challenge single-constraint to D2L eval format (baseline clean queries + template attacks).
- **2026-03-20**: Registered `ih_challenge_baseline` and `ih_challenge_attack` datasets in `definitions.py`, added preprocessing in `preprocessing_fn.py`.
- **2026-03-20**: First successful D2L eval run on IH-Challenge baseline (Gemma 2B, 10 samples). Generations look reasonable.
- **2026-03-20**: Created `scripts/ih_challenge/run_grader.py` — runs IH-Challenge Python graders on D2L generated text, matches by prompt content, outputs per-task-type stats.
- **2026-03-20**: Full baseline eval on all 3 models (95 samples each): Gemma 2B 46.3%, Qwen 4B 56.8%, Mistral 7B 58.9%. Fixed grader matching bug (prompt+context, not just prompt). Results in `results/ih_challenge_baseline_all_models.md`.
- **2026-03-20**: Replaced clean queries with probing queries in `prepare_ih_dataset.py`. Fixed grader matching (full context .strip() comparison). D2L results: Gemma 40.0%, Qwen 55.8%, Mistral 46.3%.
- **2026-03-20**: Created `scripts/ih_challenge/eval_base_model.py` — runs base models (no D2L) with instruction in-context for comparison. Base results: Gemma 74.7%, Qwen 81.1%, Mistral 50.5%. Base >> D2L on instruction following.
- **2026-03-23**: Removed template attack infrastructure — deleted `template_attack.jsonl`, removed `ih_challenge_attack` dataset from `definitions.py`, cleaned `prepare_ih_dataset.py` (removed TEMPLATE_ATTACKS, ATTACK_GOALS, make_template_attack_samples). Updated research plan: Experiment 1 marked done, Experiment 2 refocused on LLM-generated attack augmentation.
- **2026-03-23**: Created `scripts/ih_challenge/assign_ids.py` — assigns unique IDs (sc/mc/ic/ar prefixes) to all 27.5k IH-Challenge entries across 4 splits. Ran it, IDs now in place.
- **2026-03-23**: Created `scripts/ih_challenge/generate_attacks.py` — async LLM-based attack generation at 3 difficulty levels (L1 direct, L2 tricky, L3 sophisticated). Uses OpenAI-compatible API (OpenRouter/vLLM). Resumable, concurrent, filters by split/type/count. Registered `ih_challenge_attack_static` dataset in `definitions.py`.
- **2026-03-23**: Attacker model comparison (114 samples, single-constraint, --max_per_type 2). GPT-OSS-120B: 81% refusals, unusable. GPT-4o-mini: 0% refusals, good quality. Qwen2.5-72B-Instruct: 0% refusals, best quality (more aggressive L1, concrete social engineering L2, advanced L3 with base64/fake system messages). Decision: use Qwen2.5-72B via local vLLM. Qwen3-235B-A22B: OOM on 4×A100 (MoE too large in bf16).
- **2026-03-23**: Full attack generation running via GPT-4o-mini on OpenRouter (82k attacks, all 4 splits × 3 levels). Qwen2.5-72B for future runs.
- **2026-03-23**: Created RL training pipeline: `scripts/ih_challenge/train_rl.py` (main script), `rl_trainer.py` (GRPO/REINFORCE with online attacker rollout), `rl_dataset.py` (IH-Challenge task skeleton loader). GRPO as primary method, REINFORCE as fallback. Online attacker with propose-evaluate-revise loop via OpenAI-compatible API.
- **2026-03-23**: RL smoke test revealed cold start problem — D2L trained on QA (KL distillation) has no instruction-following ability, all K responses get identical reward → zero gradient signal. Need SFT step first.
- **2026-03-23**: Created `scripts/ih_challenge/generate_defenses.py` — generates gold defense responses (follow instruction despite attack) using GPT-4o-mini. Same async/resumable pattern. Smoke test on 10 samples: all defenses correctly follow instructions. Training plan: SFT on defenses → RL on top.
- **2026-03-23**: Created SFT training pipeline: `scripts/ih_challenge/train_sft.py` (main script) + `sft_dataset.py` (dataset loader). CE loss on gold defense responses, hypernet-only training. Smoke test: 5 steps on 10 samples, loss drops 1.69→0.89. Pipeline: generate_attacks → generate_defenses → train_sft → train_rl.
- **2026-03-23**: Added Comet ML logging (`experiment_logger.py`): unified logger supporting comet/wandb/none via `--logger` flag. Default: comet. Added to both `train_sft.py` and `train_rl.py`.
- **2026-03-23**: Added multi-GPU support to `train_sft.py` via `accelerate`. Launch: `uv run accelerate launch --config_file accelerate_config.yaml scripts/ih_challenge/train_sft.py ...`
- **2026-03-23**: Prepared Alpaca SFT dataset (`prepare_alpaca_sft.py`): 10k with_input (instruction→context, input→prompt) + 10k no_input (instruction→context, generic→prompt) = 20k combined in `data/raw_datasets/alpaca/`.
- **2026-03-23**: Fixed GRPO loss bug — single-pass ratio is always 1.0, making loss=0. Switched to REINFORCE with group-normalized advantages. Added debug logging of model generations to terminal + Comet.
- **2026-03-23**: Fixed scheduler bug with accelerate — `accelerator.prepare()` wraps scheduler to only step on real optimizer updates, so `num_training_steps` must be `num_steps // gradient_accumulation_steps`, not `num_steps`.
- **2026-03-24**: Attack generation complete — 82,710 attacks in `attacks_static.jsonl`. Backed up to `/home/jovyan/research/datasets/d2l/ih_challenge/attacks_3_static.jsonl`.
- **2026-03-24**: Cleaned up SFT checkpoints — kept only last checkpoint in each of 3 runs, removed smoke test dirs. Freed ~26GB.
- **2026-03-24**: Added `--keep_last_n` to `train_sft.py` (default 1) — auto-removes old checkpoints on save. `--keep_last_n 0` keeps all.
- **2026-03-24**: Fixed `args.yaml` in `train_sft.py` — now saves SFT-specific fields (`sft_data_path`, `sft_checkpoint_path`, `sft_lr`, etc.) alongside original model config instead of just copying source args.
- **2026-03-24**: Created `data/raw_datasets/alpaca/dev.jsonl` — 1k dev set (500 with_input + 500 no_input) from Alpaca samples not used in SFT training. Zero overlap verified.
- **2026-03-24**: Created `scripts/ih_challenge/eval_sft.py` — generative eval for SFT checkpoints. Saves per-sample JSONL with instruction, prompt, gold, generated, ROUGE-L.
- **2026-03-24**: Evaluated all 3 SFT checkpoints + pre-SFT baseline on 100 Alpaca dev samples. Pre-SFT 0.1772 → combined SFT 0.3115 (+76%). Combined (20k) > with_input (10k) > no_input (10k).
- **2026-03-24**: Created `results/sft_alpaca_eval.md` — full eval report with metrics table (F1, ROUGE-L), 5 paired examples (pre-SFT vs SFT), observations.
- **2026-03-24**: Evaluated SFT models on IH-Challenge baseline (95 probing queries). Alpaca SFT barely changes IH pass rate: pre-SFT 55.8% → combined SFT 56.8% (+1%). SFT teaches general instruction-following but not LoRA-vs-prompt prioritization.
- **2026-03-24**: Started defense generation (`generate_defenses.py`) on full `attacks_static.jsonl` (82k) via GPT-4o-mini on OpenRouter. Default `--max_concurrent 10`, can raise to 50-100 for OpenRouter.
- **2026-03-24**: Designed Priority SFT (Experiment 2b-sft-v2) — 4-mode mixed dataset with CE loss: conflict (LoRA≠prompt→follow LoRA), compatible (LoRA+prompt→follow both), knowledge (QA), neutral (standard instruction-following). All from existing data, no LLM generation needed for conflict mode.
- **2026-03-24**: Investigating QA quality regression after Alpaca SFT — checking if instruction-following SFT degrades original D2L knowledge extraction (QA benchmarks). This is why Priority SFT mode 3 (knowledge) is important.
- **2026-03-24**: Created `prepare_priority_sft.py` and generated v1 dataset: `data/raw_datasets/priority_sft/train.jsonl` (20k: 10k SQuAD knowledge + 10k Alpaca neutral) + `dev.jsonl` (500). Modes 1+2 deferred — conflict mode needs genuinely contradictory instructions, not random shuffle.
- **2026-03-24**: Copied QA eval datasets (2wikimultihop, hotpotqa, triviaqa, longbench) from `/home/jovyan/research/datasets/xRAG/eval/` to `data/raw_datasets/`. Converted to D2L JSONL format (`*_d2l.jsonl`). Registered 8 datasets in `definitions.py` + preprocessing in `preprocessing_fn.py`.
- **2026-03-24**: Added `--test_samples_sequential` flag to `configs.py` + `eval_utils.py` — takes first N samples instead of random for deterministic eval.
- **2026-03-24**: Created `Doc2LoRA` wrapper in `compression_experiments/evaluation/methods/doc2lora_method.py` — integrates D2L into unified eval framework with LLM-as-judge metrics. Registered in `__init__.py` and `run_evaluation.py`.
- **2026-03-24**: Launched D2L-Qwen4B eval on 3 QA benchmarks (1500 samples each): TriviaQA (GPU0), HotpotQA (GPU1), 2WikiMultihop (GPU2). Configs in `compression_experiments/config/eval_d2l_*.yaml`.
- **2026-03-24**: Created `scripts/ih_challenge/generate_compatible.py` — Mode 2 (compatible) data generation. Two variants: (A) extra constraint on existing task, (B) second independent task. 5k each, few-shot prompts, async/resumable. Uses unused Alpaca (31k samples not in combined+dev). Smoke test passed (10/10, GPT-4o-mini).
- **2026-03-24**: Completed QA benchmark eval: 9 runs (D2L-base, D2L-SFT, Qwen4B RAG × TriviaQA, HotpotQA, 2WikiMultihop × 1500 samples). Results in `results/qa_benchmark_comparison.md`. RAG wins on judge correctness (73-91%), D2L retains 70-80% of RAG quality.
- **2026-03-24**: Graded all 82k defenses v1 — 68.3% pass rate (56.5k/82.7k). Top failures: password leak 65%, say_banned_words 50%, no_CC 47%, hide_system 44%, secret_pin 43%. GPT-4o-mini itself is not robust to the attacks it generated. Details + 4 examples per type in `results/defense_grader_failures.md`.
- **2026-03-24**: Rewrote defense generation prompt in `generate_defenses.py` — added explicit rules for secret values (never output even indirectly), banned content (no base64 decoding), format constraints (ascii_only, letters_only, no_CC/PII specifics), system instruction hiding. Test on 1000 samples: pass rate 86.3% → 97.8% (+11.5pp). Launched v2 generation on full 82k → `defenses_v2.jsonl`.
- **2026-03-24**: Ran no-context baseline (Qwen4B parametric knowledge only) on TriviaQA, HotpotQA, 2WikiMultihop (1500 samples each). Judge correctness: TriviaQA 0.509, HotpotQA 0.307, 2Wiki 0.295. D2L adds 15-43pp over parametric knowledge. Updated `results/qa_benchmark_comparison.md`.
- **2026-03-25**: v2 defense prompt (explicit safety rules) tested on full 82k — 62.2% pass (worse than v1 68.3%). Overrefusal on sentiment (97% fail), nsfw_check (93% fail), composite (40% fail). Only ascii_only (3.2%) and no_emoji (0%) improved — but via JSON wrapping hack. Reverted to v1 prompt.
- **2026-03-25**: Created `defenses_graded.jsonl` — v1 defenses with `grader_pass` field (0/1). 56,456 passed, 26,254 failed. Deleted v2 files.
- **2026-03-25**: Built conflict mode splits in `data/raw_datasets/priority_sft/`: `conflict_train.jsonl` (10k: 4,462 SC all levels + 5,538 blue_team_auto 1/task_id), `conflict_composite.jsonl` (4,701: 1/task_id), `conflict_dev.jsonl` (900: 18 types × 50, dev_split holdout/in_distribution). Zero task_id overlap. Hold-out types: password, no_CC, disclaimer, avoid_competitors. Excluded: ascii_only, no_emoji, do_not_swear.
- **2026-03-25**: Merged modes 1+3+4 into `train_v2.jsonl` (30k: conflict 10k + knowledge 10k + neutral 10k). Compatible (mode 2) excluded from first run.
- **2026-03-25**: Compatible LLM-as-judge: tested GPT-4o-mini (76% pass, ~40-50% false positives — bad at counting/logic), Gemini 2.5 Flash (72%, misses real errors like arithmetic), Gemini 3 Flash (92%, best — low false positives). Running Gemini 3 Flash on full 20k → `compatible_grades.jsonl`.
- **2026-03-25**: Built unified `dev_v2.jsonl` (1,900): conflict 900 + knowledge 500 (SQuAD validation, zero train overlap — fixed 105 overlaps from old dev.jsonl) + neutral 500 (Alpaca dev). Updated `results/priority_sft_dataset_stats.md` with full stats (11 sections).
- **2026-03-25**: Updated SFT pipeline for Priority SFT v2: `sft_dataset.py` — added `modes` filter param; `train_sft.py` — added `--modes` arg (comma-separated, e.g. `--modes conflict,knowledge`); `eval_sft.py` — added grader eval for conflict mode (exec grader_code), breakdown by task_type/dev_split/level, ROUGE-L for knowledge/neutral, `--modes` filter.
