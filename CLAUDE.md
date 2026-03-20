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
- `demo/app.py` — Gradio interface
- `data/` — data prep scripts + raw_datasets

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

### Demo
```bash
uv run demo/app.py
```

## Work direction

Adversarial attacks and instruction-following robustness for parametric knowledge injection (D2L and pRAG).

Both architectures share the same LoRA-vs-context split: instructions are encoded into LoRA weights, adversarial/conflicting input comes through the context window. The difference is how LoRA is produced:
- **D2L**: hypernetwork generates LoRA in a single forward pass (fast inference, trains the hypernetwork)
- **pRAG**: separate LoRA trained per document via SFT on augmented data (slow offline, trains the LoRA directly)

### Research plan

**Experiment 1 — IH-Challenge as eval:**
- Use IH-Challenge dataset as a test benchmark for both D2L and pRAG
- LoRA ← high-priority instructions (system/developer role). D2L: hypernetwork encodes the instruction document; pRAG: LoRA trained on augmented instruction document (rewrites + QA pairs, or adapted augmentation)
- Context window ← adversarial low-priority prompt (user/tool role, the "attack")
- Measure how well the model follows LoRA-encoded instructions despite adversarial context
- Analogy: LoRA = system-level instructions, context = untrusted user input → tests natural instruction hierarchy
- May need to adapt training pipeline for instruction-following (both are currently QA-oriented; pRAG uses rewrite+QA augmentation, could switch to paraphrase)

**Experiment 2 — RL training for IH robustness:**
- Build on Experiment 1 baseline
- Add RL training loop from IH-Challenge paper: frozen attacker generates adversarial context → defender scored by Python grader → policy gradient update
- RL updates whatever produced the LoRA: D2L → hypernetwork weights; pRAG → the LoRA adapter itself (same loop, different trainable params)
- Goal: improve robustness to adversarial prompts in context window

**Experiment 3 — Constraint prioritization (ConInstruct-style):**
- LoRA ← one set of output constraints (e.g., "don't use word X")
- Context ← conflicting constraint (e.g., "start sentence with phrase containing X")
- Goal: model should prioritize LoRA-channel over context-channel
- Closer to pure instruction-following than adversarial attacks
- Uses ConInstruct conflict types (CC, KK, PP, LL, FF, SS, KP, PC, PS) to systematically test prioritization
- Applies to both D2L and pRAG identically (only LoRA production differs)
- Future: extend with different RL strategies for constraint prioritization training (TBD)

## Related papers

PDFs in `papers/`:

- **Doc-to-LoRA** (`papers/doc-to-lora.pdf`) — [arxiv 2602.15902](https://arxiv.org/abs/2602.15902). Sakana AI. This repo's paper. Hypernetwork (ctx_encoder + Perceiver + HyperLoRA head) reads a document in one forward pass and generates LoRA weights for the LLM. Trained via KL-distillation from teacher (LLM with document in context). Eliminates need for per-document training or long context at inference.
- **Parametric RAG** (`papers/parametric-rag.pdf`) — [arxiv 2501.15915](https://arxiv.org/abs/2501.15915), [code](https://github.com/oneal2000/PRAG). Tsinghua. Injects documents into FFN parameters as LoRA adapters instead of context window. Offline pipeline: (1) Document Augmentation — LLM rewrites document n times + generates m QA pairs → cross-product of (rewrite, Q, A) triplets; (2) Parametric Document Encoding — trains randomly-initialized LoRA (FFN only, W'=W+AB^T) per document on augmented data with standard LM objective. Online: Retrieve-Update-Generate (RUG) — retrieve top-k docs → average-merge their LoRAs → update LLM → generate from query alone (no docs in context). Can combine with in-context RAG for further gains. Key difference from D2L: separate LoRA trained per document (slow offline, no hypernetwork) vs D2L's single forward pass. Models: LLaMA-3.2-1B, Qwen2.5-1.5B, LLaMA-3.1-8B. Benchmarks: 2WikiMultihopQA, HotpotQA, PopQA, ComplexWebQuestions.

### Attack-related papers

- **IH-Challenge** (`papers/ih-challenge.pdf`) — [arxiv 2603.10521](https://arxiv.org/abs/2603.10521), [dataset](https://huggingface.co/datasets/openai/ih-challenge). OpenAI. RL training dataset for instruction hierarchy (IH) robustness. IH policy: system ≻ developer ≻ user ≻ tool (lower-priority instructions honored only when compatible with higher-priority). Task skeletons constructed offline: high-priority instructions + placeholder for attack + Python grader as verifiable reward. Online: frozen attacker LLM fills the placeholder via budgeted propose-evaluate-revise loop with tool access to current defender. 4 task splits: Single-Constraint (IFEval-like: contain-word, hide-PIN, JSON-only, etc.), Multi-Constraint (composition of 2–6 atomic), Input-Conditioned (closed-ended: parse pattern → output in schema), Anti-Overrefusal (benign requests rewritten to look forbidden; refusal = failure). GPT-5-Mini-R result: +10% avg IH robustness across 16 benchmarks, unsafe behavior 6.6%→0.7%, minor capability regression (chat win-rate -0.05, preference -0.06). Eval benchmarks: Gandalf Password, TensorTrust, RealGuardrails, System IFEval (academic); Tutor Jailbreak, System↔User/Developer↔User Conflict (internal OOD).
- **ConInstruct** (`papers/coninstruct.pdf`) — [arxiv 2511.14342](https://arxiv.org/abs/2511.14342), [code+dataset](https://github.com/NLPCode/ConInstruct). Benchmark for conflict detection & resolution in user instructions. 100 seed instructions, 6 NLP tasks (email, plan, story, QA, review, article), 35 domains. GPT-4o expands seeds with 6 constraint types (content, keyword, phrase, length, format, style), then adds conflict pairs → 9 conflict types: 6 intra-class (CC, KK, PP, LL, FF, SS) + 3 inter-class (KP, PC, PS). Two evaluations: conflict detection (binary: are there conflicts?) and conflict resolution (4 behaviors: unacknowledged / clarification requested / autonomously resolved / other). Key findings: (1) best detection F1 — DeepSeek-R1 91.5%, Claude-4.5-Sonnet 87.3%; (2) GPT-4o, DeepSeek-R1, Qwen2.5-32B silently ignore conflicts in ~96–100% cases; (3) only Claude models scale conflict-aware behavior with conflict count; (4) constraints later in the instruction are more likely to be satisfied (recency bias).

## Rules

- **Log every change**: when modifying any file in this repo — add an entry to the "Changelog" section below with date, what was changed, and why.

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
