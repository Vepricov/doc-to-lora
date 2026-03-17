"""
Utilities for merging / aggregating LoRA adapters coming from multiple chunks.
"""

import torch
from einops import rearrange
from jaxtyping import Float, Integer
from torch import Tensor


def compute_rank(n_lora, rank):
    return (n_lora + 1) * rank


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Integer[Tensor, "n_ctx"],
    lora_bias: dict[str, dict[str, Tensor]] | None = None,
    scalers: Float[Tensor, "n_ctx"] | None = None,
    bias_scaler: float | None = None,
    avg_loras: bool = False,
) -> dict[str, dict[str, Tensor]]:
    """Combine per-chunk LoRA weights into one LoRA per context.

    Two modes:
    - ``avg_loras=False`` (default): **stack** — concatenate ranks, producing a
      LoRA of rank ``(n_chunks + 1) * base_rank`` (extra slot for the shared bias).
    - ``avg_loras=True``: **average** — element-wise mean across chunks, keeping
      rank at ``base_rank`` (plus one bias slot if applicable).
    """
    if bias_scaler is None:
        bias_scaler = 1
    # Assume all modules share same base rank r
    first_module = next(iter(generated_loras))
    sampled_lora = generated_loras[first_module]["A"]
    base_rank = sampled_lora.shape[-2]
    device = sampled_lora.device
    dtype = sampled_lora.dtype

    if avg_loras:
        # rank stays at base_rank (+ one bias slot when a shared bias is present)
        max_rank_needed = base_rank + (base_rank if lora_bias is not None else 0)
    else:
        max_rank_needed = int(compute_rank(n_chunks.max(), base_rank))

    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }
    rank_dim = 2
    num_groups = len(n_chunks)
    rank_per_group = (n_chunks * base_rank).tolist()
    bias_tensor = None
    for module_name, module_loras in generated_loras.items():
        for matrix_key in ("A", "B"):
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key]
            loras = module_loras[matrix_key]

            if avg_loras:
                # Expand per-context scalers to per-chunk before weighting
                if (scalers is not None) and (matrix_key == "A"):
                    expanded_scalers = torch.repeat_interleave(scalers, n_chunks)
                    loras = loras * expanded_scalers[:, None, None, None]

                # Split along the tot_chunks dimension, one slice per context
                chunks_per_group = [int(c) for c in n_chunks.tolist()]
                per_group_loras = loras.split(chunks_per_group, dim=0)

                combined_shape = [num_groups, loras.shape[1], max_rank_needed, loras.shape[3]]
                combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

                for g, group_loras in enumerate(per_group_loras):
                    # group_loras: [n_chunks_g, n_layers, r, dim]
                    avg = group_loras.mean(dim=0)  # [n_layers, r, dim]
                    combined[g, :, :base_rank, :] = avg
                    if bias_tensor is not None:
                        combined[g, :, base_rank : base_rank * 2, :] = (
                            bias_tensor * bias_scaler
                        )

            else:
                if (scalers is not None) and (matrix_key == "A"):
                    loras = loras * scalers[:, None, None, None]

                flat_loras = rearrange(
                    loras, "tot_chunks n_layers r dim -> 1 n_layers (tot_chunks r) dim"
                )
                per_group_deltas = flat_loras.split(rank_per_group, dim=rank_dim)

                combined_shape = [num_groups, *per_group_deltas[0].shape[1:]]
                combined_shape[rank_dim] = max_rank_needed

                combined = torch.zeros(*combined_shape, device=device, dtype=dtype)

                for g, deltas in enumerate(per_group_deltas):
                    combined_rank = deltas.shape[rank_dim]

                    combined[g, :, :combined_rank, :] = deltas

                    if bias_tensor is not None:
                        combined[g, :, combined_rank : combined_rank + base_rank, :] = (
                            bias_tensor * bias_scaler
                        )

            combined_loras[module_name][matrix_key] = combined

    return combined_loras
