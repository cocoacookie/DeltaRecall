"""Small, inference-only helpers for the first GRAS-HMC prototype."""

from typing import Optional

import torch
import torch.nn.functional as F


def allocate_layer_budgets(
    demand: torch.Tensor,
    total_budget: int,
    min_budget: int = 0,
    max_budget: Optional[int] = None,
    align: int = 8,
) -> list[int]:
    """Allocate a fixed token budget across layers from calibration demand.

    The allocation is offline: the resulting list is placed in model config
    and the decode path only consumes the fixed per-layer counts.  Alignment
    is used while distributing the bulk of the budget; a small remainder is
    assigned one token at a time so the requested total is preserved.
    """
    values = torch.as_tensor(demand, dtype=torch.float64).flatten()
    if values.numel() == 0:
        return []
    if total_budget < 0 or min_budget < 0 or align <= 0:
        raise ValueError("budget and alignment must be non-negative/positive")

    values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    values = values.clamp_min(0)
    if float(values.sum()) == 0:
        values = torch.ones_like(values)
    weights = values / values.sum()

    layer_count = values.numel()
    lower = int(min_budget)
    upper = int(max_budget) if max_budget is not None else None
    if upper is not None and upper < lower:
        raise ValueError("max_budget must be >= min_budget")
    if total_budget < layer_count * lower:
        raise ValueError("total_budget is smaller than the layer minimum")
    if upper is not None and total_budget > layer_count * upper:
        raise ValueError("total_budget is larger than the layer maximum")

    budgets = torch.full((layer_count,), lower, dtype=torch.long)
    remaining = int(total_budget - int(budgets.sum()))
    if remaining == 0:
        return budgets.tolist()

    # Reserve aligned units for the weighted allocation.  A lower budget is
    # allowed to be unaligned; only the distributed portion is aligned.
    unit_count = remaining // align
    fractional = weights * unit_count
    add_units = torch.floor(fractional).to(torch.long)
    if upper is not None:
        capacity = ((upper - lower) // align)
        add_units = add_units.clamp_max(capacity)
    budgets += add_units * align

    left_units = unit_count - int(add_units.sum())
    while left_units > 0:
        available = torch.ones(layer_count, dtype=torch.bool)
        if upper is not None:
            available &= budgets + align <= upper
        if not bool(available.any()):
            break
        scores = fractional - add_units.to(torch.float64)
        scores[~available] = -torch.inf
        index = int(scores.argmax())
        budgets[index] += align
        add_units[index] += 1
        left_units -= 1

    # Preserve the exact requested total when it is not divisible by align.
    remainder = int(total_budget - int(budgets.sum()))
    if remainder:
        order = torch.argsort(weights, descending=True).tolist()
        for index in order:
            if upper is not None and int(budgets[index]) >= upper:
                continue
            take = min(remainder, upper - int(budgets[index]) if upper is not None else remainder)
            budgets[index] += take
            remainder -= take
            if remainder == 0:
                break
    if int(budgets.sum()) != total_budget:
        raise ValueError("could not satisfy the requested budget")
    return budgets.tolist()


def _normalize_keys(keys: torch.Tensor) -> torch.Tensor:
    keys = keys.float()
    return keys / torch.sqrt(keys.square().sum(dim=-1, keepdim=True) + 1e-6)


def _zero_state(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    batch, _, heads, key_dim = keys.shape
    value_dim = values.shape[-1]
    return torch.zeros(
        batch,
        heads,
        key_dim,
        value_dim,
        device=keys.device,
        dtype=torch.float32,
    )


def recurrent_demand(
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
):
    """Replay the GDN state transition and return per-head beta*||error||.

    Inputs use the AHN inference layout [batch, time, heads, dim].  This is a
    prefill/calibration path; the production implementation can replace it
    with a write-statistic returned by the fused kernel without changing the
    selector interface.
    """
    if keys.shape[:3] != values.shape[:3]:
        raise ValueError("keys and values must agree on batch, time, and heads")
    if keys.shape[1] == 0:
        state = initial_state if initial_state is not None else _zero_state(keys, values)
        return keys.new_empty((*keys.shape[:3],), dtype=torch.float32), state.float()

    beta = beta.float()
    alpha = alpha.float()
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)
    if beta.ndim != 3 or alpha.ndim != 3:
        raise ValueError(f"expected [B,T,H] gates, got {tuple(beta.shape)} and {tuple(alpha.shape)}")

    # The fused kernel performs the same gated-delta transition.  With q=k
    # normalized and scale=1, its post-update output is
    #   o_t = p_t + beta_t * (v_t - p_t).
    # Recover the update error from that output while retaining the final
    # state.  Keep the small Python path for CPU/short decode segments, where
    # kernel launch overhead is larger than the replay itself.
    if keys.is_cuda and keys.shape[1] >= 64:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        fused_keys = keys.float()
        fused_values = values.float()
        post_output, state = fused_recurrent_gated_delta_rule(
            q=fused_keys,
            k=fused_keys,
            v=fused_values,
            g=alpha,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        error = (fused_values - post_output.float()) / (1.0 - beta).unsqueeze(-1)
        return beta.float() * error.norm(dim=-1), state.float()

    batch, steps, heads, key_dim = keys.shape
    state = initial_state.float() if initial_state is not None else _zero_state(keys, values)
    normalized_keys = _normalize_keys(keys)
    scores = []

    for step in range(steps):
        state_before = state * torch.exp(alpha[:, step]).view(batch, heads, 1, 1)
        key = normalized_keys[:, step]
        value = values[:, step].float()
        prediction = torch.einsum("bhk,bhkv->bhv", key, state_before)
        error = value - prediction
        gate = beta[:, step]
        scores.append(gate * error.norm(dim=-1))
        state = state_before + (
            key.unsqueeze(-1)
            * error.unsqueeze(-2)
            * gate.unsqueeze(-1).unsqueeze(-1)
        )

    return torch.stack(scores, dim=1), state


def recurrent_memory_output(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
):
    """Run the all-write gated-delta stream and return its recurrent readout.

    ``queries`` uses the full-attention head layout while keys and values may
    use grouped-query heads.  The helper is used by the Qwen3.5 training path
    to align sink-plus-outgoing updates with sink-plus-suffix queries.
    """
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        raise ValueError("expected [B,T,H,D] query/key/value tensors")
    if queries.shape[:2] != keys.shape[:2] or keys.shape[:3] != values.shape[:3]:
        raise ValueError("query/key/value streams must share batch and time")
    beta = beta.float()
    alpha = alpha.float()
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if alpha.ndim == 4 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)
    if beta.shape[:2] != queries.shape[:2] or alpha.shape[:2] != queries.shape[:2]:
        raise ValueError("gate streams must share batch and time with queries")

    query_heads = queries.shape[2]
    key_heads = keys.shape[2]
    if query_heads != key_heads:
        if query_heads % key_heads:
            raise ValueError("query heads must be divisible by key heads")
        groups = query_heads // key_heads
        keys = keys.repeat_interleave(groups, dim=2)
        values = values.repeat_interleave(groups, dim=2)
    if beta.shape[2] != query_heads or alpha.shape[2] != query_heads:
        raise ValueError("gate head count must match query heads")

    normalized_queries = F.normalize(queries.float(), dim=-1)
    normalized_keys = F.normalize(keys.float(), dim=-1)
    if queries.is_cuda and queries.shape[1] >= 64 and not torch.is_grad_enabled():
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        output, state = fused_recurrent_gated_delta_rule(
            q=normalized_queries,
            k=normalized_keys,
            v=values.float(),
            g=alpha,
            beta=beta,
            scale=1.0,
            initial_state=initial_state,
            output_final_state=True,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return output.float(), state.float()

    if queries.is_cuda and torch.is_grad_enabled():
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

        output, state = torch_chunk_gated_delta_rule(
            normalized_queries,
            normalized_keys,
            values.float(),
            g=alpha,
            beta=beta,
            chunk_size=64,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        return output.float(), state.float()

    batch, steps, heads, key_dim = normalized_keys.shape
    value_dim = values.shape[-1]
    state = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(batch, heads, key_dim, value_dim, device=keys.device, dtype=torch.float32)
    )
    outputs = []
    for step in range(steps):
        state_before = state * torch.exp(alpha[:, step]).view(batch, heads, 1, 1)
        key = normalized_keys[:, step]
        value = values[:, step].float()
        prediction = torch.einsum("bhk,bhkv->bhv", key, state_before)
        error = value - prediction
        gate = beta[:, step]
        state = state_before + key.unsqueeze(-1) * error.unsqueeze(-2) * gate.unsqueeze(-1).unsqueeze(-1)
        outputs.append(torch.einsum("bhk,bhkv->bhv", normalized_queries[:, step], state))
    return torch.stack(outputs, dim=1), state


def select_topk(scores: torch.Tensor, budget: int) -> torch.Tensor:
    """Aggregate heads, select largest demand, and return chronological indices."""
    if scores.ndim != 3:
        raise ValueError(f"expected [B,T,H] scores, got {tuple(scores.shape)}")
    length = scores.shape[1]
    count = min(max(int(budget), 0), length)
    if count == 0:
        return torch.empty((scores.shape[0], 0), dtype=torch.long, device=scores.device)
    aggregate = scores.mean(dim=-1)
    selected = aggregate.topk(count, dim=-1, largest=True).indices
    return selected.sort(dim=-1).values


def gather_sequence(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a BLHD sequence by a per-batch index tensor [B,K]."""
    if indices.numel() == 0:
        return values[:, :0]
    gather_index = indices.to(values.device)[:, :, None]
    while gather_index.ndim < values.ndim:
        gather_index = gather_index.unsqueeze(-1)
    gather_index = gather_index.expand(-1, -1, *values.shape[2:])
    return values.gather(dim=1, index=gather_index)


def gather_cache_sequence(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a BHLD DynamicCache tensor [B,H,L,D] by [B,K] indices."""
    if indices.numel() == 0:
        return values[:, :, :0]
    gather_index = indices.to(values.device)[:, None, :, None]
    gather_index = gather_index.expand(-1, values.shape[1], -1, values.shape[-1])
    return values.gather(dim=2, index=gather_index)


def remove_sequence_item(values: torch.Tensor, index: int) -> torch.Tensor:
    """Remove one physical sequence item from a BHLD cache tensor."""
    return torch.cat([values[:, :, :index], values[:, :, index + 1 :]], dim=2).contiguous()
