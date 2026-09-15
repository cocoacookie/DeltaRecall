"""GRAS-HMC adapter for Qwen3.5 full-attention layers.

Qwen3.5 already contains native linear-attention/GDN layers.  This adapter
only replaces the full-attention layers and leaves the native hybrid cache
state untouched.  Inference keeps a fixed exact cache plus an all-write
recurrent state.  The optional training path freezes the Qwen3.5 backbone and
learns only the small memory readout attached to each full-attention layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gras_hmc_selector import (
    gather_cache_sequence,
    gather_sequence,
    recurrent_demand,
    recurrent_memory_output,
    select_topk,
)


class GRASQwen35Attention(nn.Module):
    """Qwen3.5 full attention with a fixed ``[sink][select][window]`` cache."""

    def __init__(
        self,
        base_attention: nn.Module,
        config,
        layer_idx: int,
        select_budget: int,
        sliding_window: int,
        num_attn_sinks: int,
        demand_mode: str = "proxy",
        recurrent_token_limit: int = 2048,
        memory_output_scale: float = 0.0,
        train_memory: bool = False,
        memory_projection_rank: int = 0,
        sink_warm_start: bool = True,
        selection_include_sinks: bool = False,
        memory_write_mode: str = "all",
        selector_mode: str = "gras",
        retain_recurrent_state: bool = True,
    ):
        super().__init__()
        self.base_attention = base_attention
        self.config = config
        self.layer_idx = layer_idx
        self.select_budget = int(select_budget)
        self.sliding_window = int(sliding_window)
        self.num_attn_sinks = int(num_attn_sinks)
        self.demand_mode = demand_mode
        self.recurrent_token_limit = int(recurrent_token_limit)
        self.head_dim = base_attention.head_dim
        self.num_key_value_groups = base_attention.num_key_value_groups
        self.scaling = base_attention.scaling
        self.train_memory = bool(train_memory)
        self.memory_projection_rank = int(memory_projection_rank)
        self.sink_warm_start = bool(sink_warm_start)
        self.selection_include_sinks = bool(selection_include_sinks)
        self.memory_write_mode = memory_write_mode
        self.selector_mode = selector_mode
        self.retain_recurrent_state = bool(retain_recurrent_state)
        self.selector_head_weights: Optional[torch.Tensor] = None
        self.training_context_length: Optional[int] = None
        self.defer_compression = False
        self._deferred_query_states = None
        # Analysis-only causal intervention.  Entries are local attention-head
        # indices for this full-attention layer and persist across reset_state.
        self.masked_attention_heads: Optional[set[int]] = None
        self.num_heads = int(config.num_attention_heads)
        device = base_attention.q_proj.weight.device
        dtype = base_attention.q_proj.weight.dtype
        if self.train_memory:
            self.memory_output_scale = nn.Parameter(
                torch.tensor(float(memory_output_scale), device=device, dtype=torch.float32)
            )
            self.memory_beta_bias = nn.Parameter(
                torch.zeros(self.num_heads, device=device, dtype=torch.float32)
            )
            self.memory_log_decay = nn.Parameter(
                torch.full((self.num_heads,), -9.0, device=device, dtype=torch.float32)
            )
        else:
            self.memory_output_scale = float(memory_output_scale)
            self.register_parameter("memory_beta_bias", None)
            self.register_parameter("memory_log_decay", None)
        # Memory lives in attention-output space. This differs from hidden_size
        # for Qwen3.5-27B (24 * 256 != 5120).
        attn_output_dim = self.num_heads * self.head_dim
        if self.memory_projection_rank > 0:
            self.memory_down = nn.Linear(
                attn_output_dim, self.memory_projection_rank, bias=False, device=device, dtype=dtype
            )
            self.memory_up = nn.Linear(
                self.memory_projection_rank, attn_output_dim, bias=False, device=device, dtype=dtype
            )
            nn.init.normal_(self.memory_down.weight, std=0.02)
            nn.init.zeros_(self.memory_up.weight)
        else:
            self.memory_down = None
            self.memory_up = None
        self._local_mask: Optional[torch.Tensor] = None
        self.reset_state()

    def reset_state(self):
        self.select_positions: list[int] = []
        self.window_positions: list[int] = []
        # The sink is created only for the prefix that has fallen outside the
        # recent window.  For a short prefill this is zero, so the whole
        # context is represented by the window.
        self.active_sink_len = 0
        self.beta_cache: Optional[torch.Tensor] = None
        self.alpha_cache: Optional[torch.Tensor] = None
        self.mem_state: Optional[torch.Tensor] = None
        self.last_demand: Optional[torch.Tensor] = None
        self.prefill_scores: Optional[torch.Tensor] = None
        # Preserve the unreduced [batch, token, head] signal for Insight
        # analysis; selection continues to use the head mean.
        self.prefill_head_scores: Optional[torch.Tensor] = None
        self.prefill_score_offset = 0
        self.seen_tokens = 0
        # Optional analysis-only counterfactual controls.  Positions are
        # global prefill indices; the normal selector remains unchanged when
        # these lists are empty.
        self.force_include_positions: Optional[set[int]] = None
        self.force_exclude_positions: Optional[set[int]] = None
        self.force_state_exclude_positions: Optional[set[int]] = None

    def _project(self, hidden_states, position_embeddings):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.base_attention.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.base_attention.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.base_attention.k_norm(
            self.base_attention.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

    def _attention(self, query_states, key_states, value_states, attention_mask, q_len):
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

        use_gqa = (
            attention_mask is None
            and key_states.shape[-1] == value_states.shape[-1] <= 256
            and self.num_key_value_groups > 1
        )
        if not use_gqa:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        if q_len == 1:
            attention_mask = None
        is_causal = q_len > 1 and attention_mask is None
        kwargs = {
            "attn_mask": attention_mask,
            "dropout_p": 0.0,
            "is_causal": is_causal,
            "scale": self.scaling,
        }
        if use_gqa:
            kwargs["enable_gqa"] = True
        output = F.scaled_dot_product_attention(query_states, key_states, value_states, **kwargs)
        if self.masked_attention_heads:
            heads = sorted(h for h in self.masked_attention_heads if 0 <= h < output.shape[1])
            if heads:
                output = output.clone()
                output[:, heads] = 0
        return output

    def _gates(self, gate):
        batch, length, heads_times_dim = gate.shape
        heads = heads_times_dim // self.head_dim
        gate_heads = gate.view(batch, length, heads, self.head_dim).float().mean(dim=-1)
        if self.memory_beta_bias is None:
            beta = torch.sigmoid(gate_heads)
            alpha = torch.zeros_like(beta)
        else:
            beta = torch.sigmoid(gate_heads + self.memory_beta_bias.view(1, 1, -1))
            alpha = -F.softplus(self.memory_log_decay).view(1, 1, -1).expand_as(beta)
        return beta, alpha

    def _project_memory(self, memory):
        if self.memory_down is None:
            return memory
        return memory + self.memory_up(self.memory_down(memory))

    def _local_attention_mask(self, length, device):
        if self._local_mask is not None and self._local_mask.shape[-1] == length:
            return self._local_mask
        positions = torch.arange(length, device=device)
        query = positions[:, None]
        key = positions[None, :]
        causal = key <= query
        recent = key >= query - self.sliding_window + 1
        sink_len = min(self.num_attn_sinks, max(0, length - self.sliding_window))
        sink = key < sink_len
        self._local_mask = (causal & (recent | sink))[None, None]
        return self._local_mask

    def _training_memory_output(self, query_states, key_states, value_states, beta, alpha):
        sequence_length = query_states.shape[2]
        sink_len = min(self.num_attn_sinks, max(0, sequence_length - self.sliding_window))
        total_window = min(sink_len + self.sliding_window, sequence_length)
        outgoing_len = sequence_length - total_window
        if outgoing_len <= 0:
            return None, total_window

        update_end = sink_len + outgoing_len
        query_stream = torch.cat(
            [
                query_states[:, :, :sink_len].transpose(1, 2),
                query_states[:, :, total_window:].transpose(1, 2),
            ],
            dim=1,
        )
        key_stream = key_states[:, :, :update_end].transpose(1, 2)
        value_stream = value_states[:, :, :update_end].transpose(1, 2)
        memory, _ = recurrent_memory_output(
            query_stream,
            key_stream,
            value_stream,
            beta[:, :update_end],
            alpha[:, :update_end],
        )
        memory = memory[:, sink_len:].reshape(
            query_states.shape[0], outgoing_len, -1
        )
        return self._project_memory(memory.to(self.base_attention.q_proj.weight.dtype)), total_window

    def _zero_state(self, keys, values):
        return torch.zeros(
            keys.shape[0],
            keys.shape[2],
            keys.shape[3],
            values.shape[3],
            device=keys.device,
            dtype=torch.float32,
        )

    def _demand(self, keys, values, beta, alpha, initial_state=None):
        """Return per-head demand and the recurrent state after this segment."""
        if keys.shape[2] != beta.shape[2]:
            groups = beta.shape[2] // keys.shape[2]
            keys = keys.repeat_interleave(groups, dim=2)
            values = values.repeat_interleave(groups, dim=2)
        if keys.shape[1] == 0:
            state = initial_state if initial_state is not None else self._zero_state(keys, values)
            return keys.new_empty((keys.shape[0], 0, keys.shape[2]), dtype=torch.float32), state

        if self.demand_mode == "recurrent":
            chunk_size = max(1, self.recurrent_token_limit)
            state = initial_state
            score_chunks = []
            for start in range(0, keys.shape[1], chunk_size):
                end = min(start + chunk_size, keys.shape[1])
                chunk_scores, state = recurrent_demand(
                    keys[:, start:end],
                    values[:, start:end],
                    beta[:, start:end],
                    alpha[:, start:end],
                    initial_state=state,
                )
                score_chunks.append(chunk_scores)
            return torch.cat(score_chunks, dim=1), state

        # The proxy is deliberately cheap.  It is used for the first hybrid
        # migration probe, not presented as the final trained GRAS signal.
        scores = beta.float() * values.float().norm(dim=-1)
        state = initial_state if initial_state is not None else self._zero_state(keys, values)
        return scores, state

    def _attention_scores(self, query_states, keys):
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

        tail = query_states[:, :, -min(32, query_states.shape[2]):].float()
        query = F.normalize(tail.mean(dim=2), dim=-1)
        key = F.normalize(repeat_kv(keys, self.num_key_value_groups).float(), dim=-1)
        return torch.einsum("bhd,bhld->bhl", query, key).transpose(1, 2)

    def _score_segment(
        self,
        query_states,
        keys,
        values,
        beta,
        alpha,
        initial_state=None,
    ):
        if self.selector_mode == "attention":
            scores = self._attention_scores(query_states, keys)
            _, state = self._demand(keys.transpose(1, 2), values.transpose(1, 2), beta, alpha, initial_state)
            return scores, state
        return self._demand(
            keys.transpose(1, 2), values.transpose(1, 2), beta, alpha, initial_state
        )

    def _aggregate_selection_scores(self, scores):
        if self.selector_head_weights is None:
            return scores.mean(dim=-1)
        weights = self.selector_head_weights.to(device=scores.device, dtype=scores.dtype)
        total = weights.sum()
        if not torch.isfinite(weights).all() or not torch.isfinite(total) or total <= 1e-8:
            return scores.mean(dim=-1)
        weights = weights / total
        return (scores * weights.view(1, 1, -1)).sum(dim=-1)

    def _select_topk(self, scores):
        if self.selector_head_weights is None:
            return select_topk(scores, self.select_budget)
        aggregate = self._aggregate_selection_scores(scores)
        count = min(max(self.select_budget, 0), aggregate.shape[1])
        if count == 0:
            return torch.empty((aggregate.shape[0], 0), dtype=torch.long, device=aggregate.device)
        return aggregate.topk(count, dim=-1, largest=True).indices.sort(dim=-1).values

    def _state_for_cache_indices(
        self,
        keys,
        values,
        beta,
        alpha,
        indices,
        initial_state=None,
    ):
        if indices.numel() == 0:
            return initial_state if initial_state is not None else self._zero_state(keys, values)
        selected_keys = gather_cache_sequence(keys, indices)
        selected_values = gather_cache_sequence(values, indices)
        selected_beta = gather_sequence(beta, indices)
        selected_alpha = gather_sequence(alpha, indices)
        _, state = self._demand(
            selected_keys.transpose(1, 2),
            selected_values.transpose(1, 2),
            selected_beta,
            selected_alpha,
            initial_state,
        )
        return state

    def _set_cache(self, past_key_values, key_states, value_states):
        layer_cache = past_key_values.layers[self.layer_idx]
        layer_cache.keys = key_states.contiguous()
        layer_cache.values = value_states.contiguous()
        layer_cache.dtype = key_states.dtype
        layer_cache.device = key_states.device
        layer_cache.is_initialized = True

    def _cache(self, past_key_values):
        return past_key_values.layers[self.layer_idx]

    def _advance_memory(self, keys, values, beta, alpha):
        if self.demand_mode != "recurrent":
            return
        _, self.mem_state = self._demand(
            keys,
            values,
            beta,
            alpha,
            initial_state=self.mem_state,
        )

    def _memory_read(self, query_states):
        if self.mem_state is None:
            return None
        if not isinstance(self.memory_output_scale, torch.Tensor) and self.memory_output_scale == 0.0:
            return None
        query_states = F.normalize(query_states.float(), dim=-1)
        memory = torch.einsum("bhqk,bhkv->bhqv", query_states, self.mem_state)
        memory = memory / (self.head_dim**0.5)
        memory = memory.transpose(1, 2).reshape(query_states.shape[0], query_states.shape[2], -1)
        return self._project_memory(memory.to(self.base_attention.q_proj.weight.dtype))

    def _sequence_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        use_memory,
    ):
        query_states, key_states, value_states, gate = self._project(
            hidden_states, position_embeddings
        )
        q_len = query_states.shape[2]
        if use_memory:
            attention_mask = self._local_attention_mask(q_len, query_states.device)
            # During adapter training, expose the same global top-K route used
            # at inference. Selection is a frozen demand-based route; gradients
            # continue through the attention and memory readout.
            sink_len = min(self.num_attn_sinks, max(0, q_len - self.sliding_window))
            if self.select_budget > 0 and q_len > sink_len + self.sliding_window:
                beta, alpha = self._gates(gate)
                with torch.no_grad():
                    demand, _ = self._demand(
                        key_states.transpose(1, 2), value_states.transpose(1, 2), beta, alpha
                    )
                    aggregate = self._aggregate_selection_scores(demand)
                    start = sink_len
                    history_length = self.training_context_length or q_len
                    end = max(start, history_length - self.sliding_window)
                    count = min(self.select_budget, max(0, end - start))
                    selected = aggregate[:, start:end].topk(count, dim=1).indices + start
                    selected_mask = torch.zeros((query_states.shape[0], q_len), dtype=torch.bool, device=query_states.device)
                    selected_mask.scatter_(1, selected, True)
                    positions = torch.arange(q_len, device=query_states.device)
                    causal = positions[:, None] >= positions[None, :]
                    sink = positions[None, :] < sink_len
                    recent = positions[None, :] >= positions[:, None] - self.sliding_window + 1
                    allowed = sink | recent | selected_mask[:, None, :]
                    if self.training_context_length is not None:
                        allowed = allowed | (positions[:, None] < self.training_context_length)
                    attention_mask = (causal & allowed).unsqueeze(1)
        attn_output = self._attention(
            query_states, key_states, value_states, attention_mask, q_len
        )
        attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape[0], q_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        if use_memory:
            beta, alpha = self._gates(gate)
            memory_output, start = self._training_memory_output(
                query_states, key_states, value_states, beta, alpha
            )
            if memory_output is not None:
                if self.training_context_length is not None:
                    suffix_start = max(start, self.training_context_length)
                    memory_output = memory_output[:, suffix_start - start:]
                    start = suffix_start
                attn_output = attn_output.clone()
                attn_output[:, start:] = (
                    attn_output[:, start:]
                    + self.memory_output_scale.to(attn_output.dtype) * memory_output
                )
        return self.base_attention.o_proj(attn_output), None

    def _prefill_cache(
        self,
        query_states,
        key_states,
        value_states,
        beta,
        alpha,
        past_key_values,
    ):
        sequence_length = key_states.shape[2]
        window_start = max(0, sequence_length - self.sliding_window)
        sink_len = min(self.num_attn_sinks, window_start)
        self.active_sink_len = sink_len
        if not self.retain_recurrent_state and self.select_budget == 0:
            self._set_cache(
                past_key_values,
                torch.cat(
                    [
                        key_states[:, :, :sink_len],
                        key_states[:, :, window_start:],
                    ],
                    dim=2,
                ),
                torch.cat(
                    [
                        value_states[:, :, :sink_len],
                        value_states[:, :, window_start:],
                    ],
                    dim=2,
                ),
            )
            self.beta_cache = None
            self.alpha_cache = None
            self.mem_state = None
            self.select_positions = []
            self.window_positions = list(range(window_start, sequence_length))
            self.seen_tokens = sequence_length
            return
        candidate_keys = key_states[:, :, sink_len:window_start]
        candidate_values = value_states[:, :, sink_len:window_start]
        candidate_beta = beta[:, sink_len:window_start]
        candidate_alpha = alpha[:, sink_len:window_start]

        sink_keys = key_states[:, :, :sink_len]
        sink_values = value_states[:, :, :sink_len]
        sink_beta = beta[:, :sink_len]
        sink_alpha = alpha[:, :sink_len]
        sink_state = None
        if self.sink_warm_start:
            _, sink_state = self._demand(
                sink_keys.transpose(1, 2),
                sink_values.transpose(1, 2),
                sink_beta,
                sink_alpha,
            )

        if self.selection_include_sinks:
            selection_keys = torch.cat([sink_keys, candidate_keys], dim=2)
            selection_values = torch.cat([sink_values, candidate_values], dim=2)
            selection_beta = torch.cat([sink_beta, candidate_beta], dim=1)
            selection_alpha = torch.cat([sink_alpha, candidate_alpha], dim=1)
            selection_scores, full_state = self._score_segment(
                query_states,
                selection_keys,
                selection_values,
                selection_beta,
                selection_alpha,
            )
            selected_all = self._select_topk(selection_scores)
            selected = selected_all - sink_len
            selected = selected[selected >= 0].view(1, -1)
            candidate_state = full_state
            self.prefill_head_scores = selection_scores.detach().float().cpu()
            self.prefill_scores = self._aggregate_selection_scores(selection_scores).detach().float().cpu()
            self.prefill_score_offset = 0
        else:
            candidate_scores, candidate_state = self._score_segment(
                query_states,
                candidate_keys,
                candidate_values,
                candidate_beta,
                candidate_alpha,
                initial_state=sink_state,
            )
            selected = self._select_topk(candidate_scores)
            if self.force_include_positions or self.force_exclude_positions:
                include = {
                    p - sink_len
                    for p in (self.force_include_positions or set())
                    if sink_len <= p < window_start
                }
                exclude = {
                    p - sink_len
                    for p in (self.force_exclude_positions or set())
                    if sink_len <= p < window_start
                }
                include -= exclude
                include = {p for p in include if 0 <= p < candidate_keys.shape[2]}
                ranked = [int(i) for i in torch.argsort(self._aggregate_selection_scores(candidate_scores)[0], descending=True).tolist()]
                chosen = list(sorted(include))
                for idx in ranked:
                    if idx in include or idx in exclude:
                        continue
                    if len(chosen) >= min(self.select_budget, candidate_keys.shape[2]):
                        break
                    chosen.append(idx)
                selected = torch.tensor(chosen, device=candidate_keys.device, dtype=torch.long).view(1, -1)
            self.prefill_head_scores = candidate_scores.detach().float().cpu()
            self.prefill_scores = self._aggregate_selection_scores(candidate_scores).detach().float().cpu()
            self.prefill_score_offset = sink_len

        if candidate_keys.shape[2] > 0:
            select_keys = gather_cache_sequence(candidate_keys, selected)
            select_values = gather_cache_sequence(candidate_values, selected)
            selected_beta = gather_sequence(candidate_beta, selected)
            selected_alpha = gather_sequence(candidate_alpha, selected)
            self.last_demand = self.prefill_scores
            self.select_positions = (selected[0] + sink_len).detach().cpu().tolist()
        else:
            select_keys = candidate_keys[:, :, :0]
            select_values = candidate_values[:, :, :0]
            selected_beta = candidate_beta[:, :0]
            selected_alpha = candidate_alpha[:, :0]
            self.last_demand = torch.empty((1, 0), dtype=torch.float32)
            self.select_positions = []

        window_keys = key_states[:, :, window_start:]
        window_values = value_states[:, :, window_start:]
        self._set_cache(
            past_key_values,
            torch.cat([sink_keys, select_keys, window_keys], dim=2),
            torch.cat([sink_values, select_values, window_values], dim=2),
        )
        self.beta_cache = torch.cat(
            [sink_beta, selected_beta, beta[:, window_start:]], dim=1
        ).contiguous()
        self.alpha_cache = torch.cat(
            [sink_alpha, selected_alpha, alpha[:, window_start:]], dim=1
        ).contiguous()
        self.window_positions = list(range(window_start, sequence_length))
        self.seen_tokens = sequence_length
        if not self.retain_recurrent_state:
            self.beta_cache = None
            self.alpha_cache = None
            self.mem_state = None
            return
        if self.demand_mode == "recurrent":
            candidate_indices = torch.arange(candidate_keys.shape[2], device=key_states.device).view(1, -1)
            if sink_len == 0 and candidate_keys.shape[2] == 0:
                self.mem_state = None
            elif self.memory_write_mode == "all" and not self.force_state_exclude_positions:
                self.mem_state = candidate_state
            else:
                selected_mask = torch.zeros_like(candidate_indices, dtype=torch.bool)
                if selected.numel():
                    selected_mask.scatter_(1, selected, True)
                if self.memory_write_mode == "selected_only":
                    write_indices = selected
                elif self.memory_write_mode == "exclude_selected":
                    write_indices = candidate_indices[~selected_mask].view(1, -1)
                elif self.memory_write_mode == "all":
                    excluded = {
                        p - sink_len
                        for p in (self.force_state_exclude_positions or set())
                        if sink_len <= p < window_start
                    }
                    write_indices = torch.tensor(
                        [i for i in candidate_indices[0].tolist() if i not in excluded],
                        device=key_states.device,
                        dtype=torch.long,
                    ).view(1, -1)
                else:
                    raise ValueError(f"unknown memory_write_mode: {self.memory_write_mode}")
                self.mem_state = self._state_for_cache_indices(
                    candidate_keys,
                    candidate_values,
                    candidate_beta,
                    candidate_alpha,
                    write_indices,
                    sink_state,
                )
        else:
            self.mem_state = None

    def compress_deferred(self, past_key_values):
        """Compress a chunk-prefilled full cache at the history boundary."""
        if not self.defer_compression:
            return
        layer_cache = self._cache(past_key_values)
        if layer_cache.keys is None or self._deferred_query_states is None:
            return
        self.defer_compression = False
        self._prefill_cache(
            self._deferred_query_states,
            layer_cache.keys,
            layer_cache.values,
            self.beta_cache,
            self.alpha_cache,
            past_key_values,
        )
        self._deferred_query_states = None

    def _decode_cache(
        self,
        key_states,
        value_states,
        beta,
        alpha,
        query_states,
        past_key_values,
        cache_position,
    ):
        layer_cache = self._cache(past_key_values)
        old_keys, old_values = layer_cache.keys, layer_cache.values
        new_keys = torch.cat([old_keys, key_states], dim=2)
        new_values = torch.cat([old_values, value_states], dim=2)
        if self.retain_recurrent_state:
            self.beta_cache = torch.cat([self.beta_cache, beta], dim=1)
            self.alpha_cache = torch.cat([self.alpha_cache, alpha], dim=1)
        current_position = (
            int(cache_position[-1].item()) if cache_position is not None else self.seen_tokens
        )
        self.window_positions.append(current_position)

        outgoing = len(self.window_positions) > self.sliding_window
        outgoing_index = self.active_sink_len + len(self.select_positions)
        if outgoing:
            # When a short prefill grows beyond W, promote the oldest window
            # tokens to the sink until the configured sink budget is filled.
            # They remain exact KV; only tokens after the sink is full are
            # evicted into the recurrent state.
            if self.active_sink_len < self.num_attn_sinks:
                if self.sink_warm_start and self.retain_recurrent_state:
                    self._advance_memory(
                        new_keys[:, :, outgoing_index : outgoing_index + 1].transpose(1, 2),
                        new_values[:, :, outgoing_index : outgoing_index + 1].transpose(1, 2),
                        self.beta_cache[:, outgoing_index : outgoing_index + 1],
                        self.alpha_cache[:, outgoing_index : outgoing_index + 1],
                    )
                self.active_sink_len += 1
                self.window_positions.pop(0)
                self._set_cache(past_key_values, new_keys, new_values)
                self.seen_tokens = current_position + 1
                return
            outgoing_keys = new_keys[:, :, outgoing_index : outgoing_index + 1]
            outgoing_values = new_values[:, :, outgoing_index : outgoing_index + 1]
            if self.retain_recurrent_state:
                outgoing_beta = self.beta_cache[:, outgoing_index : outgoing_index + 1]
                outgoing_alpha = self.alpha_cache[:, outgoing_index : outgoing_index + 1]
                self._advance_memory(
                    outgoing_keys.transpose(1, 2),
                    outgoing_values.transpose(1, 2),
                    outgoing_beta,
                    outgoing_alpha,
                )
            new_keys = torch.cat(
                [new_keys[:, :, :outgoing_index], new_keys[:, :, outgoing_index + 1 :]], dim=2
            ).contiguous()
            new_values = torch.cat(
                [new_values[:, :, :outgoing_index], new_values[:, :, outgoing_index + 1 :]], dim=2
            ).contiguous()
            if self.retain_recurrent_state:
                self.beta_cache = torch.cat(
                    [self.beta_cache[:, :outgoing_index], self.beta_cache[:, outgoing_index + 1 :]], dim=1
                ).contiguous()
                self.alpha_cache = torch.cat(
                    [self.alpha_cache[:, :outgoing_index], self.alpha_cache[:, outgoing_index + 1 :]], dim=1
                ).contiguous()
            self.window_positions.pop(0)

        self._set_cache(past_key_values, new_keys, new_values)
        self.seen_tokens = current_position + 1

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is None:
            return self._sequence_forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                use_memory=self.training and self.train_memory,
            )

        query_states, key_states, value_states, gate = self._project(
            hidden_states, position_embeddings
        )
        q_len = query_states.shape[2]
        layer_cache = self._cache(past_key_values)
        old_length = layer_cache.get_seq_length()

        if self.defer_compression:
            old_keys, old_values = layer_cache.keys, layer_cache.values
            if old_length:
                all_keys = torch.cat([old_keys, key_states], dim=2)
                all_values = torch.cat([old_values, value_states], dim=2)
                from torch.nn.attention.bias import causal_lower_right
                attention_mask = causal_lower_right(q_len, old_length + q_len)
            else:
                all_keys, all_values = key_states, value_states
            attn_output = self._attention(query_states, all_keys, all_values, attention_mask, q_len)
            beta, alpha = self._gates(gate)
            self._set_cache(past_key_values, all_keys, all_values)
            self.beta_cache = beta if old_length == 0 else torch.cat([self.beta_cache, beta], dim=1)
            self.alpha_cache = alpha if old_length == 0 else torch.cat([self.alpha_cache, alpha], dim=1)
            self._deferred_query_states = query_states
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape[0], q_len, -1)
            attn_output = attn_output * torch.sigmoid(gate)
            return self.base_attention.o_proj(attn_output), None

        if q_len > 1 or old_length == 0:
            if old_length != 0:
                raise ValueError("Qwen3.5 GRAS adapter currently supports one-shot prefill only")
            attn_output = self._attention(
                query_states, key_states, value_states, attention_mask, q_len
            )
            beta, alpha = self._gates(gate)
            self._prefill_cache(
                query_states,
                key_states,
                value_states,
                beta,
                alpha,
                past_key_values,
            )
            memory_output = None
        else:
            attn_output = self._attention(
                query_states, 
                torch.cat([layer_cache.keys, key_states], dim=2),
                torch.cat([layer_cache.values, value_states], dim=2),
                None,
                q_len,
            )
            beta, alpha = self._gates(gate)
            self._decode_cache(
                key_states,
                value_states,
                beta,
                alpha,
                query_states,
                past_key_values,
                kwargs.get("cache_position"),
            )
            memory_output = self._memory_read(query_states)

        attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape[0], q_len, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        if memory_output is not None:
            scale = (
                self.memory_output_scale.to(attn_output.dtype)
                if isinstance(self.memory_output_scale, torch.Tensor)
                else self.memory_output_scale
            )
            attn_output = attn_output + scale * memory_output.to(attn_output.dtype)
        return self.base_attention.o_proj(attn_output), None


def install_gras_hmc(
    model,
    select_budget: int = 512,
    sliding_window: int = 4096,
    num_attn_sinks: int = 128,
    layer_budgets: Optional[list[int]] = None,
    demand_mode: str = "proxy",
    recurrent_token_limit: int = 2048,
    memory_output_scale: float = 0.0,
    train_memory: bool = False,
    memory_projection_rank: int = 0,
    sink_warm_start: bool = True,
    selection_include_sinks: bool = False,
    memory_write_mode: str = "all",
    selector_mode: str = "gras",
    retain_recurrent_state: bool = True,
):
    """Patch only Qwen3.5 full-attention layers and return their wrappers."""
    layers = model.model.layers
    full_indices = [i for i, layer in enumerate(layers) if layer.block_type == "full_attention"]
    wrappers = []
    for full_order, layer_idx in enumerate(full_indices):
        if layer_budgets is None:
            budget = select_budget
        elif len(layer_budgets) == len(layers):
            budget = layer_budgets[layer_idx]
        elif len(layer_budgets) == len(full_indices):
            budget = layer_budgets[full_order]
        else:
            raise ValueError("layer budget count must match all or full-attention layers")
        layer = layers[layer_idx]
        wrapper = GRASQwen35Attention(
            layer.self_attn,
            model.model.config,
            layer_idx,
            budget,
            sliding_window,
            num_attn_sinks,
            demand_mode=demand_mode,
            recurrent_token_limit=recurrent_token_limit,
            memory_output_scale=memory_output_scale,
            train_memory=train_memory,
            memory_projection_rank=memory_projection_rank,
            sink_warm_start=sink_warm_start,
            selection_include_sinks=selection_include_sinks,
            memory_write_mode=memory_write_mode,
            selector_mode=selector_mode,
            retain_recurrent_state=retain_recurrent_state,
        )
        layer.self_attn = wrapper
        wrappers.append(wrapper)
    model.gras_full_attention_layers = wrappers
    model.gras_full_attention_indices = full_indices
    return wrappers


def load_memory_adapter(wrappers, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    layers = payload.get("layers", payload)
    for wrapper in wrappers:
        wrapper.load_state_dict(layers[str(wrapper.layer_idx)], strict=False)
