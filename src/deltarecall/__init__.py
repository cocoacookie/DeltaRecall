"""DeltaRecall: write-demand-driven plug-and-play hybrid memory for long-context LLMs.

DeltaRecall attaches a three-resolution memory to the *frozen* full-attention
layers of a pretrained hybrid Transformer (Qwen3.5), leaving the native
linear-attention / Gated DeltaNet layers untouched:

* an **Active Cache** for attention sinks and the recent window,
* a **Selective Cache** for a few remote exact KV entries, and
* a **Compressive Memory** that covers the growing history with a fixed-size
  Gated DeltaNet state.

The public entry points mirror the plug-and-play usage described in the paper::

    from transformers import AutoModelForCausalLM
    from deltarecall import install_gras_hmc, load_memory_adapter

    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    wrappers = install_gras_hmc(
        model,
        select_budget=6016, sliding_window=2048, num_attn_sinks=128,
        demand_mode="recurrent", memory_projection_rank=64,
        memory_output_scale=0.1,
    )
    load_memory_adapter(wrappers, "model/deltarecall-9b/memory_adapter.pt")
"""

from .qwen35_gras_hmc import (
    GRASQwen35Attention,
    install_gras_hmc,
    load_memory_adapter,
)

__all__ = [
    "GRASQwen35Attention",
    "install_gras_hmc",
    "load_memory_adapter",
]

__version__ = "0.1.0"
