# NumPy reference

Pure-NumPy reference implementation of a decoder-only, Llama-style transformer
(RMSNorm, RoPE, SwiGLU, causal attention, no biases) with a KV cache and greedy
autoregressive generation. No ML frameworks. This is the correctness oracle for
every later phase.

### Layout

```
numpy_reference/
├── pyproject.toml            # uv project (numpy runtime dep, pytest dev dep)
├── .python-version           # pinned to 3.12
├── src/numpy_reference/      # implementation modules
└── tests/                    # pytest correctness suite (the oracle)
    weights/                  # saved .npz weights (gitignored — regenerable)
```

### Modules

Each module exposes a `forward(x, weights)` function

| File | Responsibility |
|------|----------------|
| `rmsnorm.py`    | RMSNorm — no mean-centering, no bias (contrast with LayerNorm) |
| `rope.py`       | Rotary positional embedding — precompute cos/sin tables, rotate Q/K |
| `attention.py`  | Causal scaled-dot-product attention, KV-cache-aware (`past_kv` in/out) |
| `multihead.py`  | Head split/merge + output proj; GQA-capable, run as MHA (`kv_heads = n_heads`) first |
| `swiglu.py`     | SwiGLU FFN — `down(silu(gate(x)) * up(x))`, three matrices, no bias |
| `block.py`      | Decoder block: pre-norm → attn → residual → pre-norm → SwiGLU → residual |
| `model.py`      | Token embedding → N blocks → final RMSNorm → `lm_head` (tied to embedding) |
| `generate.py`   | Greedy autoregressive loop driving the KV cache incrementally |

### Setup & running

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
cd numpy_reference
uv sync              # create .venv and install deps
uv run pytest        # run the correctness suite
uv run python -c "import numpy_reference"   # sanity check
```
