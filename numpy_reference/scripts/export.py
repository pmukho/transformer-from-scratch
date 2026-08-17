"""Script: export canonical weights and golden fixtures for the Phase B C++ engine.

Not part of the reference implementation package. It uses numpy_reference to build
a frozen config plus random float32 weights, runs the float32 reference forward
(prefill and token-by-token greedy decode), and writes everything to .npz under
weights/<preset>/. The C++ engine loads model.npz and checks parity against
fixtures.npz.

Two presets:
  * tiny  - small, for parity/debugging (matmuls are cheap and easy to reason about)
  * base  - bigger, for benchmarking, where SIMD and threading actually show up and
            NumPy vs C++ scalar vs C++ parallel diverge meaningfully

Regenerate with:  uv run python scripts/export.py [tiny] [base]

Two details make the float32 handoff work:
  * The whole reference runs in float32 (including float32 cos/sin tables), so the
    saved logits are true float32, matching what the C++ engine computes.
  * Weights are scaled by 1/sqrt(fan_in). This keeps activations O(1) so the naive
    C++ matmul (a different summation order than NumPy) still lands within 1e-5.
    Unscaled N(0,1) weights would grow the residual stream to ~100s and blow past
    that tolerance for pure float32 rounding reasons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from numpy_reference import model, rope

PRESETS = {
    "tiny": {
        "dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "kv_heads": 4,
        "head_dim": 16,
        "ffn_hidden": 172,
        "vocab": 256,
        "rope_base": 10000.0,
        "eps": 1e-6,
    },
    "base": {
        "dim": 256,
        "n_layers": 6,
        "n_heads": 8,
        "kv_heads": 8,
        "head_dim": 32,
        "ffn_hidden": 688,
        "vocab": 4096,
        "rope_base": 10000.0,
        "eps": 1e-6,
    },
}

PROMPT_LEN = 8
MAX_NEW = 8
WEIGHT_SEED = 0
TOKEN_SEED = 1

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"


def build_weights(cfg, seed=WEIGHT_SEED):
    """Random float32 weights, scaled by 1/sqrt(fan_in); norm gains set to one."""
    rng = np.random.default_rng(seed)
    dim, hidden = cfg["dim"], cfg["ffn_hidden"]
    n_heads, kv_heads, head_dim = cfg["n_heads"], cfg["kv_heads"], cfg["head_dim"]
    qkv_out = n_heads * head_dim
    kv_out = kv_heads * head_dim

    def w(fan_in, *shape):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    blocks = []
    for _ in range(cfg["n_layers"]):
        blocks.append(
            {
                "attn_norm": np.ones(dim, dtype=np.float32),
                "attn": {
                    "wq": w(dim, dim, qkv_out),
                    "wk": w(dim, dim, kv_out),
                    "wv": w(dim, dim, kv_out),
                    "wo": w(qkv_out, qkv_out, dim),
                },
                "ffn_norm": np.ones(dim, dtype=np.float32),
                "ffn": {
                    "w_gate": w(dim, dim, hidden),
                    "w_up": w(dim, dim, hidden),
                    "w_down": w(hidden, hidden, dim),
                },
            }
        )
    return {
        "embedding": w(dim, cfg["vocab"], dim),
        "blocks": blocks,
        "final_norm": np.ones(dim, dtype=np.float32),
    }


def run_reference(cfg, weights, tokens):
    """Run prefill then greedy decode in float32. Returns the golden arrays.

    tokens is (prompt_len,). Batch is 1 internally and squeezed out on return.

    Returns:
        prefill_logits: (prompt_len, vocab)
        next_logits: (max_new, vocab), the logits each generated token is argmaxed from
        gen_tokens: (max_new,)
    """
    n_heads, kv_heads, head_dim = cfg["n_heads"], cfg["kv_heads"], cfg["head_dim"]
    cos, sin = rope.precompute_cos_sin(PROMPT_LEN + MAX_NEW, head_dim, base=cfg["rope_base"])
    cos, sin = cos.astype(np.float32), sin.astype(np.float32)

    prompt = tokens[None, :]  # (1, prompt_len)
    prefill_logits, past = model.forward(
        prompt, weights, cos, sin, n_heads=n_heads, kv_heads=kv_heads
    )

    next_logits = [prefill_logits[:, -1, :]]  # (1, vocab)
    cur = np.argmax(next_logits[0], axis=-1)  # (1,)
    gen = [cur]
    for _ in range(MAX_NEW - 1):
        step_logits, past = model.forward(
            cur[:, None], weights, cos, sin, past, n_heads=n_heads, kv_heads=kv_heads
        )
        last = step_logits[:, -1, :]
        next_logits.append(last)
        cur = np.argmax(last, axis=-1)
        gen.append(cur)

    return (
        prefill_logits[0].astype(np.float32),
        np.concatenate(next_logits, axis=0).astype(np.float32),
        np.concatenate(gen, axis=0).astype(np.int32),
    )


def flatten_weights(cfg, weights):
    """Nested weight dict -> flat {key: array} for .npz, plus config scalars."""
    flat = {
        "embedding": weights["embedding"],
        "final_norm": weights["final_norm"],
    }
    for i, blk in enumerate(weights["blocks"]):
        flat[f"blocks.{i}.attn_norm"] = blk["attn_norm"]
        for k in ("wq", "wk", "wv", "wo"):
            flat[f"blocks.{i}.attn.{k}"] = blk["attn"][k]
        flat[f"blocks.{i}.ffn_norm"] = blk["ffn_norm"]
        for k in ("w_gate", "w_up", "w_down"):
            flat[f"blocks.{i}.ffn.{k}"] = blk["ffn"][k]

    int_keys = ("dim", "n_layers", "n_heads", "kv_heads", "head_dim", "ffn_hidden", "vocab")
    for k in int_keys:
        flat[f"config.{k}"] = np.int32(cfg[k])
    # float64 so they round-trip exactly; the C++ loader casts to float on use
    flat["config.rope_base"] = np.float64(cfg["rope_base"])
    flat["config.eps"] = np.float64(cfg["eps"])
    return flat


def load_model(path):
    """Inverse of the flat .npz layout: returns (cfg, weights). Mirrors the C++ loader."""
    data = np.load(path)
    int_keys = ("dim", "n_layers", "n_heads", "kv_heads", "head_dim", "ffn_hidden", "vocab")
    cfg = {k: int(data[f"config.{k}"]) for k in int_keys}
    cfg["rope_base"] = float(data["config.rope_base"])
    cfg["eps"] = float(data["config.eps"])

    blocks = []
    for i in range(cfg["n_layers"]):
        blocks.append(
            {
                "attn_norm": data[f"blocks.{i}.attn_norm"],
                "attn": {k: data[f"blocks.{i}.attn.{k}"] for k in ("wq", "wk", "wv", "wo")},
                "ffn_norm": data[f"blocks.{i}.ffn_norm"],
                "ffn": {k: data[f"blocks.{i}.ffn.{k}"] for k in ("w_gate", "w_up", "w_down")},
            }
        )
    weights = {
        "embedding": data["embedding"],
        "blocks": blocks,
        "final_norm": data["final_norm"],
    }
    return cfg, weights


def export(preset="tiny", out_root=WEIGHTS_DIR):
    """Build weights + fixtures for one preset and write them to <out_root>/<preset>/."""
    cfg = PRESETS[preset]
    out_dir = Path(out_root) / preset
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = build_weights(cfg)
    tokens = (
        np.random.default_rng(TOKEN_SEED).integers(0, cfg["vocab"], size=PROMPT_LEN).astype(np.int32)
    )
    prefill_logits, next_logits, gen_tokens = run_reference(cfg, weights, tokens)

    np.savez(out_dir / "model.npz", **flatten_weights(cfg, weights))
    np.savez(
        out_dir / "fixtures.npz",
        tokens=tokens,
        prefill_logits=prefill_logits,
        next_logits=next_logits,
        gen_tokens=gen_tokens,
    )
    return out_dir


if __name__ == "__main__":
    names = sys.argv[1:] or list(PRESETS)
    for name in names:
        path = export(name)
        print(f"[{name}] wrote model.npz + fixtures.npz to {path}")
