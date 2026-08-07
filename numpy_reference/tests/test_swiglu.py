"""SwiGLU parity tests against torch as the oracle.

The FFN is fully per-token (no sequence mixing, no cache), so a single batched
forward covers it. torch is a test-only dependency.
"""

import numpy as np
import torch
import torch.nn.functional as F

from numpy_reference import swiglu


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _weights(dim, hidden, seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return {
        "w_gate": r(dim, hidden),
        "w_up": r(dim, hidden),
        "w_down": r(hidden, dim),
    }


def _torch_swiglu(x, weights):
    tx = torch.from_numpy(x)
    wg = torch.from_numpy(weights["w_gate"])
    wu = torch.from_numpy(weights["w_up"])
    wd = torch.from_numpy(weights["w_down"])
    hidden = F.silu(tx @ wg) * (tx @ wu)
    return (hidden @ wd).numpy()


def test_matches_torch():
    """Batched forward matches torch silu(gate) * up projected by w_down."""
    dim, hidden = 32, 64
    x = _rand(2, 6, dim, seed=0)
    weights = _weights(dim, hidden, seed=1)

    got = swiglu.forward(x, weights)
    ref = _torch_swiglu(x, weights)

    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_matches_torch_wide_hidden():
    """Same, with a wider hidden dimension and a different batch shape."""
    dim, hidden = 16, 128
    x = _rand(4, dim, seed=2)
    weights = _weights(dim, hidden, seed=3)

    got = swiglu.forward(x, weights)
    ref = _torch_swiglu(x, weights)

    np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)
