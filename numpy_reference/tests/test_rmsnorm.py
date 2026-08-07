"""RMSNorm parity tests against torch as the oracle.

torch is a test-only dependency. These checks are part of the Phase A oracle:
Phase B (C++) must reproduce the same outputs to within 1e-5.
"""

import numpy as np
import torch

from numpy_reference import rmsnorm


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_matches_torch_rms_norm():
    """forward matches torch.nn.functional.rms_norm for a batched input."""
    dim = 64
    x = _rand(4, 10, dim, seed=0)
    weight = _rand(dim, seed=1)
    eps = 1e-6

    got = rmsnorm.forward(x, weight, eps=eps)
    ref = torch.nn.functional.rms_norm(
        torch.from_numpy(x), (dim,), weight=torch.from_numpy(weight), eps=eps
    ).numpy()

    assert np.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_matches_torch_unit_weight():
    """Same, with weight = ones, so the check isolates the normalization itself."""
    dim = 32
    x = _rand(7, dim, seed=2)
    weight = np.ones(dim, dtype=np.float32)
    eps = 1e-5

    got = rmsnorm.forward(x, weight, eps=eps)
    ref = torch.nn.functional.rms_norm(
        torch.from_numpy(x), (dim,), weight=torch.from_numpy(weight), eps=eps
    ).numpy()

    assert np.allclose(got, ref, atol=1e-5, rtol=1e-5)
