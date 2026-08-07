"""root-mean-square layer normalization.

Definition, normalizing over the last (feature) axis of size ``dim``:

    rms(x) = sqrt( mean_i(x_i^2) + eps )
    y      = (x / rms(x)) * weight

where ``mean_i`` averages over the feature axis and ``weight`` (gamma) is the
learned per-feature gain.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def forward(
    x: NDArray[np.floating],
    weight: NDArray[np.floating],
    *,
    eps: float = 1e-6,
) -> NDArray[np.floating]:
    """Apply RMSNorm over the last axis of ``x``.

    Args:
        x: Input activations, shape ``(..., dim)``. Any number of leading
            (batch / sequence) dimensions; normalization is over the final axis.
        weight: Learned per-feature gain (gamma), shape ``(dim,)``. Broadcast
            against the last axis of ``x``.
        eps: Numerical-stability term added inside the square root. Llama uses
            1e-5 or 1e-6 depending on the variant; keep this configurable so the
            Phase B C++ engine can match the exact value bit-for-bit.

    Returns:
        Normalized, gain-scaled activations, same shape and dtype-family as ``x``.
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight
