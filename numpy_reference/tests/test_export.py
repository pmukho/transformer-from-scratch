"""Round-trip test for the weight/fixture export script.

Exports the tiny preset to a temp dir, reloads via load_model, and re-runs the
reference to confirm the flat .npz layout and the golden fixtures are consistent.
Keeps the Phase A -> Phase B handoff honest.
"""

import numpy as np

import export


def test_export_roundtrip(tmp_path):
    out = export.export("tiny", out_root=tmp_path)

    cfg, weights = export.load_model(out / "model.npz")
    assert cfg == export.PRESETS["tiny"]

    fx = np.load(out / "fixtures.npz")
    prefill_logits, next_logits, gen_tokens = export.run_reference(cfg, weights, fx["tokens"])

    np.testing.assert_array_equal(gen_tokens, fx["gen_tokens"])
    np.testing.assert_allclose(prefill_logits, fx["prefill_logits"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(next_logits, fx["next_logits"], atol=1e-6, rtol=1e-6)
