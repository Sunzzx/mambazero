"""
Tests for MambaLM model and generation.
"""

import pytest
import numpy as np
from mambazero.model import MambaLM


def test_mambalm_forward_vs_step():
    """MambaLM parallel forward logits must match step-by-step logits."""
    rng = np.random.default_rng(42)
    model = MambaLM(vocab_size=32, d_model=16, n_layers=2, d_state=8, rng=rng)
    
    ids = rng.integers(0, 32, size=(2, 10))
    logits_seq = model.forward_sequence(ids)
    
    states = None
    outs = []
    for t in range(10):
        l_t, states = model.step(ids[:, t], states)
        outs.append(l_t)
    logits_step = np.stack(outs, axis=1)
    
    assert np.allclose(logits_seq, logits_step, atol=1e-12)


def test_mambalm_generation():
    """Test autoregressive token generation."""
    rng = np.random.default_rng(42)
    model = MambaLM(vocab_size=64, d_model=16, n_layers=2, d_state=8, rng=rng)
    
    prompt = np.array([[10, 11, 12]], dtype=np.int64)
    gen = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=5)
    
    assert gen.shape == (1, 8)
    assert np.array_equal(gen[0, :3], prompt[0])
    assert np.all(gen >= 0) and np.all(gen < 64)
