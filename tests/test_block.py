"""
Tests for MambaBlock architecture.
"""

import pytest
import numpy as np
from mambazero.block import MambaBlock


def test_mamba_block_causality():
    """Modifying token at position t > 0 must not affect output at positions < t."""
    rng = np.random.default_rng(77)
    block = MambaBlock(d_model=8, d_state=8, d_conv=4, expand=2, rng=rng)
    
    x1 = rng.normal(0, 1, size=(1, 10, 8))
    x2 = np.array(x1, copy=True)
    # Mutate position 5 and onward
    x2[0, 5:, :] += rng.normal(0, 1, size=(5, 8))
    
    out1, _ = block.forward_sequence(x1)
    out2, _ = block.forward_sequence(x2)
    
    # Positions 0..4 must be strictly identical
    assert np.allclose(out1[0, :5, :], out2[0, :5, :], atol=1e-14)
    # Positions 5..9 must differ
    assert not np.allclose(out1[0, 5:, :], out2[0, 5:, :])


def test_mamba_block_sequence_vs_step():
    """Sequence forward and step cache must match exactly."""
    rng = np.random.default_rng(101)
    block = MambaBlock(d_model=16, d_state=8, d_conv=4, expand=2, rng=rng)
    
    B, L, D = 2, 14, 16
    x = rng.normal(0, 1, size=(B, L, D))
    
    out_seq, final_st_seq = block.forward_sequence(x)
    
    outs = []
    st = None
    for t in range(L):
        ot, st = block.step(x[:, t, :], st)
        outs.append(ot)
    out_step = np.stack(outs, axis=1)
    
    assert np.allclose(out_seq, out_step, atol=1e-12)
    assert np.allclose(final_st_seq[0], st[0], atol=1e-12)  # conv buffer
    assert np.allclose(final_st_seq[1], st[1], atol=1e-12)  # ssm state
