"""
Tests for Selective State Space Model (SSM).
"""

import pytest
import numpy as np
from mambazero.ssm import SelectiveSSM


def test_ssm_parallel_recurrent_equivalence():
    """SelectiveSSM parallel scan must produce identical output to recurrent step inference."""
    rng = np.random.default_rng(99)
    ssm = SelectiveSSM(d_model=12, d_state=16, dt_rank=4, rng=rng)
    
    B_batch, L, D = 2, 16, 12
    u = rng.normal(0, 1, size=(B_batch, L, D))
    
    # Parallel forward pass
    y_par, h_final_par = ssm.forward_parallel(u)
    
    # Recurrent step pass
    h_st = None
    y_steps = []
    for t in range(L):
        y_t, h_st = ssm.step(u[:, t, :], h_st)
        y_steps.append(y_t)
    y_rec = np.stack(y_steps, axis=1)
    
    assert np.allclose(y_par, y_rec, atol=1e-11)
    assert np.allclose(h_final_par, h_st, atol=1e-11)


def test_selection_mechanism_input_dependence():
    """Verify that differing inputs create differing selective delta and state updates."""
    rng = np.random.default_rng(42)
    ssm = SelectiveSSM(d_model=8, d_state=8, rng=rng)
    
    u1 = np.ones((1, 8)) * 0.1
    u2 = np.ones((1, 8)) * 5.0
    
    y1, h1 = ssm.step(u1)
    y2, h2 = ssm.step(u2)
    
    assert not np.allclose(h1, h2)
    assert not np.allclose(y1, y2)
