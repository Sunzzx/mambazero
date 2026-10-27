"""
Comprehensive Unit & Property Tests for Parallel Associative Scan.
"""

import pytest
import numpy as np
from mambazero.scan import associative_op, sequential_scan, parallel_associative_scan, blelloch_scan


def test_associative_operator_property():
    """Test that (a3, b3) ∘ ((a2, b2) ∘ (a1, b1)) == ((a3, b3) ∘ (a2, b2)) ∘ (a1, b1)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        e1 = (rng.uniform(0.1, 2.0), rng.normal(0, 1))
        e2 = (rng.uniform(0.1, 2.0), rng.normal(0, 1))
        e3 = (rng.uniform(0.1, 2.0), rng.normal(0, 1))
        
        # Left grouping
        e12 = associative_op(e2, e1)
        left = associative_op(e3, e12)
        
        # Right grouping
        e23 = associative_op(e3, e2)
        right = associative_op(e23, e1)
        
        assert np.isclose(left[0], right[0], atol=1e-14)
        assert np.isclose(left[1], right[1], atol=1e-14)


@pytest.mark.parametrize("seq_len", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100])
def test_parallel_vs_sequential_scan(seq_len):
    """Test numerical equivalence across power-of-two and non-power-of-two lengths."""
    rng = np.random.default_rng(seq_len)
    batch_size = 3
    d_model = 8
    d_state = 4
    
    a = rng.uniform(0.8, 0.99, size=(batch_size, seq_len, d_model, d_state))
    b = rng.normal(0.0, 1.0, size=(batch_size, seq_len, d_model, d_state))
    
    seq_out = sequential_scan(a, b)
    par_out = parallel_associative_scan(a, b)
    blelloch_out = blelloch_scan(a, b)
    
    assert np.allclose(seq_out, par_out, atol=1e-12)
    assert np.allclose(seq_out, blelloch_out, atol=1e-12)


def test_scan_with_initial_hidden_state():
    """Test recurrence with non-zero initial hidden state."""
    rng = np.random.default_rng(123)
    L = 12
    a = rng.uniform(0.8, 0.99, size=(2, L, 4, 4))
    b = rng.normal(0, 1, size=(2, L, 4, 4))
    h_init = rng.normal(0, 1, size=(2, 4, 4))
    
    seq_out = sequential_scan(a, b, h_init=h_init)
    par_out = parallel_associative_scan(a, b, h_init=h_init)
    
    assert np.allclose(seq_out, par_out, atol=1e-12)
