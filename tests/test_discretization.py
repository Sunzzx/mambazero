"""
Tests for Continuous-to-Discrete Discretization Routines.
"""

import pytest
import numpy as np
from mambazero.discretization import discretize_zoh_diagonal, discretize_bilinear_diagonal, discretize_euler_diagonal, expm1_div_z


def test_expm1_div_z_numerical_stability():
    """Test that expm1_div_z handles both normal and infinitesimal numbers without NaN or Inf."""
    z = np.array([1.0, 0.1, 1e-4, 1e-7, 1e-15, -1e-15, 0.0])
    res = expm1_div_z(z)
    assert not np.isnan(res).any()
    assert not np.isinf(res).any()
    # At z=0, limit is 1.0
    assert np.isclose(res[-1], 1.0, atol=1e-7)
    assert np.isclose(res[0], np.expm1(1.0), atol=1e-7)


def test_discretize_zoh_decay():
    """Under negative A, discrete A_bar must be strictly in (0, 1) for contractive stability."""
    delta = np.array([0.01, 0.05, 0.1])
    A = np.array([-1.0, -5.0, -10.0])
    B = np.array([1.0, 1.0, 1.0])
    
    A_bar, B_bar = discretize_zoh_diagonal(delta, A, B)
    assert np.all(A_bar > 0.0)
    assert np.all(A_bar < 1.0)
    assert np.all(B_bar > 0.0)


def test_bilinear_vs_zoh_small_step():
    """For tiny delta, ZOH, Bilinear, and Euler discretization should converge."""
    delta = np.array([1e-5])
    A = np.array([-2.0])
    B = np.array([1.0])
    
    A_zoh, B_zoh = discretize_zoh_diagonal(delta, A, B)
    A_bil, B_bil = discretize_bilinear_diagonal(delta, A, B)
    A_eul, B_eul = discretize_euler_diagonal(delta, A, B)
    
    assert np.allclose(A_zoh, A_bil, atol=1e-7)
    assert np.allclose(A_zoh, A_eul, atol=1e-7)
    assert np.allclose(B_zoh, B_bil, atol=1e-7)
