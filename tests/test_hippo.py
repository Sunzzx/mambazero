"""
Tests for HiPPO Continuous-Time Memory & Orthogonal Polynomials.
"""

import pytest
import numpy as np
from mambazero.hippo import make_hippo_legs, make_s4d_diagonal, legendre_polynomials, reconstruct_signal_from_state


def test_hippo_legs_hurwitz_stability():
    """HiPPO continuous transition matrix must have strictly negative real eigenvalues."""
    for N in [4, 8, 16, 32]:
        A, B = make_hippo_legs(N)
        assert A.shape == (N, N)
        assert B.shape == (N, 1)
        eigs = np.linalg.eigvals(A)
        assert np.all(np.real(eigs) < -1e-6)


def test_s4d_diagonal_modes():
    """Test S4D diagonal generation modes."""
    N = 16
    s4d_real = make_s4d_diagonal(N, "real")
    assert np.all(s4d_real < 0)
    assert np.all(np.diff(s4d_real) < 0)  # strictly decreasing
    
    s4d_lin = make_s4d_diagonal(N, "lin")
    assert np.all(np.real(s4d_lin) == -0.5)
    assert len(s4d_lin) == N


def test_legendre_orthogonality():
    """Test that shifted Legendre polynomials are orthonormal over [0, 1]."""
    N = 5
    t = np.linspace(0.0, 1.0, 5000)
    P = legendre_polynomials(t, N)  # (N, len(t))
    dt = t[1] - t[0]
    
    # Gram matrix: G_{nm} = integral_0^1 P_n(t) P_m(t) dt
    gram = (P @ P.T) * dt
    identity = np.eye(N)
    assert np.allclose(gram, identity, atol=1e-2)


def test_signal_reconstruction():
    """Test signal reconstruction from state vector."""
    h = np.array([1.0, 0.5, -0.2])
    tau, rec = reconstruct_signal_from_state(h, num_sample_points=50)
    assert len(tau) == 50
    assert len(rec) == 50
