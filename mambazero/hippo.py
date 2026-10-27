"""
HiPPO (High-Order Polynomial Projection Operators) & Structured State Space Primitives.

Implements continuous-time memory operators projecting online temporal functions 
onto orthogonal polynomial bases (Legendre polynomials):
    d/dt h(t) = A h(t) + B x(t)

Provides:
- HiPPO-LegS (Scaled Legendre Measure)
- S4D-Lin (Diagonalized S4 with complex linear frequencies)
- S4D-Real (Decaying non-oscillatory diagonal modes)
- Function reconstruction from state trajectories
"""

import numpy as np
from typing import Tuple, Literal


def make_hippo_legs(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate HiPPO-LegS continuous state transition matrix A and input matrix B.
    
    Parameters
    ----------
    N : int
        State dimension (order of orthogonal Legendre polynomial basis).
        
    Returns
    -------
    A : np.ndarray, shape (N, N)
        Continuous state transition matrix.
    B : np.ndarray, shape (N, 1)
        Continuous input projection vector.
    """
    A = np.zeros((N, N), dtype=np.float64)
    B = np.zeros((N, 1), dtype=np.float64)
    
    for n in range(N):
        B[n, 0] = np.sqrt(2 * n + 1)
        for k in range(N):
            if n > k:
                A[n, k] = -np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif n == k:
                A[n, k] = -(n + 1)
            else:
                A[n, k] = 0.0
                
    return A, B


def make_s4d_diagonal(
    N: int,
    mode: Literal["lin", "real", "inv"] = "real",
) -> np.ndarray:
    """
    Generate structured diagonal state matrix A according to S4D (Gu et al., 2022).
    
    Parameters
    ----------
    N : int
        State expansion dimension.
    mode : str
        - 'real': Purely real negative decay modes: A_n = -(n + 1)
        - 'lin': Complex modes with linear frequency spacing: -1/2 + i * pi * n
        - 'inv': Complex modes with inverse frequency spacing
        
    Returns
    -------
    np.ndarray, shape (N,)
        Complex or real diagonal eigenvalues.
    """
    if mode == "real":
        # Contractive decay: A_n = -(1 + n)
        return -(np.arange(1, N + 1, dtype=np.float64))
    elif mode == "lin":
        # Complex diagonal: real part -0.5, imaginary part pi * n
        n = np.arange(N, dtype=np.float64)
        return -0.5 + 1j * np.pi * n
    elif mode == "inv":
        n = np.arange(N, dtype=np.float64)
        real = -0.5 * np.ones(N, dtype=np.float64)
        imag = N / np.pi * (N / (2 * n + 1) - 1.0)
        return real + 1j * imag
    else:
        raise ValueError(f"Unknown S4D mode: {mode}")


def legendre_polynomials(t: np.ndarray, N: int) -> np.ndarray:
    """
    Evaluate orthonormal Legendre polynomials P_n(t) for t in [0, 1].
    
    Normalized such that integral_0^1 P_n(t) P_m(t) dt = delta_{nm}.
    """
    # Map t in [0, 1] to x in [-1, 1]: x = 2t - 1
    x = 2.0 * t - 1.0
    L = len(t)
    P = np.zeros((N, L), dtype=np.float64)
    
    if N > 0:
        P[0] = np.ones(L) * np.sqrt(1.0)
    if N > 1:
        P[1] = x * np.sqrt(3.0)
        
    # Bonnet's recurrence relation for standard Legendre polynomials p_n(x):
    # (n + 1) p_{n+1}(x) = (2n + 1) x p_n(x) - n p_{n-1}(x)
    standard_p = np.zeros((N, L), dtype=np.float64)
    if N > 0:
        standard_p[0] = np.ones(L)
    if N > 1:
        standard_p[1] = x
    for n in range(1, N - 1):
        standard_p[n + 1] = ((2 * n + 1) * x * standard_p[n] - n * standard_p[n - 1]) / (n + 1)
        P[n + 1] = standard_p[n + 1] * np.sqrt(2 * (n + 1) + 1)
        
    return P


def reconstruct_signal_from_state(
    h: np.ndarray,
    num_sample_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct temporal function x(tau) from current HiPPO state vector h.
    
    h is the coordinate vector in the orthonormal Legendre basis.
    
    Parameters
    ----------
    h : np.ndarray, shape (N,)
        State vector at current time step.
    num_sample_points : int
        Number of evaluation points in normalized historical window tau in [0, 1].
        
    Returns
    -------
    tau : np.ndarray, shape (num_sample_points,)
    x_rec : np.ndarray, shape (num_sample_points,)
    """
    N = len(h)
    tau = np.linspace(0.0, 1.0, num_sample_points)
    P = legendre_polynomials(tau, N)  # Shape (N, num_sample_points)
    x_rec = h @ P
    return tau, x_rec
