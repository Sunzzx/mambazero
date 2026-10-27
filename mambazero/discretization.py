"""
Discretization Methods for Continuous-Time State Space Models.

Implements:
- Zero-Order Hold (ZOH) with Taylor-expansion numerical stabilization
- Bilinear (Tustin) transform
- Generalized Euler discretization

Handles scalar, diagonal (S4D/Mamba), and full matrix representations.
"""

import numpy as np
from typing import Tuple, Literal


def expm1_div_z(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable computation of (exp(z) - 1) / z.
    
    Avoids catastrophic cancellation and divide-by-zero when z -> 0 using Taylor series:
        (exp(z) - 1) / z = 1 + z/2 + z^2/6 + z^3/24 + O(z^4)
    """
    small_mask = np.abs(z) < 1e-4
    out = np.empty_like(z, dtype=np.promote_types(z.dtype, np.float64))
    
    # Safe evaluation for larger values
    with np.errstate(divide='ignore', invalid='ignore'):
        out[~small_mask] = np.expm1(z[~small_mask]) / z[~small_mask]
        
    # Taylor series approximation for small values
    z_sm = z[small_mask]
    out[small_mask] = 1.0 + 0.5 * z_sm + (1.0 / 6.0) * (z_sm ** 2) + (1.0 / 24.0) * (z_sm ** 3)
    return out


def discretize_zoh_diagonal(
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero-Order Hold (ZOH) discretization for diagonal state matrices (Mamba / S4D).
    
    Continuous model:
        h'(t) = A h(t) + B x(t)
    
    Discrete ZOH model:
        h_t = A_bar * h_{t-1} + B_bar * x_t
    
    Where:
        A_bar = exp(delta * A)
        B_bar = (exp(delta * A) - I) / A * B = delta * B * [(exp(delta * A) - 1) / (delta * A)]
    
    Parameters
    ----------
    delta : np.ndarray
        Step sizes, shape (..., D) or scalar.
    A : np.ndarray
        Diagonal state matrix elements, shape (D, N) or (N,).
    B : np.ndarray
        Input matrix elements, shape (..., N).
        
    Returns
    -------
    A_bar : np.ndarray
    B_bar : np.ndarray
    """
    # Expand shapes for broadcasting (e.g. delta: (B, L, D, 1), A: (1, 1, D, N))
    dA = delta * A
    A_bar = np.exp(dA)
    
    # B_bar = delta * B * expm1_div_z(dA)
    B_bar = (delta * B) * expm1_div_z(dA)
    return A_bar, B_bar


def discretize_bilinear_diagonal(
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinear (Tustin / trapezoidal) discretization for diagonal state matrices.
    
    A_bar = (1 + delta * A / 2) / (1 - delta * A / 2)
    B_bar = (delta * B) / (1 - delta * A / 2)
    """
    denom = 1.0 - 0.5 * delta * A
    A_bar = (1.0 + 0.5 * delta * A) / denom
    B_bar = (delta * B) / denom
    return A_bar, B_bar


def discretize_euler_diagonal(
    delta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward Euler discretization for diagonal state matrices.
    
    A_bar = 1 + delta * A
    B_bar = delta * B
    """
    A_bar = 1.0 + delta * A
    B_bar = delta * B
    return A_bar, B_bar
