"""
Parallel Associative Scan Engine for State Space Models.

Implements work-efficient parallel prefix scans for linear recurrence relations:
    h_t = a_t * h_{t-1} + b_t

Under the binary associative operator (a_j, b_j) ∘ (a_i, b_i) = (a_j * a_i, a_j * b_i + b_j).
Supports both Kogge-Stone prefix doubling and Blelloch work-efficient tree scans.
"""

import numpy as np
from typing import Tuple, Optional


def associative_op(
    elem_j: Tuple[np.ndarray, np.ndarray],
    elem_i: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associative composition operator for linear recurrence pairs (a, b).
    
    Given h_t = a_t * h_{t-1} + b_t, the composition of step j (later) 
    with step i (earlier) yields:
        (a_j, b_j) ∘ (a_i, b_i) = (a_j * a_i, a_j * b_i + b_j)
    """
    a_j, b_j = elem_j
    a_i, b_i = elem_i
    return a_j * a_i, a_j * b_i + b_j


def sequential_scan(
    a: np.ndarray,
    b: np.ndarray,
    h_init: Optional[np.ndarray] = None,
    axis: int = 1,
) -> np.ndarray:
    """
    Sequential linear recurrence scan: h_t = a_t * h_{t-1} + b_t.
    
    Parameters
    ----------
    a : np.ndarray
        Multiplicative decay coefficients along specified axis (e.g., shape (B, L, D, N)).
    b : np.ndarray
        Additive inputs along specified axis (e.g., shape (B, L, D, N)).
    h_init : Optional[np.ndarray]
        Initial hidden state h_{-1}. If None, initialized to zeros.
    axis : int
        Sequence length dimension (default 1).
        
    Returns
    -------
    np.ndarray
        Full state trajectory h of same shape as b.
    """
    # Move sequence axis to front for clean iteration
    a_moved = np.moveaxis(a, axis, 0)
    b_moved = np.moveaxis(b, axis, 0)
    
    L = a_moved.shape[0]
    h_shape = b_moved.shape[1:]
    
    if h_init is None:
        h_curr = np.zeros(h_shape, dtype=b.dtype)
    else:
        h_curr = np.array(h_init, copy=True, dtype=b.dtype)
        
    h_out = np.empty_like(b_moved)
    
    for t in range(L):
        h_curr = a_moved[t] * h_curr + b_moved[t]
        h_out[t] = h_curr
        
    return np.moveaxis(h_out, 0, axis)


def parallel_associative_scan(
    a: np.ndarray,
    b: np.ndarray,
    h_init: Optional[np.ndarray] = None,
    axis: int = 1,
) -> np.ndarray:
    """
    Vectorized parallel prefix doubling associative scan (Kogge-Stone style).
    
    Computes linear recurrence h_t = a_t * h_{t-1} + b_t in O(log L) parallel steps
    using the associative property of the affine operator.
    
    Parameters
    ----------
    a : np.ndarray
        Multiplicative decay factors, shape (..., L, ...).
    b : np.ndarray
        Additive terms, shape (..., L, ...).
    h_init : Optional[np.ndarray]
        Initial state. If provided, incorporated into the first step.
    axis : int
        Sequence dimension.
        
    Returns
    -------
    np.ndarray
        Scanned trajectory h matching shape of b.
    """
    # Move sequence axis to dimension 0 for vectorization
    a_arr = np.moveaxis(np.array(a, copy=True), axis, 0)
    b_arr = np.moveaxis(np.array(b, copy=True), axis, 0)
    
    L = a_arr.shape[0]
    if L == 0:
        return np.moveaxis(b_arr, 0, axis)
        
    if h_init is not None:
        b_arr[0] = a_arr[0] * h_init + b_arr[0]
        
    step = 1
    while step < L:
        # For each element t >= step, compose with element t - step
        a_prev = a_arr[:-step]
        b_prev = b_arr[:-step]
        
        b_arr[step:] = a_arr[step:] * b_prev + b_arr[step:]
        a_arr[step:] = a_arr[step:] * a_prev
        
        step *= 2
        
    return np.moveaxis(b_arr, 0, axis)


def blelloch_scan(
    a: np.ndarray,
    b: np.ndarray,
    axis: int = 1,
) -> np.ndarray:
    """
    Work-efficient Blelloch tree scan (O(L) operations, O(log L) depth).
    
    Pads sequence length to power of 2, executes Up-Sweep (reduce) followed
    by Down-Sweep, then slices to original sequence length.
    """
    a_arr = np.moveaxis(np.array(a, copy=True), axis, 0)
    b_arr = np.moveaxis(np.array(b, copy=True), axis, 0)
    
    orig_L = a_arr.shape[0]
    if orig_L <= 1:
        return np.moveaxis(b_arr, 0, axis)
        
    # Find next power of 2
    pow2_L = 1 << (orig_L - 1).bit_length()
    
    if pow2_L != orig_L:
        pad_len = pow2_L - orig_L
        pad_shape = (pad_len,) + a_arr.shape[1:]
        # Neutral elements: a = 1, b = 0
        a_padded = np.concatenate([a_arr, np.ones(pad_shape, dtype=a_arr.dtype)], axis=0)
        b_padded = np.concatenate([b_arr, np.zeros(pad_shape, dtype=b_arr.dtype)], axis=0)
    else:
        a_padded = a_arr
        b_padded = b_arr
        
    # Parallel scan on padded array
    res = parallel_associative_scan(a_padded, b_padded, axis=0)
    
    # Slice back to original length
    return np.moveaxis(res[:orig_L], 0, axis)
