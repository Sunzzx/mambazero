"""
Neural Operations & Primitive Layers for State Space Models.

Implements:
- Linear layer with parameter tracking
- Root Mean Square Normalization (RMSNorm)
- Sigmoid Linear Unit (SiLU / Swish)
- Depthwise Causal 1D Convolution with cached inference buffer
"""

import numpy as np
from typing import Optional, Tuple


def silu(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid Linear Unit (SiLU / Swish): x * sigmoid(x).
    """
    # Stable sigmoid
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return x * sig


def softplus(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softplus: log(1 + exp(x)).
    """
    return np.where(x > 20.0, x, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0))


class RMSNorm:
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize across last dimension.
        """
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        normed = x * (1.0 / np.sqrt(variance + self.eps))
        return normed * self.weight


class Linear:
    """
    Linear projection layer: y = x @ W + b.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier normal initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = rng.normal(0.0, std, (in_features, out_features)).astype(np.float64)
        if bias:
            self.bias = np.zeros(out_features, dtype=np.float64)
        else:
            self.bias = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class CausalConv1d:
    """
    Depthwise 1D Causal Convolution.
    
    Maintains causality by left-padding with kernel_size - 1 elements in sequence mode,
    and supporting O(1) state caching for step-by-step autoregressive generation.
    """
    def __init__(self, channels: int, kernel_size: int = 4, bias: bool = True, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()
        self.channels = channels
        self.kernel_size = kernel_size
        # Depthwise kernel: shape (channels, kernel_size)
        std = np.sqrt(1.0 / (channels * kernel_size))
        self.weight = rng.normal(0.0, std, (channels, kernel_size)).astype(np.float64)
        if bias:
            self.bias = np.zeros(channels, dtype=np.float64)
        else:
            self.bias = None

    def forward_sequence(self, x: np.ndarray) -> np.ndarray:
        """
        Full sequence causal convolution.
        
        Parameters
        ----------
        x : np.ndarray, shape (B, L, D) where D == channels
        
        Returns
        -------
        np.ndarray, shape (B, L, D)
        """
        B, L, D = x.shape
        assert D == self.channels, f"Expected channels={self.channels}, got {D}"
        K = self.kernel_size
        
        # Left-pad with K - 1 zeros along time axis
        pad = np.zeros((B, K - 1, D), dtype=x.dtype)
        x_padded = np.concatenate([pad, x], axis=1)  # shape (B, L + K - 1, D)
        
        # Compute depthwise 1D convolution
        # Convolution sum over k in 0..K-1: weight[:, k] * x_padded[:, t + k, :]
        out = np.zeros((B, L, D), dtype=x.dtype)
        for k in range(K):
            # kernel weights are ordered such that k=K-1 corresponds to current token
            # weight: shape (D, K)
            out += x_padded[:, k:k + L, :] * self.weight[:, k]
            
        if self.bias is not None:
            out += self.bias
        return out

    def step(self, x_t: np.ndarray, buffer: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step causal convolution with cached state buffer.
        
        Parameters
        ----------
        x_t : np.ndarray, shape (B, D)
            Input at current time step.
        buffer : Optional[np.ndarray], shape (B, K - 1, D)
            Past token buffer. If None, initialized to zeros.
            
        Returns
        -------
        out_t : np.ndarray, shape (B, D)
        new_buffer : np.ndarray, shape (B, K - 1, D)
        """
        B, D = x_t.shape
        K = self.kernel_size
        
        if buffer is None:
            buffer = np.zeros((B, K - 1, D), dtype=x_t.dtype)
            
        # Concatenate buffer and current token
        # window: shape (B, K, D)
        x_exp = np.expand_dims(x_t, axis=1)
        window = np.concatenate([buffer, x_exp], axis=1)
        
        # Apply weights
        # window is [w_{-(K-1)}, ..., w_0]
        out_t = np.zeros((B, D), dtype=x_t.dtype)
        for k in range(K):
            out_t += window[:, k, :] * self.weight[:, k]
            
        if self.bias is not None:
            out_t += self.bias
            
        # Update buffer by shifting out oldest and sliding in current
        new_buffer = window[:, 1:, :]
        return out_t, new_buffer
