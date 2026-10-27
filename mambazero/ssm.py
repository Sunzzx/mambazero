"""
Selective State Space Model (Mamba Core Mechanism).

Implements the data-dependent selection mechanism where transition matrices (B, C, delta)
are computed dynamically from inputs at each time-step, breaking LTI (linear time-invariance)
while preserving O(L) training parallelism via parallel associative scan and O(1) inference.
"""

import numpy as np
from typing import Optional, Tuple
from .scan import parallel_associative_scan, sequential_scan
from .discretization import expm1_div_z
from .ops import softplus


class SelectiveSSM:
    """
    Selective State Space Model layer.
    
    Parameters
    ----------
    d_model : int
        Hidden / channel dimension (D).
    d_state : int
        SSM latent state expansion factor (N, typically 16).
    dt_rank : Optional[int]
        Rank of Delta projection. Defaults to ceil(d_model / 16).
    dt_min : float
        Minimum step size (default 0.001).
    dt_max : float
        Maximum step size (default 0.1).
    rng : Optional[np.random.Generator]
        Random number generator for reproducible initialization.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
            
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = int(np.ceil(d_model / 16.0)) if dt_rank is None else dt_rank
        
        # S4D-real initialization for A: A_{d, n} = -(n + 1)
        # We store A_log to ensure contractive stability: A = -exp(A_log)
        # Shape (d_model, d_state)
        n_vals = np.repeat(np.arange(1, d_state + 1, dtype=np.float64)[None, :], d_model, axis=0)
        self.A_log = np.log(n_vals)
        
        # Residual skip connection parameter D_res, shape (d_model,)
        self.D_res = np.ones(d_model, dtype=np.float64)
        
        # Input projection into [delta_proj, B_proj, C_proj]
        # Projects d_model -> (dt_rank + 2 * d_state)
        proj_dim = self.dt_rank + 2 * self.d_state
        std_in = np.sqrt(2.0 / (d_model + proj_dim))
        self.x_proj_weight = rng.normal(0.0, std_in, (d_model, proj_dim)).astype(np.float64)
        
        # Delta projection: dt_rank -> d_model
        std_dt = np.sqrt(2.0 / (self.dt_rank + d_model))
        self.dt_proj_weight = rng.normal(0.0, std_dt, (self.dt_rank, d_model)).astype(np.float64)
        
        # Delta bias initialization: uniform in [log(dt_min), log(dt_max)]
        dt_init = np.exp(rng.uniform(np.log(dt_min), np.log(dt_max), size=(d_model,)))
        # Inverse softplus: log(exp(y) - 1)
        self.dt_proj_bias = np.log(np.expm1(dt_init)).astype(np.float64)

    @property
    def A(self) -> np.ndarray:
        """Get structured transition matrix A = -exp(A_log) of shape (d_model, d_state)."""
        return -np.exp(self.A_log)

    def forward_parallel(
        self,
        u: np.ndarray,
        h_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parallel forward pass across sequence using associative scan.
        
        Parameters
        ----------
        u : np.ndarray, shape (B, L, D)
            Input sequence tensor.
        h_init : Optional[np.ndarray], shape (B, D, N)
            Initial latent state.
            
        Returns
        -------
        y : np.ndarray, shape (B, L, D)
            Output sequence.
        h_final : np.ndarray, shape (B, D, N)
            Final latent hidden state.
        """
        B, L, D = u.shape
        N = self.d_state
        
        # 1. Project input: u @ x_proj -> [delta_low, B, C]
        # x_proj shape: (B, L, dt_rank + 2*N)
        x_proj = u @ self.x_proj_weight
        
        delta_low = x_proj[..., :self.dt_rank]              # (B, L, dt_rank)
        B_proj = x_proj[..., self.dt_rank:self.dt_rank + N] # (B, L, N)
        C_proj = x_proj[..., self.dt_rank + N:]             # (B, L, N)
        
        # 2. Compute delta = softplus(delta_low @ dt_proj_weight + dt_proj_bias)
        delta = softplus(delta_low @ self.dt_proj_weight + self.dt_proj_bias)  # (B, L, D)
        
        # 3. Discretize continuous (A, B) using ZOH
        # A shape: (D, N)
        # delta shape: (B, L, D, 1)
        # dA shape: (B, L, D, N)
        delta_exp = np.expand_dims(delta, axis=-1)  # (B, L, D, 1)
        dA = delta_exp * self.A                     # (B, L, D, N)
        A_bar = np.exp(dA)                          # (B, L, D, N)
        
        # B_proj shape: (B, L, 1, N)
        B_exp = np.expand_dims(B_proj, axis=2)      # (B, L, 1, N)
        # ZOH formula: B_bar = delta * B * expm1_div_z(dA)
        B_bar = (delta_exp * B_exp) * expm1_div_z(dA)  # (B, L, D, N)
        
        # Multiply by input u: u shape (B, L, D, 1)
        u_exp = np.expand_dims(u, axis=-1)          # (B, L, D, 1)
        B_u = B_bar * u_exp                         # (B, L, D, N)
        
        # 4. Parallel associative scan along sequence dimension (axis=1)
        # Recurrence: h_t = A_bar_t * h_{t-1} + B_u_t
        h = parallel_associative_scan(A_bar, B_u, h_init=h_init, axis=1)  # (B, L, D, N)
        
        # 5. Output projection: y_t = sum_n (h_t * C_t) + D_res * u_t
        # C_proj shape: (B, L, 1, N)
        C_exp = np.expand_dims(C_proj, axis=2)
        # Contracting over N
        y_ssm = np.sum(h * C_exp, axis=-1)          # (B, L, D)
        
        # Add skip connection
        y = y_ssm + u * self.D_res
        
        h_final = h[:, -1, :, :] if L > 0 else np.zeros((B, D, N), dtype=u.dtype)
        return y, h_final

    def step(
        self,
        u_t: np.ndarray,
        h_prev: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step recurrent inference: O(1) time and memory per token.
        
        Parameters
        ----------
        u_t : np.ndarray, shape (B, D)
            Input at current time step.
        h_prev : Optional[np.ndarray], shape (B, D, N)
            Hidden state from previous time step.
            
        Returns
        -------
        y_t : np.ndarray, shape (B, D)
            Output at current step.
        h_curr : np.ndarray, shape (B, D, N)
            Updated hidden state.
        """
        B, D = u_t.shape
        N = self.d_state
        if h_prev is None:
            h_prev = np.zeros((B, D, N), dtype=u_t.dtype)
            
        # 1. Project input
        x_proj = u_t @ self.x_proj_weight
        delta_low = x_proj[:, :self.dt_rank]
        B_proj = x_proj[:, self.dt_rank:self.dt_rank + N]
        C_proj = x_proj[:, self.dt_rank + N:]
        
        # 2. Compute delta
        delta = softplus(delta_low @ self.dt_proj_weight + self.dt_proj_bias)  # (B, D)
        
        # 3. Discretize
        delta_exp = np.expand_dims(delta, axis=-1)  # (B, D, 1)
        dA = delta_exp * self.A                     # (B, D, N)
        A_bar = np.exp(dA)                          # (B, D, N)
        
        B_exp = np.expand_dims(B_proj, axis=1)      # (B, 1, N)
        B_bar = (delta_exp * B_exp) * expm1_div_z(dA)  # (B, D, N)
        
        u_exp = np.expand_dims(u_t, axis=-1)        # (B, D, 1)
        B_u = B_bar * u_exp                         # (B, D, N)
        
        # 4. State update: h_t = A_bar * h_{t-1} + B_u
        h_curr = A_bar * h_prev + B_u               # (B, D, N)
        
        # 5. Output: sum_n (h_curr * C_exp) + D_res * u_t
        C_exp = np.expand_dims(C_proj, axis=1)      # (B, 1, N)
        y_ssm = np.sum(h_curr * C_exp, axis=-1)     # (B, D)
        y_t = y_ssm + u_t * self.D_res
        
        return y_t, h_curr
