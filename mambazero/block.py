"""
Mamba Residual Block Architecture (Gu & Dao, 2023).

Integrates:
- Input projection with expansion factor (expand = 2)
- Causal 1D depthwise convolution (kernel_size = 4)
- SiLU / Swish non-linear activations
- Selective State Space Model core with parallel associative scan
- Multiplicative gating branch
- Output projection and residual skip connection
- Exact state caching for step-by-step autoregressive generation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from .ops import Linear, CausalConv1d, silu
from .ssm import SelectiveSSM


class MambaBlock:
    """
    Full Mamba block with dual-branch gated architecture.
    
    Parameters
    ----------
    d_model : int
        Model dimension (D).
    d_state : int
        SSM latent state expansion (N, default 16).
    d_conv : int
        Local causal convolution kernel size (default 4).
    expand : int
        Inner dimension expansion factor (default 2 -> d_inner = expand * d_model).
    dt_rank : Optional[int]
        Rank of Delta projection.
    rng : Optional[np.random.Generator]
        Random number generator.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
            
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank
        
        # 1. Dual input projection: projects d_model -> 2 * d_inner
        # Branch 1: x_ssm (to be convolved and passed to SSM)
        # Branch 2: z (multiplicative gate branch)
        self.in_proj = Linear(d_model, 2 * self.d_inner, bias=False, rng=rng)
        
        # 2. Causal 1D Depthwise Convolution on x_ssm
        self.conv1d = CausalConv1d(channels=self.d_inner, kernel_size=d_conv, bias=True, rng=rng)
        
        # 3. Selective State Space Model on convolved x_ssm
        self.ssm = SelectiveSSM(
            d_model=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
            rng=rng,
        )
        
        # 4. Output projection: projects d_inner -> d_model
        self.out_proj = Linear(self.d_inner, d_model, bias=False, rng=rng)

    def forward_sequence(
        self,
        x: np.ndarray,
        h_init: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Parallel sequence forward pass.
        
        Parameters
        ----------
        x : np.ndarray, shape (B, L, D)
        h_init : Optional[Tuple[conv_buffer, ssm_state]]
            Initial cache states.
            
        Returns
        -------
        out : np.ndarray, shape (B, L, D)
        final_state : Tuple[conv_buffer, ssm_state]
        """
        B, L, D = x.shape
        
        # 1. Dual projection
        # proj shape: (B, L, 2 * d_inner)
        xz = self.in_proj(x)
        x_ssm = xz[..., :self.d_inner]
        z = xz[..., self.d_inner:]
        
        # 2. Causal convolution + SiLU activation
        x_conv = self.conv1d.forward_sequence(x_ssm)  # (B, L, d_inner)
        x_act = silu(x_conv)
        
        # Initial SSM state if provided
        ssm_init = h_init[1] if h_init is not None else None
        
        # 3. Selective SSM forward pass (parallel associative scan)
        y_ssm, ssm_state_final = self.ssm.forward_parallel(x_act, h_init=ssm_init)  # (B, L, d_inner)
        
        # 4. Multiplicative gating: y_ssm * silu(z)
        y_gated = y_ssm * silu(z)
        
        # 5. Output projection + residual connection
        out = self.out_proj(y_gated) + x
        
        # Compute final conv buffer: last (d_conv - 1) tokens of x_ssm
        K = self.d_conv
        if L >= K - 1:
            conv_buffer_final = x_ssm[:, -(K - 1):, :]
        else:
            pad = np.zeros((B, (K - 1) - L, self.d_inner), dtype=x.dtype)
            conv_buffer_final = np.concatenate([pad, x_ssm], axis=1)
            
        return out, (conv_buffer_final, ssm_state_final)

    def step(
        self,
        x_t: np.ndarray,
        state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Step-by-step autoregressive inference in O(1) time and memory.
        
        Parameters
        ----------
        x_t : np.ndarray, shape (B, D)
        state : Optional[Tuple[conv_buffer, ssm_state]]
            conv_buffer shape (B, d_conv - 1, d_inner)
            ssm_state shape (B, d_inner, d_state)
            
        Returns
        -------
        out_t : np.ndarray, shape (B, D)
        new_state : Tuple[conv_buffer, ssm_state]
        """
        B, D = x_t.shape
        conv_buf = state[0] if state is not None else None
        ssm_st = state[1] if state is not None else None
        
        # 1. Dual projection
        xz_t = self.in_proj(x_t)  # (B, 2 * d_inner)
        x_ssm_t = xz_t[:, :self.d_inner]
        z_t = xz_t[:, self.d_inner:]
        
        # 2. Causal convolution step
        x_conv_t, new_conv_buf = self.conv1d.step(x_ssm_t, conv_buf)
        x_act_t = silu(x_conv_t)
        
        # 3. SSM step
        y_ssm_t, new_ssm_st = self.ssm.step(x_act_t, ssm_st)
        
        # 4. Gating
        y_gated_t = y_ssm_t * silu(z_t)
        
        # 5. Output projection + residual connection
        out_t = self.out_proj(y_gated_t) + x_t
        
        return out_t, (new_conv_buf, new_ssm_st)
