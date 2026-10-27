"""
MambaLM: Full Autoregressive Language Model Architecture.

Implements:
- Token embedding layer
- Stacked Mamba residual blocks with RMSNorm pre-normalization
- Final RMSNorm and LM Head projection
- Parallel sequence forward evaluation
- O(1) memory autoregressive generation with state caching
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .ops import RMSNorm, Linear
from .block import MambaBlock


class MambaLM:
    """
    Mamba Language Model.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size (V).
    d_model : int
        Hidden embedding dimension (D).
    n_layers : int
        Number of stacked Mamba layers.
    d_state : int
        SSM latent state dimension (N, default 16).
    d_conv : int
        Convolution kernel size (default 4).
    expand : int
        Expansion factor for inner dimension (default 2).
    rng : Optional[np.random.Generator]
        Random number generator.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 32,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # Token Embedding: V -> D
        std = np.sqrt(1.0 / d_model)
        self.embedding = rng.normal(0.0, std, (vocab_size, d_model)).astype(np.float64)
        
        # Layers: list of (RMSNorm, MambaBlock)
        self.layers: List[Tuple[RMSNorm, MambaBlock]] = []
        for _ in range(n_layers):
            norm = RMSNorm(d_model)
            block = MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                rng=rng,
            )
            self.layers.append((norm, block))
            
        # Final Norm & LM Head: D -> V
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False, rng=rng)

    def forward_sequence(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Full parallel forward pass.
        
        Parameters
        ----------
        input_ids : np.ndarray, shape (B, L) of integer token IDs
        
        Returns
        -------
        logits : np.ndarray, shape (B, L, V)
        """
        # 1. Embedding lookup: (B, L, D)
        x = self.embedding[input_ids]
        
        # 2. Sequential pass through Mamba layers
        for norm, block in self.layers:
            # Pre-norm followed by residual block (block internally adds residual)
            x_norm = norm(x)
            x, _ = block.forward_sequence(x_norm)
            
        # 3. Final norm & LM Head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, L, V)
        return logits

    def step(
        self,
        input_id_t: np.ndarray,
        layer_states: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Single step autoregressive token pass with state caching.
        
        Parameters
        ----------
        input_id_t : np.ndarray, shape (B,) of token IDs
        layer_states : Optional[List[Tuple[conv_buffer, ssm_state]]]
            One cache per layer.
            
        Returns
        -------
        logits_t : np.ndarray, shape (B, V)
        new_states : List[Tuple[conv_buffer, ssm_state]]
        """
        B = input_id_t.shape[0]
        x_t = self.embedding[input_id_t]  # (B, D)
        
        if layer_states is None:
            layer_states = [None] * self.n_layers
            
        new_states = []
        for i, (norm, block) in enumerate(self.layers):
            x_norm = norm(x_t)
            x_t, new_st = block.step(x_norm, layer_states[i])
            new_states.append(new_st)
            
        x_t = self.final_norm(x_t)
        logits_t = self.lm_head(x_t)  # (B, V)
        return logits_t, new_states

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        """
        Autoregressive generation with O(1) state caching per step.
        
        Parameters
        ----------
        prompt_ids : np.ndarray, shape (B, L_prompt)
        max_new_tokens : int
        temperature : float
        top_k : Optional[int]
        
        Returns
        -------
        generated_ids : np.ndarray, shape (B, L_prompt + max_new_tokens)
        """
        B, L_prompt = prompt_ids.shape
        
        # Warmup cache by processing prompt sequentially
        states = None
        for t in range(L_prompt):
            logits_t, states = self.step(prompt_ids[:, t], states)
            
        # Generated sequence tokens
        generated = [prompt_ids[:, t] for t in range(L_prompt)]
        curr_token = prompt_ids[:, -1]
        
        for _ in range(max_new_tokens):
            # Compute next token probabilities from current logits
            logits = logits_t / max(temperature, 1e-6)
            
            if top_k is not None and top_k > 0:
                # Top-k filtering
                top_k = min(top_k, self.vocab_size)
                for b in range(B):
                    indices_to_remove = logits[b] < np.partition(logits[b], -top_k)[-top_k]
                    logits[b, indices_to_remove] = -np.inf
                    
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Sample token for each sequence in batch
            next_tokens = np.zeros(B, dtype=np.int64)
            for b in range(B):
                next_tokens[b] = self.rng.choice(self.vocab_size, p=probs[b])
                
            generated.append(next_tokens)
            
            # Step forward with sampled token
            logits_t, states = self.step(next_tokens, states)
            
        return np.stack(generated, axis=1)
