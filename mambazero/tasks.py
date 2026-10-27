"""
Synthetic Frontier Benchmarks: Selective Copying & Associative Recall.

Demonstrates why Selective SSMs (Mamba) overcome fundamental LTI (Linear Time-Invariant)
state space model limitations:
1. Selective Copying: Storing relevant tokens across arbitrary noise distances.
2. In-Context Induction / Associative Recall: Key-value binding and retrieval.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def generate_selective_copying_data(
    batch_size: int = 32,
    seq_len: int = 64,
    num_mem_tokens: int = 4,
    vocab_size: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Selective Copying benchmark dataset.
    
    Format:
    - Vocab 0: Blank / noise token
    - Vocab 1..vocab_size-2: Informational tokens
    - Vocab vocab_size-1: Delimiter / trigger token
    
    A small number of information tokens (num_mem_tokens) appear at random
    positions in the first half of the sequence. The second half contains noise,
    followed by a delimiter, after which the model must output the exact memory tokens.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    BLANK = 0
    DELIM = vocab_size - 1
    INFO_VOCAB_START = 1
    INFO_VOCAB_END = vocab_size - 1
    
    inputs = np.zeros((batch_size, seq_len), dtype=np.int64)
    targets = np.full((batch_size, seq_len), -100, dtype=np.int64)  # -100 = ignore index
    
    # Positions for memory tokens: randomly placed in first half
    first_half_len = seq_len // 2
    
    for b in range(batch_size):
        # Pick distinct positions in first half
        mem_pos = np.sort(rng.choice(first_half_len, size=num_mem_tokens, replace=False))
        # Sample information tokens
        tokens = rng.integers(INFO_VOCAB_START, INFO_VOCAB_END, size=num_mem_tokens)
        inputs[b, mem_pos] = tokens
        
        # Place delimiter at seq_len - num_mem_tokens - 1
        delim_pos = seq_len - num_mem_tokens - 1
        inputs[b, delim_pos] = DELIM
        
        # Expected targets immediately following delimiter
        targets[b, delim_pos + 1 : delim_pos + 1 + num_mem_tokens] = tokens
        
    return inputs, targets


def generate_associative_recall_data(
    batch_size: int = 32,
    num_pairs: int = 6,
    vocab_size: int = 32,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Query Associative Recall (MQAR) benchmark.
    
    Input format: K1 V1 K2 V2 ... Kn Vn ... Query_K -> Target: Query_V
    """
    if rng is None:
        rng = np.random.default_rng()
        
    half_vocab = vocab_size // 2
    
    seq_len = num_pairs * 2 + 1
    inputs = np.zeros((batch_size, seq_len), dtype=np.int64)
    targets = np.full((batch_size, seq_len), -100, dtype=np.int64)
    
    for b in range(batch_size):
        # Keys in range [1, half_vocab - 1]
        keys = rng.choice(np.arange(1, half_vocab), size=num_pairs, replace=False)
        # Values in range [half_vocab, vocab_size - 1]
        vals = rng.integers(half_vocab, vocab_size, size=num_pairs)
        
        # Interleave K, V
        for i in range(num_pairs):
            inputs[b, 2 * i] = keys[i]
            inputs[b, 2 * i + 1] = vals[i]
            
        # Pick one key to query at the end
        query_idx = rng.integers(0, num_pairs)
        inputs[b, -1] = keys[query_idx]
        targets[b, -1] = vals[query_idx]
        
    return inputs, targets


def compute_cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
) -> Tuple[float, float]:
    """
    Compute cross-entropy loss and token accuracy on valid target positions.
    """
    mask = targets != ignore_index
    if not np.any(mask):
        return 0.0, 1.0
        
    # Extract active logits and targets
    active_logits = logits[mask]      # shape (M, V)
    active_targets = targets[mask]    # shape (M,)
    
    # Stable log-softmax
    max_logit = np.max(active_logits, axis=-1, keepdims=True)
    log_sum_exp = max_logit + np.log(np.sum(np.exp(active_logits - max_logit), axis=-1, keepdims=True))
    log_probs = active_logits - log_sum_exp
    
    # Negative log likelihood
    nll = -log_probs[np.arange(len(active_targets)), active_targets]
    loss = float(np.mean(nll))
    
    # Accuracy
    preds = np.argmax(active_logits, axis=-1)
    acc = float(np.mean(preds == active_targets))
    return loss, acc
