"""
Tests for synthetic frontier tasks.
"""

import pytest
import numpy as np
from mambazero.tasks import generate_selective_copying_data, generate_associative_recall_data, compute_cross_entropy_loss


def test_selective_copying_format():
    """Verify selective copying dataset structure and ground truth alignment."""
    B, L, num_mem, V = 4, 32, 4, 16
    inp, tgt = generate_selective_copying_data(B, L, num_mem, V)
    
    assert inp.shape == (B, L)
    assert tgt.shape == (B, L)
    # Active target count per sequence must equal num_mem
    for b in range(B):
        active = tgt[b, tgt[b] != -100]
        assert len(active) == num_mem
        assert np.all((active >= 1) & (active < V - 1))


def test_associative_recall_format():
    """Verify MQAR dataset structure."""
    B, num_pairs, V = 4, 5, 32
    inp, tgt = generate_associative_recall_data(B, num_pairs, V)
    
    assert inp.shape == (B, num_pairs * 2 + 1)
    assert tgt.shape == (B, num_pairs * 2 + 1)
    for b in range(B):
        # Query key is the last token
        query_k = inp[b, -1]
        target_v = tgt[b, -1]
        assert target_v != -100
        
        # Verify that query_k and target_v match one of the pairs
        matched = False
        for i in range(num_pairs):
            if inp[b, 2 * i] == query_k and inp[b, 2 * i + 1] == target_v:
                matched = True
                break
        assert matched, "Target value must match key-value pair in sequence!"


def test_cross_entropy_loss_perfect_accuracy():
    """Verify that ideal logits produce minimal loss and 1.0 accuracy."""
    targets = np.array([[1, 2, -100]])
    logits = np.zeros((1, 3, 5))
    logits[0, 0, 1] = 50.0  # high confidence on 1
    logits[0, 1, 2] = 50.0  # high confidence on 2
    
    loss, acc = compute_cross_entropy_loss(logits, targets)
    assert acc == 1.0
    assert loss < 1e-4
