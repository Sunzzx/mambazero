"""
MambaZero: High-Performance Selective State Space Model from Mathematical Primitives.
"""

from .scan import (
    sequential_scan,
    parallel_associative_scan,
    blelloch_scan,
    associative_op,
)
from .hippo import (
    make_hippo_legs,
    make_s4d_diagonal,
    legendre_polynomials,
    reconstruct_signal_from_state,
)
from .discretization import (
    discretize_zoh_diagonal,
    discretize_bilinear_diagonal,
    discretize_euler_diagonal,
    expm1_div_z,
)
from .ops import (
    RMSNorm,
    Linear,
    CausalConv1d,
    silu,
    softplus,
)
from .ssm import SelectiveSSM
from .block import MambaBlock
from .model import MambaLM
from .tasks import (
    generate_selective_copying_data,
    generate_associative_recall_data,
    compute_cross_entropy_loss,
)

__version__ = "0.1.0"
__all__ = [
    "sequential_scan",
    "parallel_associative_scan",
    "blelloch_scan",
    "associative_op",
    "make_hippo_legs",
    "make_s4d_diagonal",
    "legendre_polynomials",
    "reconstruct_signal_from_state",
    "discretize_zoh_diagonal",
    "discretize_bilinear_diagonal",
    "discretize_euler_diagonal",
    "expm1_div_z",
    "RMSNorm",
    "Linear",
    "CausalConv1d",
    "silu",
    "softplus",
    "SelectiveSSM",
    "MambaBlock",
    "MambaLM",
    "generate_selective_copying_data",
    "generate_associative_recall_data",
    "compute_cross_entropy_loss",
]
