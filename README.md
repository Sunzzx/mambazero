# MambaZero: Frontier Selective State Space Models & Parallel Scan Engine

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-31%2F31%20passing-success.svg)](./tests)

**MambaZero** is an elite, mathematically rigorous, zero-dependency implementation of Selective State Space Models (Mamba / S4) and parallel associative scan engines built purely on mathematical primitives in NumPy.

---

## Architectural Pillars

### 1. HiPPO (High-Order Polynomial Projection Operators)
Implements continuous-time memory operators projecting online temporal functions onto orthogonal polynomial bases (Legendre polynomials):
$$\frac{d}{dt} h(t) = A h(t) + B x(t)$$
- **Hurwitz Stability Guarantee**: All state transition matrix $A$ eigenvalues possess strictly negative real parts (verified $\max \text{Re}(\lambda) < 0$).
- **S4D Diagonalization**: Efficient diagonal Structured State Space representations supporting continuous-time memory compression.

### 2. Parallel Associative Scan (O(log L) Span)
Breaks linear recurrence sequential bottlenecks by formulating linear scans as associative reductions over pairs $(a_t, b_t)$:
$$(a_j, b_j) \circ (a_i, b_i) = (a_j \cdot a_i, \quad a_j \cdot b_i + b_j)$$
- **Kogge-Stone Doubling Tree**: Implements $O(\log L)$ parallel prefix scan.
- **Blelloch Work-Efficient Scan**: Optimal work complexity parallel reduction.
- **Exact Equivalence**: Numerically verified absolute difference $< 10^{-14}$ against sequential recurrence.

### 3. Data-Dependent Selection Mechanism (Mamba Core)
Replaces Linear Time-Invariance (LTI) systems with input-dependent projection matrices ($\Delta_t, B_t, C_t$), enabling content-based reasoning while retaining parallel training and $O(1)$ autoregressive inference.

### 4. Numerical Stabilization & ZOH Discretization
- **Taylor Stabilized $\text{expm1}(z)/z$**: Prevents catastrophic cancellation when step size $\Delta \to 0$.
- **Zero-Order Hold (ZOH)**: Exact continuous-to-discrete conversion for diagonal state matrices.

---

## Installation & Verification

```bash
git clone https://github.com/Sunzzx/mambazero.git
cd mambazero
pip install -e .[dev]

# Run comprehensive test suite (31 tests)
pytest -v

# Run verification and benchmark suites
python3 -m mambazero.cli verify
python3 -m mambazero.cli benchmark
python3 -m mambazero.cli demo
```

---

## Project Structure

```
mambazero/
├── mambazero/
│   ├── __init__.py          # Public API exports
│   ├── hippo.py             # HiPPO LegS & S4D matrices & reconstruction
│   ├── discretization.py    # ZOH, Bilinear, Euler, stable expm1_div_z
│   ├── ops.py               # RMSNorm, Linear, CausalConv1d, SiLU, Softplus
│   ├── scan.py              # Parallel associative scan (Kogge-Stone & Blelloch)
│   ├── ssm.py               # Selective SSM (input-dependent B, C, Delta)
│   ├── block.py             # Mamba Residual Block with SSM & Conv1d
│   ├── model.py             # Full Mamba Language Model (LM) & O(1) state cache
│   ├── tasks.py             # Selective copying & associative recall benchmarks
│   └── cli.py               # CLI interface, benchmarks, and terminal visualizer
├── tests/                   # 31 comprehensive pytest unit tests
├── pyproject.toml           # Project metadata & build configuration
└── README.md                # Mathematical documentation
```
