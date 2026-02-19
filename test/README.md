# Test Suite — QCD Vortex Entanglement

Comprehensive test suite for the QCD Vortex Entanglement project.

## Overview

| Metric | Value |
|---|---|
| Total tests | ~95 across 4 modules |
| Coverage target | > 90% |
| Framework | pytest 7.0+ |
| Execution time | ~5–10 min (full suite) |

---

## Directory Structure

```
test/
├── __init__.py                  # Package initialization (v1.1.0)
├── conftest.py                  # Shared fixtures and markers
├── README.md                    # This file
├── validation_tools.py          # Shared injection helpers (TMST + Qiskit)
├── test_vortex_mps.py           # MPS simulation tests (43 tests)
├── test_entanglement.py         # Entanglement measure tests (38 tests)
├── test_tmst_injection.py       # TMST injection tests — Theorem 4.3.1 (5 tests)
└── test_injection_qiskit.py     # Qiskit 2-qubit injection tests (4 tests)
```

---

## Quick Start

```bash
pip install -e ".[dev]"
pytest test/ -v
pip install -r requirements-quantum.txt
pytest test/ -v
pytest test/test_tmst_injection.py test/test_injection_qiskit.py -v
```

---

## Test Modules

### `test_vortex_mps.py` — 43 tests

| Group | Tests |
|---|---|
| MPS Initialization | 5 |
| Collective Squeezing | 4 |
| TMST State Properties | 6 |
| Time Evolution | 3 |
| MVC Threshold Detection | 3 |
| Exceptional Points | 5 |
| Entanglement Measures | 4 |
| Export / Import HDF5 | 3 |
| Performance & Edge Cases | 10 |

### `test_entanglement.py` — 38 tests

| Group | Tests |
|---|---|
| Log-Negativity | 6 |
| Concurrence | 5 |
| Entanglement Witnesses | 4 |
| Bell Inequalities (CHSH, Mermin) | 5 |
| Correlation Functions | 4 |
| Separability Criteria (PPT, CCNR) | 4 |
| Gaussian State Entanglement | 4 |

### `test_tmst_injection.py` — 5 tests *(NEW)*

- **`test_tmst_separable_below_threshold`** (×7 temps) — E_N ≈ 0 for r < r_c(T)
- **`test_tmst_entangled_above_threshold`** (×7 temps) — E_N > 0 for r > r_c(T)
- **`test_log_negativity_monotone_with_squeezing`** (×3 temps) — fixes failing monotonicity test
- **`test_nu_minus_below_half_iff_entangled`** (×4 temps) — Simon criterion, ν₋(r_c)=½ exactly
- **`test_zero_temperature_limit`** — r_c → 0, E_N > 0 for any r > 0
- **`test_critical_squeezing_formula`** (×10 temps) — Theorem 4.3.1 self-consistency

### `test_injection_qiskit.py` — 4 tests *(NEW)*

Auto-skipped if `qiskit` / `qiskit-aer` not installed.

- **`test_bell_state_log_negativity`** — |Φ+⟩ must yield E_N ≈ 1.0
- **`test_product_noise_log_negativity`** (×5 seeds) — RX noise must yield E_N ≈ 0
- **`test_bell_fidelity_is_one`** — fidelity with ideal |Φ+⟩ = 1.0
- **`test_injection_signal_clearly_above_noise`** — E_N(signal) − E_N(noise) > 0.8
- **`test_full_injection_run_passes`** — `run_injection_test()` returns `passed=True`

---

## Running Tests

```bash
# Core tests (no Qiskit)
pytest test/ -v
pytest test/ --ignore=test/test_injection_qiskit.py -v
pytest test/ --ignore=test/test_injection_qiskit.py --cov=src --cov-report=html --cov-report=term
pytest test/ -n auto
pytest test/ -x
pytest test/ -m "not slow"

# Quantum tests
pip install -r requirements-quantum.txt
pytest test/test_injection_qiskit.py -v
pytest test/ -m quantum -v

# Useful flags
pytest test/ -v -s
pytest test/ --lf
pytest test/ -vv --tb=long
pytest test/ --timeout=60
pytest test/ --durations=10
```

---

## Markers

| Marker | Description |
|---|---|
| `slow` | Tests > 1 second |
| `integration` | Integration tests |
| `gpu` | Requires GPU/CUDA |
| `quantum` | Requires qiskit + qiskit-aer *(NEW)* |

```bash
pytest test/ -m "not quantum"
pytest test/ -m "not slow"
```

---

## Fixtures (`conftest.py`)

| Fixture | Scope | Description |
|---|---|---|
| `reset_random_seed` | function (autouse) | `np.random.seed(42)` before each test |
| `suppress_warnings` | function (autouse) | Filters `DeprecationWarning` |
| `tolerance` | function | Returns `1e-10` |
| `temp_directory` | function | `tmp_path` directory |
| `qiskit_available` | session *(NEW)* | `True`/`False` for qiskit-aer |

```python
def test_something(qiskit_available):
    if not qiskit_available:
        pytest.skip("qiskit not installed")
```

---

## CI Jobs (`.github/workflows/tests.yml`)

**Job 1 — Core** (Ubuntu + macOS + Windows × Python 3.10, 3.11):

```bash
pytest test/ --ignore=test/test_injection_qiskit.py --cov=src --cov-report=xml
```

**Job 2 — Quantum** (Ubuntu × Python 3.11, `continue-on-error: true`):

```bash
pip install -r requirements-quantum.txt
pytest test/test_injection_qiskit.py -v
```

The quantum job **never blocks a merge**.

---

## Coverage Targets

| Module | Target |
|---|---|
| `src/seemps_vortex/` | > 90% |
| `src/belle2_analysis/` | > 85% |
| `src/ibm_validation/` | > 80% |

```bash
pytest test/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: src` | `pip install -e .` |
| `ModuleNotFoundError: seemps2` | `pip install seemps2` |
| Qiskit tests skipped | `pip install -r requirements-quantum.txt` |
| Tests hang | `pytest test/ --timeout=60` |
| `test.validation_tools` not found | Confirm folder is `test/` not `tests/` |

---

## Resources

- [pytest docs](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Qiskit docs](https://docs.quantum.ibm.com/)
- [SeeMPS](https://github.com/juanjosegarciaripoll/seemps2)

---

**Last Updated**: February 2026 | **Version**: 1.1.0
