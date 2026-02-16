# Test Suite Documentation

Comprehensive test suite for the QCD Vortex Entanglement project.

## Overview

- **Total Tests**: ~81 tests across 2 modules
- **Coverage Target**: > 90%
- **Framework**: pytest 7.0+
- **Execution Time**: ~5-10 minutes (full suite)

## Test Structure

```
tests/
├── __init__.py              # Package initialization
├── conftest.py              # Shared fixtures and configuration
├── test_vortex_mps.py       # MPS simulation tests (43 tests)
└── test_entanglement.py     # Entanglement measure tests (38 tests)
```

## Test Coverage

### test_vortex_mps.py

Tests for QCD vortex MPS simulations:

- **MPS Initialization** (5 tests)
  - System creation
  - Product state initialization
  - Random state initialization
  - GHZ state preparation
  - Parameter validation

- **Collective Squeezing** (4 tests)
  - Squeezing operator creation
  - Parameter range validation
  - Collective mode preparation
  - Entanglement enhancement

- **TMST State Properties** (6 tests)
  - State creation
  - Thermal occupation calculation
  - Entanglement threshold (Theorem 4.3.1)
  - Critical squeezing
  - Log-negativity computation
  - Covariance matrix

- **Time Evolution** (3 tests)
  - Short-time Lindblad evolution
  - Decoherence effects
  - Long-time dynamics

- **MVC Threshold Detection** (3 tests)
  - Detector initialization
  - Critical density calculation
  - Confinement phase detection
  - Hadronization multiplicity

- **Exceptional Points** (5 tests)
  - EP analyzer creation
  - Effective Hamiltonian
  - EP detection
  - Petermann factor
  - PT-symmetry phases

- **Entanglement Measures** (4 tests)
  - Entropy of product states
  - Entropy of GHZ states
  - Entanglement spectrum
  - Mutual information

- **Export/Import** (3 tests)
  - HDF5 export
  - HDF5 import
  - Metadata preservation

- **Performance & Edge Cases** (10 tests)
  - Zero squeezing
  - High/low temperature limits
  - Different system sizes
  - Boundary conditions

### test_entanglement.py

Tests for entanglement quantification:

- **Log-Negativity** (6 tests)
  - Product state (zero negativity)
  - EPR state (positive negativity)
  - Thermal states
  - Bounds verification
  - Monotonicity with squeezing
  - Density matrix computation

- **Concurrence** (5 tests)
  - Bell state (maximal)
  - Product state (zero)
  - Werner states
  - Bounds verification
  - Gaussian state approximation

- **Entanglement Witnesses** (4 tests)
  - Bell witness
  - Witness on separable states
  - PPT criterion
  - CCNR criterion

- **Bell Inequalities** (5 tests)
  - CHSH classical bound
  - CHSH quantum violation
  - Computation from correlations
  - Clauser-Horne inequality
  - Mermin inequality (3 qubits)

- **Correlation Functions** (4 tests)
  - Spin correlations (product states)
  - Spin correlations (Bell states)
  - Two-point functions
  - Connected correlations

- **Separability Criteria** (4 tests)
  - PPT on separable states
  - PPT on entangled states
  - Reduction criterion
  - Cross-norm criterion

- **Gaussian State Entanglement** (4 tests)
  - Vacuum state (separable)
  - Squeezed state (entangled)
  - Symplectic eigenvalues
  - Logarithmic negativity scaling

## Running Tests

### Basic Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_vortex_mps.py -v
pytest tests/test_entanglement.py -v

# Run specific test class
pytest tests/test_vortex_mps.py::TestMPSInitialization -v

# Run specific test
pytest tests/test_entanglement.py::TestLogNegativity::test_epr_state_positive_negativity -v
```

### Advanced Options

```bash
# Run with coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Parallel execution (faster)
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Run only fast tests (exclude slow)
pytest tests/ -m "not slow"

# Verbose output with print statements
pytest tests/ -v -s

# Only run failed tests from last run
pytest tests/ --lf

# Detailed failure output
pytest tests/ -vv --tb=long
```

### Markers

Tests are marked with custom markers:

```bash
# Run only slow tests
pytest tests/ -m slow

# Run only integration tests
pytest tests/ -m integration

# Run only GPU tests
pytest tests/ -m gpu
```

## Coverage Report

Generate HTML coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html  # View in browser
```

Expected coverage by module:
- `seemps_vortex/`: > 90%
- `belle2_analysis/`: > 85%
- `ibm_validation/`: > 80%

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Nightly builds

CI configuration: `.github/workflows/tests.yml`

## Writing New Tests

### Template

```python
import pytest
import numpy as np

class TestNewFeature:
    """Test description."""

    @pytest.fixture
    def setup_data(self):
        """Fixture for test data."""
        return some_data

    def test_basic_functionality(self, setup_data):
        """Test basic functionality."""
        result = my_function(setup_data)
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            my_function(invalid_input)

    @pytest.mark.slow
    def test_expensive_computation(self):
        """Test marked as slow."""
        result = expensive_function()
        assert result > 0
```

### Best Practices

1. **Use descriptive names**: `test_feature_behavior_condition`
2. **One assertion per test**: Focus on single behavior
3. **Use fixtures**: Share setup code with `@pytest.fixture`
4. **Parametrize**: Test multiple inputs with `@pytest.mark.parametrize`
5. **Mark slow tests**: Use `@pytest.mark.slow` for tests > 1 second
6. **Mock external dependencies**: Use `unittest.mock` or `pytest-mock`
7. **Test edge cases**: Zero, negative, infinity, empty inputs
8. **Check error handling**: Verify exceptions with `pytest.raises`

## Fixtures

Available in `conftest.py`:

- `reset_random_seed`: Ensures reproducibility
- `suppress_warnings`: Filters out known warnings
- `tolerance`: Standard numerical tolerance (1e-10)
- `temp_directory`: Temporary directory for outputs

## Debugging Failed Tests

```bash
# Run with debugger on failure
pytest tests/ --pdb

# Show local variables on failure
pytest tests/ --showlocals

# Capture print output
pytest tests/ -s

# Increase verbosity
pytest tests/ -vv
```

## Performance Profiling

```bash
# Profile test execution time
pytest tests/ --durations=10

# Profile with cProfile
pytest tests/ --profile

# Memory profiling (requires pytest-memprof)
pytest tests/ --memprof
```

## Troubleshooting

### Issue: Import errors

```bash
# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Issue: SeeMPS not found

```bash
# Install SeeMPS
pip install seemps2
```

### Issue: Tests hang

```bash
# Set timeout
pytest tests/ --timeout=60
```

### Issue: Random failures

- Check if test depends on random seed
- Use `@pytest.fixture(autouse=True)` to reset seed
- Increase numerical tolerance

## Resources

- **pytest docs**: https://docs.pytest.org/
- **Coverage.py**: https://coverage.readthedocs.io/
- **pytest plugins**: https://docs.pytest.org/en/latest/reference/plugin_list.html

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure > 90% coverage
3. Run full test suite before PR
4. Update this documentation

---

**Last Updated**: February 16, 2026
