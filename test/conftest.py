"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import warnings


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        'markers', 'slow: marks tests as slow (deselect with \'-m "not slow"\')'
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    # ── NEW ───────────────────────────────────────────────────────────────────
    config.addinivalue_line(
        "markers", "quantum: marks tests that require qiskit and qiskit-aer"
    )


# ── Existing fixtures (unchanged) ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress specific warnings during tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for assertions."""
    return 1e-10


@pytest.fixture
def temp_directory(tmp_path):
    """Create temporary directory for test outputs."""
    return tmp_path


# ── NEW: Qiskit availability fixture ─────────────────────────────────────────

@pytest.fixture(scope="session")
def qiskit_available():
    """
    Session-scoped fixture: returns True if qiskit + qiskit-aer are installed.
    Used by test_injection_qiskit.py to skip gracefully without import errors.

    Usage in tests:
        def test_something(qiskit_available):
            if not qiskit_available:
                pytest.skip("qiskit not installed")
    """
    try:
        import qiskit          # noqa: F401
        from qiskit_aer import AerSimulator  # noqa: F401
        return True
    except ImportError:
        return False
