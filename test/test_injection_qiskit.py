# test/test_injection_qiskit.py
"""
Qiskit 2-qubit injection test suite.
Verifies that log-negativity and Bell fidelity correctly distinguish
entangled states from product-state noise.

Skipped automatically if qiskit / qiskit-aer are not installed.
"""

import pytest

try:
    from qiskit import QuantumCircuit  # noqa: F401
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

skip_if_no_qiskit = pytest.mark.skipif(
    not QISKIT_AVAILABLE,
    reason="qiskit / qiskit-aer not installed"
)


@skip_if_no_qiskit
def test_bell_state_log_negativity():
    """Bell state |Φ+⟩ must have E_N ≈ 1.0 (maximum for 2 qubits)."""
    from src.ibm_validation.injection_tests import (
        make_bell_circuit, compute_log_negativity
    )
    EN = compute_log_negativity(make_bell_circuit())
    assert EN > 0.95, f"Bell state E_N should be ≈1.0, got {EN:.4f}"


@skip_if_no_qiskit
def test_product_noise_log_negativity():
    """Product (noise) states must have E_N ≈ 0 regardless of angles."""
    from src.ibm_validation.injection_tests import (
        make_noise_circuit, compute_log_negativity
    )
    for seed in [0, 7, 42, 99, 123]:
        qc = make_noise_circuit(seed=seed)
        EN = compute_log_negativity(qc)
        assert EN < 1e-9, f"Seed {seed}: noise E_N should be 0, got {EN:.6f}"


@skip_if_no_qiskit
def test_bell_fidelity_is_one():
    """Bell circuit fidelity with ideal |Φ+⟩ should be exactly 1."""
    from src.ibm_validation.injection_tests import (
        make_bell_circuit, compute_bell_fidelity
    )
    fidelity = compute_bell_fidelity(make_bell_circuit())
    assert abs(fidelity - 1.0) < 1e-9, f"Bell fidelity should be 1.0, got {fidelity}"


@skip_if_no_qiskit
def test_injection_signal_clearly_above_noise():
    """
    Core injection test: E_N(signal) >> E_N(noise).
    This mirrors the BasQ vorticity-vs-EM-noise validation pattern.
    """
    from src.ibm_validation.injection_tests import (
        make_bell_circuit, make_noise_circuit, compute_log_negativity
    )
    EN_signal = compute_log_negativity(make_bell_circuit())
    EN_noise = compute_log_negativity(make_noise_circuit(seed=42))

    assert EN_signal > EN_noise + 0.8, (
        f"Signal and noise not clearly separated: "
        f"E_N(signal)={EN_signal:.4f}, E_N(noise)={EN_noise:.4f}"
    )


@skip_if_no_qiskit
def test_full_injection_run_passes():
    """End-to-end injection test: run_injection_test() must pass."""
    from src.ibm_validation.injection_tests import run_injection_test
    results = run_injection_test(shots=2000, noise_seed=42)
    assert results["passed"], (
        f"Injection test failed.\n"
        f"  Signal E_N = {results['signal']['log_negativity']:.4f}\n"
        f"  Noise  E_N = {results['noise']['log_negativity']:.4f}\n"
        f"  Status: {results['status']}"
    )
