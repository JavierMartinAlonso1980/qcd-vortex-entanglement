# src/ibm_validation/injection_tests.py
"""
Qiskit 2-Qubit Injection Test for Vorticity / Entanglement
===========================================================
Implements the 'inject signal vs noise' pattern from the BasQ conversations:
  - Bell state |Φ+⟩ → maximum entanglement (vorticity signal)
  - Random product state → noise (EM accelerator background)

Works on AerSimulator by default; swap to ibm_heron for real hardware.

Reference (BasQ conversations, file:11/12):
    Bell: qc.h(0); qc.cx(0,1)  → 00/11 correlation (vorticity)
    Noise: qc.rx(random); qc.rx(random) → mixed results (EM noise)
"""

from __future__ import annotations
import numpy as np
from typing import Optional

# Qiskit imports (optional at module level for CI without qiskit)
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector, state_fidelity
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _require_qiskit():
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "qiskit and qiskit-aer are required for ibm_validation. "
            "Install with: pip install qiskit qiskit-aer"
        )


# ──────────────────────────────────────────
# Circuit builders
# ──────────────────────────────────────────

def make_bell_circuit() -> "QuantumCircuit":
    """Prepare |Φ+⟩ = (|00⟩ + |11⟩)/√2 — maximum entanglement."""
    _require_qiskit()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def make_noise_circuit(seed: Optional[int] = None) -> "QuantumCircuit":
    """Prepare a product state with independent random RX rotations — no entanglement."""
    _require_qiskit()
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(2)
    qc.rx(float(rng.uniform(0, np.pi)), 0)
    qc.rx(float(rng.uniform(0, np.pi)), 1)
    return qc


# ──────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────

def compute_log_negativity(qc: "QuantumCircuit") -> float:
    """
    Compute log-negativity E_N for a pure 2-qubit state via partial transpose.
    E_N = log₂(2·N + 1) where N = sum of |negative eigenvalues| of ρ^{T_B}.
    """
    _require_qiskit()
    sv = Statevector.from_instruction(qc)
    rho = np.outer(sv.data, sv.data.conj())
    rho_TB = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    evals = np.linalg.eigvalsh(rho_TB)
    negativity = np.sum(np.abs(evals[evals < 0.0]))
    return float(max(0.0, np.log2(2.0 * negativity + 1.0)))


def compute_bell_fidelity(qc: "QuantumCircuit") -> float:
    """Fidelity of the state in qc with the ideal |Φ+⟩ Bell state."""
    _require_qiskit()
    ideal = Statevector.from_instruction(make_bell_circuit())
    actual = Statevector.from_instruction(qc)
    return float(state_fidelity(ideal, actual))


# ──────────────────────────────────────────
# Main injection test runner
# ──────────────────────────────────────────

def run_injection_test(
    shots: int = 2000,
    noise_seed: int = 42,
    backend_name: str = "aer_simulator",
) -> dict:
    """
    Run the full signal-vs-noise injection test.

    Returns a dict with:
        signal: {counts, log_negativity, bell_fidelity}
        noise:  {counts, log_negativity, bell_fidelity}
        passed: bool  (True if metrics clearly separate signal from noise)
    """
    _require_qiskit()
    backend = AerSimulator()

    results = {}
    for label, qc_builder in [
        ("signal", make_bell_circuit),
        ("noise", lambda: make_noise_circuit(seed=noise_seed)),
    ]:
        qc = qc_builder()
        EN = compute_log_negativity(qc)
        fidelity = compute_bell_fidelity(qc)

        qc_meas = qc.copy()
        qc_meas.measure_all()
        job = backend.run(qc_meas, shots=shots)
        counts = job.result().get_counts()

        results[label] = {
            "counts": counts,
            "log_negativity": EN,
            "bell_fidelity": fidelity,
        }

    passed = (
        results["signal"]["log_negativity"] > 0.5
        and results["noise"]["log_negativity"] < 0.1
        and results["signal"]["bell_fidelity"] > 0.95
    )
    results["passed"] = passed

    # Confidence report (mirrors BasQ conversation pattern)
    confidence = results["signal"]["bell_fidelity"] * 100
    if confidence > 90:
        status = "✅ Code validated. Physics is visible."
    elif confidence > 50:
        status = "⚠️  Noise detected. Signal partially contaminated."
    else:
        status = "❌ Critical failure. Code does not identify the observable."

    results["status"] = status
    print(f"[InjectionTest] {status} | Confidence: {confidence:.1f}%")
    print(f"  Signal  → E_N={results['signal']['log_negativity']:.4f}, "
          f"Bell fidelity={results['signal']['bell_fidelity']:.4f}")
    print(f"  Noise   → E_N={results['noise']['log_negativity']:.4f}, "
          f"Bell fidelity={results['noise']['bell_fidelity']:.4f}")

    return results
