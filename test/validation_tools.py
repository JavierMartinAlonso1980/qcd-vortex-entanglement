# test/validation_tools.py
"""
Injection-Test Validation Tools
================================
Shared helpers for all injection-style tests:
- TMST Gaussian entanglement (point 4.2 / 4.5)
- Qiskit 2-qubit Bell vs noise (point 4.1 / 4.2)

These functions implement the "inject known signal vs pure noise,
then verify metrics respond correctly" pattern described in the
BasQ conversations.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt

# ─────────────────────────────────────────────
# TMST injection helpers
# ─────────────────────────────────────────────

def make_tmst_signal(
    T: float,
    delta_above: float = 0.3,
    omega: float = 1.0,
) -> tuple[float, float]:
    """
    Return (r_signal, n_bar) for a squeezing value ABOVE the threshold.
    'delta_above' sets how far above r_c we place the signal.
    """
    from src.seemps_vortex.tmst_threshold import bose_einstein, critical_squeezing
    n_bar = float(bose_einstein(np.array([T]), omega=omega)[0])
    rc = float(critical_squeezing(n_bar))
    return rc + delta_above, n_bar


def make_tmst_noise(
    T: float,
    alpha_below: float = 0.5,
    omega: float = 1.0,
) -> tuple[float, float]:
    """
    Return (r_noise, n_bar) for a squeezing value BELOW the threshold.
    'alpha_below' is the fraction of r_c to use (< 1 → separable).
    """
    from src.seemps_vortex.tmst_threshold import bose_einstein, critical_squeezing
    n_bar = float(bose_einstein(np.array([T]), omega=omega)[0])
    rc = float(critical_squeezing(n_bar))
    return alpha_below * rc, n_bar


# ─────────────────────────────────────────────
# Qiskit 2-qubit injection helpers
# ─────────────────────────────────────────────

def bell_state_circuit():
    """
    Returns a 2-qubit QuantumCircuit preparing |Φ+⟩ = (|00⟩+|11⟩)/√2.
    This is the 'maximum entanglement' injection signal.
    Requires qiskit.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def product_noise_circuit(seed: int | None = None):
    """
    Returns a 2-qubit QuantumCircuit with independent random RX rotations.
    This is the 'no entanglement / pure noise' injection.
    """
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(2)
    qc.rx(rng.uniform(0, np.pi), 0)
    qc.rx(rng.uniform(0, np.pi), 1)
    return qc


def log_negativity_from_statevector(qc) -> float:
    """
    Compute log-negativity E_N for a pure 2-qubit state via partial transpose.
    Works directly on a QuantumCircuit (no measurement needed).
    """
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    rho = np.outer(sv.data, sv.data.conj())

    # Partial transpose on qubit 1 (reshape → swap indices → reshape back)
    rho_TB = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    evals = np.linalg.eigvalsh(rho_TB)
    negativity = np.sum(np.abs(evals[evals < 0.0]))
    EN = np.log2(2.0 * negativity + 1.0)
    return float(max(0.0, EN))


def state_fidelity_to_bell(qc) -> float:
    """
    Fidelity of the state in qc with the ideal Bell state |Φ+⟩.
    """
    from qiskit.quantum_info import Statevector, state_fidelity
    ideal = Statevector.from_instruction(bell_state_circuit())
    actual = Statevector.from_instruction(qc)
    return float(state_fidelity(ideal, actual))
