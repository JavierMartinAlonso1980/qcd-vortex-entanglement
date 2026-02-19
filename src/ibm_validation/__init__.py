"""
IBM Quantum Validation Package
===============================

Hardware validation of quantum entanglement predictions on IBM Quantum System One.

Validates theoretical predictions from:
- file:6: Two-Mode Squeezed Thermal State (TMST) entanglement threshold
- file:9: Collective squeezing and superradiant amplification
- file:7: Non-Hermitian dynamics and exceptional points

Modules:
--------
- squeezed_state_prep: TMST circuit preparation and measurement
- quantum_verification: Hardware verification protocols and benchmarks
- injection_tests: Signal-vs-noise injection tests (Bell state / AerSimulator)  ← NEW

IBM Quantum Platforms:
- IBM Quantum System One (on-premises systems)
- IBM Cloud quantum processors
- Qiskit Runtime primitives (Sampler, Estimator)

Hardware Requirements:
- Minimum 2 qubits (for two-mode TMST)
- Native gate set: {RZ, SX, X, CX}
- Coherence times: T1 > 100 μs, T2 > 50 μs
- Gate fidelity: > 99% (single-qubit), > 95% (two-qubit)

Author: [Your Name]
Date: February 2026
Version: 1.1.0
"""

# ── BUGFIX: typing imports missing in v1.0.0 ──────────────────────────────────
from typing import Dict, Optional

# ── Existing modules (unchanged) ──────────────────────────────────────────────
from .squeezed_state_prep import (
    SqueezeStateValidator,
    TMSTCircuitBuilder,
    EntanglementWitnessProtocol
)
from .quantum_verification import (
    HardwareVerifier,
    CoherenceAnalyzer,
    ErrorMitigationPipeline,
    BenchmarkSuite
)

# ── NEW: Injection tests module ───────────────────────────────────────────────
from .injection_tests import (
    make_bell_circuit,
    make_noise_circuit,
    compute_log_negativity,
    compute_bell_fidelity,
    run_injection_test,
)

__version__ = "1.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # ── Existing ──────────────────────────────────────────────────────────────
    # Squeezed state preparation
    'SqueezeStateValidator',
    'TMSTCircuitBuilder',
    'EntanglementWitnessProtocol',

    # Verification protocols
    'HardwareVerifier',
    'CoherenceAnalyzer',
    'ErrorMitigationPipeline',
    'BenchmarkSuite',

    # ── NEW ───────────────────────────────────────────────────────────────────
    # Injection tests
    'make_bell_circuit',
    'make_noise_circuit',
    'compute_log_negativity',
    'compute_bell_fidelity',
    'run_injection_test',
]

# ── El resto del fichero es IDÉNTICO al original ──────────────────────────────

# IBM Quantum backend specifications
IBM_BACKENDS = {
    'ibm_sherbrooke': {
        'num_qubits': 127,
        'processor': 'Eagle r3',
        'topology': 'heavy-hex',
        'typical_T1': 150e-6,
        'typical_T2': 100e-6,
        'gate_time_cx': 500e-9,
        'gate_time_sx': 100e-9,
    },
    'ibm_kyiv': {
        'num_qubits': 127,
        'processor': 'Eagle r3',
        'topology': 'heavy-hex',
        'typical_T1': 200e-6,
        'typical_T2': 120e-6,
        'gate_time_cx': 450e-9,
        'gate_time_sx': 90e-9,
    },
    'ibm_brisbane': {
        'num_qubits': 127,
        'processor': 'Eagle r3',
        'topology': 'heavy-hex',
        'typical_T1': 180e-6,
        'typical_T2': 110e-6,
        'gate_time_cx': 480e-9,
        'gate_time_sx': 95e-9,
    },
    'ibm_osaka': {
        'num_qubits': 127,
        'processor': 'Eagle r3',
        'topology': 'heavy-hex',
        'typical_T1': 170e-6,
        'typical_T2': 105e-6,
        'gate_time_cx': 490e-9,
        'gate_time_sx': 100e-9,
    },
}

# Validation protocols (matching theoretical framework)
VALIDATION_PROTOCOLS = {
    'tmst_threshold': {
        'description': 'Validate Theorem 4.3.1 (file:6): Entanglement threshold',
        'r_range': (0.1, 2.0),
        'T_range': (0.1, 2.0),
        'expected_accuracy': 0.85,
        'min_shots': 5000,
    },
    'superradiant_gain': {
        'description': 'Measure superradiant amplification (file:9 Eq. 4)',
        'omega_range': (0.5, 2.0),
        'gamma_SR': 0.5,
        'gamma_loss': 0.1,
        'min_shots': 8000,
    },
    'exceptional_point': {
        'description': 'Detect EP2 signature in non-Hermitian dynamics (file:7)',
        'r_scan_points': 50,
        'petermann_factor_threshold': 10.0,
        'min_shots': 10000,
    },
    'collective_projection': {
        'description': 'Multi-qubit collective vortex projection (file:9 Section 5.4)',
        'n_qubits': 4,
        'coincidence_window': 10e-9,
        'min_shots': 15000,
    },
}

# Error mitigation strategies
ERROR_MITIGATION_CONFIG = {
    'readout_mitigation': True,
    'zero_noise_extrapolation': True,
    'dynamical_decoupling': True,
    'gate_twirling': False,
    'resilience_level': 2,
}

# Citation information
CITATION_INFO = """
If you use this validation framework, please cite:

@software{qcd_vortex_ibm_2026,
  author = {Your Name},
  title = {IBM Quantum Validation of QCD Vortex Entanglement},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/yourusername/qcd-vortex-entanglement}
}

IBM Quantum acknowledgment:
"We acknowledge the use of IBM Quantum services for this work.
The views expressed are those of the authors, and do not reflect
the official policy or position of IBM or the IBM Quantum team."
"""


def print_backend_info(backend_name: str = 'ibm_sherbrooke'):
    """Print IBM Quantum backend specifications."""
    if backend_name not in IBM_BACKENDS:
        print(f"Backend {backend_name} not found in database.")
        print(f"Available backends: {list(IBM_BACKENDS.keys())}")
        return
    specs = IBM_BACKENDS[backend_name]
    print("="*70)
    print(f"IBM Quantum Backend: {backend_name}")
    print("="*70)
    for key, value in specs.items():
        if 'T1' in key or 'T2' in key:
            print(f"  {key:25s}: {value*1e6:.1f} μs")
        elif 'gate_time' in key:
            print(f"  {key:25s}: {value*1e9:.1f} ns")
        else:
            print(f"  {key:25s}: {value}")
    print("="*70)


def estimate_circuit_duration(
    n_qubits: int,
    n_cx_gates: int,
    backend: str = 'ibm_sherbrooke',
) -> float:
    """Estimate total circuit duration including decoherence effects."""
    specs = IBM_BACKENDS.get(backend, IBM_BACKENDS['ibm_sherbrooke'])
    n_single_qubit = 4 * n_qubits
    return (n_single_qubit * specs['gate_time_sx'] +
            n_cx_gates * specs['gate_time_cx'])


def check_coherence_limit(
    circuit_duration: float,
    backend: str = 'ibm_sherbrooke',
) -> Dict[str, float]:          # ← BUGFIX: Dict ahora importado correctamente
    """Check if circuit duration is within coherence limits."""
    specs = IBM_BACKENDS.get(backend, IBM_BACKENDS['ibm_sherbrooke'])
    T1_ratio = circuit_duration / specs['typical_T1']
    T2_ratio = circuit_duration / specs['typical_T2']
    return {
        'circuit_duration_us': circuit_duration * 1e6,
        'T1_ratio': T1_ratio,
        'T2_ratio': T2_ratio,
        'within_T1_limit': T1_ratio < 0.5,
        'within_T2_limit': T2_ratio < 0.5,
        'recommended': T1_ratio < 0.3 and T2_ratio < 0.3,
    }


def get_optimal_backend(n_qubits: int, circuit_depth: int) -> str:
    """Recommend optimal backend based on circuit requirements."""
    suitable_backends = {
        name: specs for name, specs in IBM_BACKENDS.items()
        if specs['num_qubits'] >= n_qubits
    }
    if not suitable_backends:
        return 'ibm_sherbrooke'
    scores = {
        name: 0.4 * specs['typical_T1'] + 0.6 * specs['typical_T2']
        for name, specs in suitable_backends.items()
    }
    return max(scores, key=scores.get)


def initialize_ibm_quantum(
    token: Optional[str] = None,
    channel: str = 'ibm_quantum',
):
    """Initialize IBM Quantum connection."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    if token:
        service = QiskitRuntimeService(channel=channel, token=token)
    else:
        try:
            service = QiskitRuntimeService(channel=channel)
        except Exception as e:
            print(f"Failed to initialize IBM Quantum service: {e}")
            print("Please set QISKIT_IBM_TOKEN environment variable or provide token explicitly")
            raise
    return service


def print_module_info():
    """Print IBM Quantum validation module information."""
    print("="*70)
    print("IBM Quantum Validation Module")
    print("="*70)
    print(f"Version: {__version__}")
    print(f"\nAvailable backends: {len(IBM_BACKENDS)}")
    for backend in IBM_BACKENDS.keys():
        print(f"  - {backend}")
    print(f"\nValidation protocols: {len(VALIDATION_PROTOCOLS)}")
    for protocol in VALIDATION_PROTOCOLS.keys():
        print(f"  - {protocol}")
    print("="*70)


if __name__ == "__main__":
    print_module_info()
