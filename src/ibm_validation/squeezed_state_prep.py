"""
Squeezed State Preparation and Validation on IBM Quantum
=========================================================

Preparation and measurement of Two-Mode Squeezed Thermal States (TMST)
on IBM Quantum hardware for validation of theoretical predictions.

Validates:
- file:6 Theorem 4.3.1: Entanglement threshold for TMST
- file:9 Eq. (3): Collective TMST projection and entanglement entropy
- file:7: Non-Hermitian dynamics near exceptional points

Circuit implementations:
- Two-mode squeezing gates S(r) = exp[r(a†b† - ab)]
- Thermal state preparation via depolarizing channels
- Entanglement witnesses for CV systems mapped to qubits

Hardware constraints:
- Native gate set: {RZ, SX, X, CX}
- Coherence-limited circuit depth
- Shot-noise in entanglement estimation
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
import warnings

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Options, Session
    from qiskit.quantum_info import (
        SparsePauliOp, DensityMatrix, partial_trace, 
        state_fidelity, entropy, Statevector
    )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.circuit.library import UnitaryGate
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn("Qiskit not available, IBM Quantum validation disabled")

try:
    from scipy.linalg import expm, eigvalsh, sqrtm
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TMSTValidationResult:
    """Container for TMST validation results."""
    r_squeeze: float
    T_temperature: float
    n_thermal_theory: float
    EN_measured: float
    EN_theory: float
    EN_error: float
    is_entangled: bool
    threshold_verified: bool
    raw_counts: Dict[str, int]
    n_shots: int


class TMSTCircuitBuilder:
    """
    Builder for Two-Mode Squeezed Thermal State circuits.
    
    Implements quantum circuit approximation of continuous-variable TMST:
    |ψ_TMST⟩ = S(r) |thermal⟩
    
    where S(r) = exp[r(a†b† - ab)] is two-mode squeezing operator.
    """
    
    def __init__(self, n_qubits_per_mode: int = 1):
        """
        Args:
            n_qubits_per_mode: Number of qubits per oscillator mode (default 1)
        """
        self.n_qubits_per_mode = n_qubits_per_mode
        self.total_qubits = 2 * n_qubits_per_mode
    
    def build_tmst_circuit(self, r_squeeze: float, 
                          n_thermal: float = 0.0,
                          include_measurement: bool = True) -> QuantumCircuit:
        """
        Build quantum circuit for TMST preparation.
        
        Discrete approximation:
        1. Prepare thermal mixture via depolarization
        2. Apply two-mode squeezing via controlled rotations
        
        Args:
            r_squeeze: Squeezing parameter
            n_thermal: Thermal occupation (for mixed state simulation)
            include_measurement: Add measurement gates
        
        Returns:
            QuantumCircuit for TMST
        """
        n_total = self.total_qubits
        qr = QuantumRegister(n_total, 'q')
        cr = ClassicalRegister(n_total, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Mode A: qubits 0 to n_qubits_per_mode-1
        # Mode B: qubits n_qubits_per_mode to 2*n_qubits_per_mode-1
        
        # Step 1: Prepare thermal state (simplified: depolarizing channel)
        if n_thermal > 0:
            thermal_prob = n_thermal / (1 + n_thermal)
            
            for i in range(n_total):
                # Apply bit flip with thermal probability
                if np.random.rand() < thermal_prob:
                    qc.x(i)
        
        # Step 2: Apply two-mode squeezing
        # Decomposition: S(r) ≈ exp[-i r (XX + YY)/2]
        #
        # For single qubit per mode: controlled rotations
        theta_squeeze = 2 * np.arctan(np.tanh(r_squeeze))
        
        q_a = 0
        q_b = self.n_qubits_per_mode
        
        # Approximate squeezing via beam-splitter-like operation
       
