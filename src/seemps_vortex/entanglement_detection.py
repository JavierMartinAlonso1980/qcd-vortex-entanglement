"""
Entanglement Detection & Measures
==================================

Implements entanglement quantification for QCD vortex systems:
- Log-negativity (file:6 Eq. 2.4)
- von Neumann entropy
- Concurrence (for Belle II τ⁺τ⁻ pairs, file:3)
- Entanglement witnesses

Mathematical framework:
- file:6 Appendix A: Gaussian state entanglement
- file:3 Eq. (17): Concurrence for fermionic systems
- file:9 Section 2.3: TMSS entanglement entropy
"""

import numpy as np
from scipy.linalg import eigvalsh, sqrtm, logm
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional, Dict, List
import warnings


class EntanglementMeasures:
    """
    Collection of entanglement quantifiers for bipartite systems.
    
    Supports both:
    - Gaussian continuous-variable states (covariance matrix)
    - Discrete qubit systems (density matrix)
    """
    
    def __init__(self, state_type: str = 'gaussian'):
        """
        Args:
            state_type: 'gaussian' for CV states, 'discrete' for qubits
        """
        if state_type not in ['gaussian', 'discrete']:
            raise ValueError("state_type must be 'gaussian' or 'discrete'")
        
        self.state_type = state_type
    
    def log_negativity(self, state: np.ndarray, 
                       partition: Optional[List[int]] = None) -> float:
        """
        Compute log-negativity E_N (file:6 Eq. 2.4).
        
        E_N = log₂(||ρ^{T_B}||₁)
        
        Args:
            state: Covariance matrix (Gaussian) or density matrix (discrete)
            partition: Subsystem B indices (None = bipartite split)
        
        Returns:
            Log-negativity in bits
        """
        if self.state_type == 'gaussian':
            return self._log_negativity_gaussian(state)
        else:
            return self._log_negativity_discrete(state, partition)
    
    def _log_negativity_gaussian(self, sigma: np.ndarray) -> float:
        """
        Log-negativity for Gaussian states via symplectic eigenvalues.
        
        For two-mode Gaussian state with covariance σ:
        1. Partial transpose: σ^{T_B} (flip sign of p_B)
        2. Compute smallest symplectic eigenvalue ν₋
        3. E_N = max(0, -log₂(2ν₋))
        
        Reference: file:6 Appendix A.3
        """
        # Partial transpose (flip momentum of mode B)
        sigma_TB = sigma.copy()
        n_modes = sigma.shape[0] // 2
        
        # Flip sign of momentum quadrature of second mode
        sigma_TB[n_modes+1::2, :] *= -1
        sigma_TB[:, n_modes+1::2] *= -1
        
        # Compute symplectic eigenvalues
        nu_minus = self._smallest_symplectic_eigenvalue(sigma_TB)
        
        # Log-negativity
        EN = max(0.0, -np.log2(2 * nu_minus))
        
        return EN
    
    def _smallest_symplectic_eigenvalue(self, sigma: np.ndarray) -> float:
        """
        Compute smallest symplectic eigenvalue of covariance matrix.
        
        Algorithm:
        1. Construct Ω σ (symplectic form times covariance)
        2. Compute eigenvalues of (Ω σ)²
        3. Take square root of smallest magnitude
        """
        n = sigma.shape[0] // 2
        
        # Symplectic form Ω
        Omega = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-np.eye(n), np.zeros((n, n))]
        ])
        
        # Compute eigenvalues of (Ω σ)²
        Omega_sigma = Omega @ sigma
        eigenvalues = eigvalsh(Omega_sigma @ Omega_sigma)
        
        # Smallest symplectic eigenvalue
        nu_minus = np.sqrt(np.min(np.abs(eigenvalues)))
        
        return nu_minus
    
    def _log_negativity_discrete(self, rho: np.ndarray, 
                                 partition: Optional[List[int]] = None) -> float:
        """
        Log-negativity for discrete density matrices.
        
        Args:
            rho: Density matrix
            partition: Indices of subsystem B
        
        Returns:
            Log-negativity
        """
        # Partial transpose
        rho_TB = self._partial_transpose(rho, partition)
        
        # Trace norm ||ρ^{T_B}||₁
        eigenvalues = eigvalsh(rho_TB)
        trace_norm = np.sum(np.abs(eigenvalues))
        
        EN = np.log2(trace_norm)
        
        return max(0.0, EN)
    
    def _partial_transpose(self, rho: np.ndarray, 
                          partition: Optional[List[int]] = None) -> np.ndarray:
        """Compute partial transpose with respect to subsystem B."""
        dim = rho.shape[0]
        
        if partition is None:
            # Default: equal bipartition
            dim_A = int(np.sqrt(dim))
            dim_B = dim // dim_A
        else:
            dim_B = len(partition)
            dim_A = dim // dim_B
        
        # Reshape and transpose
        rho_reshaped = rho.reshape(dim_A, dim_B, dim_A, dim_B)
        rho_TB = rho_reshaped.transpose(0, 3, 2, 1).reshape(dim, dim)
        
        return rho_TB
    
    def von_neumann_entropy(self, state: np.ndarray, 
                           subsystem: Optional[str] = None) -> float:
        """
        Compute von Neumann entropy S = -Tr(ρ log ρ).
        
        For Gaussian states: S computed from symplectic eigenvalues
        For discrete: S computed from density matrix eigenvalues
        
        Args:
            state: Covariance or density matrix
            subsystem: 'A' or 'B' for reduced entropy (None = full system)
        
        Returns:
            Entropy in bits
        """
        if self.state_type == 'gaussian':
            return self._entropy_gaussian(state, subsystem)
        else:
            return self._entropy_discrete(state)
    
    def _entropy_gaussian(self, sigma: np.ndarray, 
                         subsystem: Optional[str] = None) -> float:
        """
        von Neumann entropy for Gaussian states (file:6 Appendix).
        
        S(ν) = (ν + 1/2) log(ν + 1/2) - (ν - 1/2) log(ν - 1/2)
        
        where ν is symplectic eigenvalue.
        """
        # Extract subsystem covariance if needed
        if subsystem == 'A':
            sigma_sub = sigma[:2, :2]  # First mode
        elif subsystem == 'B':
            sigma_sub = sigma[2:, 2:]  # Second mode
        else:
            sigma_sub = sigma
        
        # Symplectic eigenvalues
        nu_values = self._symplectic_spectrum(sigma_sub)
        
        # Sum entropy contributions
        S_total = 0.0
        for nu in nu_values:
            if nu <= 0.5:
                continue  # Vacuum contribution is zero
            
            S_plus = (nu + 0.5) * np.log2(nu + 0.5)
            S_minus = (nu - 0.5) * np.log2(abs(nu - 0.5))
            
            S_total += (S_plus - S_minus)
        
        return S_total
    
    def _symplectic_spectrum(self, sigma: np.ndarray) -> np.ndarray:
        """Compute all symplectic eigenvalues."""
        n = sigma.shape[0] // 2
        
        Omega = np.block([
            [np.zeros((n, n)), np.eye(n)],
            [-np.eye(n), np.zeros((n, n))]
        ])
        
        eigenvalues = eigvalsh(Omega @ sigma @ Omega @ sigma)
        nu_spectrum = np.sqrt(np.abs(eigenvalues[eigenvalues > 1e-12]))
        
        return nu_spectrum
    
    def _entropy_discrete(self, rho: np.ndarray) -> float:
        """von Neumann entropy for discrete density matrix."""
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-16]  # Remove zeros
        
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return S
    
    def mutual_information(self, state: np.ndarray) -> float:
        """
        Compute mutual information I(A:B) = S(A) + S(B) - S(AB).
        
        Measures total correlations (classical + quantum).
        
        Args:
            state: Bipartite state
        
        Returns:
            Mutual information in bits
        """
        S_A = self.von_neumann_entropy(state, subsystem='A')
        S_B = self.von_neumann_entropy(state, subsystem='B')
        S_AB = self.von_neumann_entropy(state, subsystem=None)
        
        I_AB = S_A + S_B - S_AB
        
        return max(0.0, I_AB)


class LogNegativity:
    """
    Specialized log-negativity calculator with optimizations.
    
    Fast computation for large-scale MPS simulations.
    """
    
    @staticmethod
    def from_tmst(r_squeeze: float, n_thermal: float) -> float:
        """
        Analytic log-negativity for TMST (file:6 Eq. A.4).
        
        E_N = max(0, -log₂(2ν₋))
        where ν₋ = (n + 1/2) exp(-2r)
        
        Args:
            r_squeeze: Squeezing parameter
            n_thermal: Thermal occupation
        
        Returns:
            Log-negativity
        """
        nu_minus = (n_thermal + 0.5) * np.exp(-2 * r_squeeze)
        
        EN = max(0.0, -np.log2(2 * nu_minus))
        
        return EN
    
    @staticmethod
    def from_covariance(sigma: np.ndarray) -> float:
        """Fast computation from covariance matrix."""
        measures = EntanglementMeasures(state_type='gaussian')
        return measures.log_negativity(sigma)
    
    @staticmethod
    def threshold_temperature(r_squeeze: float, omega: float = 1.0) -> float:
        """
        Compute critical temperature for entanglement (file:6 Eq. 4.7).
        
        T_c = ω / ln(1 + 1/n_c)
        where n_c = 1/2 (e^{2r} - 1)^{-1}
        
        Args:
            r_squeeze: Squeezing parameter
            omega: Mode frequency
        
        Returns:
            Critical temperature T_c
        """
        n_critical = 0.5 / (np.exp(2 * r_squeeze) - 1.0)
        
        if n_critical <= 0:
            return 0.0
        
        T_c = omega / np.log(1.0 + 1.0/n_critical)
        
        return T_c


class VonNeumannEntropy:
    """
    von Neumann entropy calculator.
    
    Handles both pure and mixed states.
    """
    
    @staticmethod
    def from_density_matrix(rho: np.ndarray) -> float:
        """Standard S = -Tr(ρ log ρ)."""
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-16]
        
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    @staticmethod
    def from_mps_bipartition(chi_values: np.ndarray) -> float:
        """
        Compute S from MPS Schmidt coefficients χ.
        
        S = -Σ χ² log χ²
        
        Args:
            chi_values: Schmidt coefficients (singular values)
        
        Returns:
            Entanglement entropy
        """
        chi_squared = chi_values ** 2
        chi_squared = chi_squared / np.sum(chi_squared)  # Normalize
        
        chi_squared = chi_squared[chi_squared > 1e-16]
        
        S = -np.sum(chi_squared * np.log2(chi_squared))
        
        return S
    
    @staticmethod
    def renyi_entropy(rho: np.ndarray, alpha: float = 2.0) -> float:
        """
        Rényi entropy S_α = 1/(1-α) log Tr(ρ^α).
        
        α=1: von Neumann entropy (limit)
        α=2: Collision entropy (used in experiments)
        
        Args:
            rho: Density matrix
            alpha: Rényi parameter (α > 0, α ≠ 1)
        
        Returns:
            Rényi entropy in bits
        """
        if abs(alpha - 1.0) < 1e-8:
            return VonNeumannEntropy.from_density_matrix(rho)
        
        # Tr(ρ^α) = Σ λ^α
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-16]
        
        S_alpha = np.log2(np.sum(eigenvalues ** alpha)) / (1.0 - alpha)
        
        return S_alpha


class ConcurrenceEstimator:
    """
    Concurrence calculator for fermionic systems (Belle II τ⁺τ⁻).
    
    Implements algorithm from file:3 Eq. (17).
    """
    
    @staticmethod
    def from_density_matrix(rho: np.ndarray) -> float:
        """
        Compute concurrence C = max(0, λ₁ - λ₂ - λ₃ - λ₄).
        
        For two-qubit system:
        1. Construct R = √(√ρ ρ̃ √ρ)
        2. λ_i = eigenvalues of R in decreasing order
        3. C = max(0, λ₁ - λ₂ - λ₃ - λ₄)
        
        Args:
            rho: 4×4 density matrix for two qubits
        
        Returns:
            Concurrence 0 ≤ C ≤ 1
        """
        if rho.shape != (4, 4):
            raise ValueError("Concurrence requires 4×4 density matrix (2 qubits)")
        
        # Spin-flipped density matrix ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
        sigma_y = np.array([[0, -1j], [1j, 0]])
        flip_operator = np.kron(sigma_y, sigma_y)
        
        rho_tilde = flip_operator @ rho.conj() @ flip_operator
        
        # Construct R = √(√ρ ρ̃ √ρ)
        sqrt_rho = sqrtm(rho)
        R = sqrt_rho @ rho_tilde @ sqrt_rho
        
        # Eigenvalues in decreasing order
        lambdas = np.sort(np.sqrt(np.maximum(eigvalsh(R), 0)))[::-1]
        
        # Concurrence
        C = max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
        
        return C
    
    @staticmethod
    def from_helicity_amplitudes(psi_plus_plus: complex,
                                 psi_plus_minus: complex,
                                 psi_minus_plus: complex,
                                 psi_minus_minus: complex) -> float:
        """
        Compute concurrence from helicity basis (file:3 Section 3).
        
        For τ⁺τ⁻ pairs:
        |ψ⟩ = α|++⟩ + β|+-⟩ + γ|-+⟩ + δ|--⟩
        
        Args:
            psi_****: Helicity amplitudes
        
        Returns:
            Concurrence
        """
        # Construct density matrix
        psi = np.array([psi_plus_plus, psi_plus_minus, 
                       psi_minus_plus, psi_minus_minus])
        
        # Normalize
        psi = psi / np.linalg.norm(psi)
        
        # Pure state density matrix
        rho = np.outer(psi, psi.conj())
        
        return ConcurrenceEstimator.from_density_matrix(rho)
    
    @staticmethod
    def entanglement_of_formation(C: float) -> float:
        """
        Compute entanglement of formation from concurrence.
        
        E(C) = h((1 + √(1-C²))/2)
        where h(x) = -x log x - (1-x) log(1-x)
        
        Args:
            C: Concurrence
        
        Returns:
            Entanglement of formation in bits
        """
        if C < 0 or C > 1:
            warnings.warn("Concurrence out of range [0,1], clipping")
            C = np.clip(C, 0, 1)
        
        if C == 0:
            return 0.0
        
        # Binary entropy function
        def h(x):
            if x <= 0 or x >= 1:
                return 0.0
            return -x * np.log2(x) - (1-x) * np.log2(1-x)
        
        x = (1 + np.sqrt(1 - C**2)) / 2
        
        return h(x)


# Utility functions

def entanglement_witness(observable: np.ndarray, rho: np.ndarray) -> float:
    """
    Evaluate entanglement witness ⟨W⟩ = Tr(W ρ).
    
    If ⟨W⟩ < 0, state is entangled.
    
    Args:
        observable: Witness operator W
        rho: Density matrix
    
    Returns:
        Expectation value
    """
    witness_value = np.trace(observable @ rho).real
    
    return witness_value


def ppt_criterion(sigma: np.ndarray) -> bool:
    """
    Check PPT (Positive Partial Transpose) criterion for Gaussian states.
    
    State is separable iff σ^{T_B} satisfies uncertainty relation.
    
    Args:
        sigma: Covariance matrix
    
    Returns:
        True if PPT holds (separable), False if violated (entangled)
    """
    measures = EntanglementMeasures(state_type='gaussian')
    
    # Smallest symplectic eigenvalue of partial transpose
    sigma_TB = sigma.copy()
    n = sigma.shape[0] // 2
    sigma_TB[n+1::2, :] *= -1
    sigma_TB[:, n+1::2] *= -1
    
    nu_minus = measures._smallest_symplectic_eigenvalue(sigma_TB)
    
    # PPT holds if ν₋ ≥ 1/2
    return (nu_minus >= 0.5)


# Example usage
if __name__ == "__main__":
    print("=== Entanglement Detection Examples ===\n")
    
    # Test 1: Gaussian log-negativity
    print("1. Gaussian TMST State")
    r = 1.2
    n_T = 0.1
    
    EN_analytic = LogNegativity.from_tmst(r, n_T)
    T_critical = LogNegativity.threshold_temperature(r)
    
    print(f"   r = {r}, n(T) = {n_T}")
    print(f"   Log-negativity: {EN_analytic:.4f}")
    print(f"   Critical temperature: {T_critical:.4f}\n")
    
    # Test 2: Bell state concurrence
    print("2. Bell State (|Φ+⟩)")
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_bell = np.outer(bell_state, bell_state.conj())
    
    C_bell = ConcurrenceEstimator.from_density_matrix(rho_bell)
    E_bell = ConcurrenceEstimator.entanglement_of_formation(C_bell)
    
    print(f"   Concurrence: {C_bell:.4f}")
    print(f"   Entanglement of formation: {E_bell:.4f} ebits\n")
    
    # Test 3: von Neumann entropy
    print("3. Mixed State Entropy")
    p = 0.7  # Mixture parameter
    rho_mixed = p * rho_bell + (1-p) * np.eye(4)/4
    
    S_vn = VonNeumannEntropy.from_density_matrix(rho_mixed)
    S_2 = VonNeumannEntropy.renyi_entropy(rho_mixed, alpha=2.0)
    
    print(f"   von Neumann entropy: {S_vn:.4f} bits")
    print(f"   Rényi-2 entropy: {S_2:.4f} bits\n")
    
    # Test 4: PPT criterion
    print("4. PPT Criterion for Gaussian States")
    
    # Separable state (low squeezing)
    sigma_sep = CollectiveSqueezing(0.2, N_modes=2).generate_covariance_matrix(0.5)
    is_sep = ppt_criterion(sigma_sep)
    print(f"   Low squeezing: Separable = {is_sep}")
    
    # Entangled state (high squeezing)
    sigma_ent = CollectiveSqueezing(1.5, N_modes=2).generate_covariance_matrix(0.1)
    is_ent = not ppt_criterion(sigma_ent)
    print(f"   High squeezing: Entangled = {is_ent}")
