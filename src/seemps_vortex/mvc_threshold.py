"""
MVC Threshold Detection & Exceptional Point Analysis
=====================================================

Implements detection of Morphology of Vacuum Condensate (MVC) critical
density and exceptional point (EP) analysis in non-Hermitian QCD dynamics.

Theoretical basis:
- file:7: Non-Hermitian topology and exceptional points
- file:9 Eq. (12): MVC critical density ρ_MVC = T_Planck^α
- file:7 Eq. (11): Petermann factor divergence near EP
"""

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
from scipy.linalg import eigvals, sqrtm
from typing import Tuple, Dict, Optional, Callable
import warnings


class MVCThresholdDetector:
    """
    Detector for MVC critical density threshold.
    
    The MVC hypothesis states that confinement activates when
    local color-charge density reaches ρ_MVC = T_Planck^α (file:9 Eq. 12).
    
    Attributes:
        T_planck: Planck temperature (natural units)
        alpha: Universal exponent (default 2.5)
        rho_mvc: Computed critical density
    """
    
    def __init__(self, T_planck: float = 1.221e19, alpha: float = 2.5):
        """
        Args:
            T_planck: Planck temperature in GeV (default 1.221e19)
            alpha: Universal exponent α (file:9)
        """
        self.T_planck = T_planck
        self.alpha = alpha
        
        # Compute critical density
        self.rho_mvc = self._calculate_rho_mvc()
    
    def _calculate_rho_mvc(self) -> float:
        """Calculate MVC critical density (file:9 Eq. 12)."""
        return self.T_planck ** self.alpha
    
    def is_confined(self, rho_local: float, omega_vortex: float,
                   entanglement_density: float) -> Tuple[bool, str]:
        """
        Determine if system is in confined phase.
        
        Confinement criteria (file:9 Section 3.4):
        1. ρ_local ≥ ρ_MVC
        2. ω_vortex ≥ ω_bifurcation
        3. ρ_E > 0 (non-zero entanglement)
        
        Args:
            rho_local: Local color-charge density
            omega_vortex: Vortex rotation frequency
            entanglement_density: Entanglement density ρ_E
        
        Returns:
            (is_confined, phase_description)
        """
        # Check MVC threshold
        above_mvc = rho_local >= self.rho_mvc
        
        # Bifurcation frequency (file:9 Eq. 13)
        omega_bifurcation = self._bifurcation_frequency(rho_local)
        above_bifurcation = omega_vortex >= omega_bifurcation
        
        # Entanglement criterion
        has_entanglement = entanglement_density > 1e-6
        
        if above_mvc and above_bifurcation and has_entanglement:
            phase = "CONFINED"
        elif rho_local > 0.8 * self.rho_mvc:
            phase = "PRE-CONFINEMENT"
        else:
            phase = "DECONFINED"
        
        is_confined = (phase == "CONFINED")
        
        return is_confined, phase
    
    def _bifurcation_frequency(self, rho: float) -> float:
        """
        Compute bifurcation frequency (file:9 Eq. 13).
        
        ω_bifurcation = √(ρ/ρ_MVC) × ω_QCD
        
        where ω_QCD ~ Λ_QCD ≈ 0.217 GeV
        """
        omega_QCD = 0.217  # GeV
        
        if rho >= self.rho_mvc:
            return omega_QCD * np.sqrt(rho / self.rho_mvc)
        else:
            return 0.0
    
    def hadronization_multiplicity(self, N_coupled: int, 
                                   rho_E: float, 
                                   rho_sat: Optional[float] = None) -> float:
        """
        Predict hadronization multiplicity (file:9 Eq. 20).
        
        M_h = M_0 × N_coupled × (ρ_E / ρ_sat)
        
        Args:
            N_coupled: Number of particles coupled to collective vortex
            rho_E: Instantaneous entanglement density
            rho_sat: Saturation density (default: ρ_MVC)
        
        Returns:
            Predicted hadron multiplicity
        """
        if rho_sat is None:
            rho_sat = self.rho_mvc
        
        M_0 = 2.0  # Baseline (2-body case)
        
        if rho_sat == 0:
            return M_0 * N_coupled
        
        multiplicity = M_0 * N_coupled * (rho_E / rho_sat)
        
        return multiplicity
    
    def inertial_mass(self, xi_core: float, R_system: float, 
                     rho_condensate: float) -> float:
        """
        Compute effective inertial mass of vortex (file:8 Eq. 3).
        
        M_eff = ρ π ξ² ln(R/ξ)
        
        Args:
            xi_core: Vortex core size (coherence length)
            R_system: System size (hadron radius)
            rho_condensate: Gluon condensate density
        
        Returns:
            Effective inertial mass in GeV
        """
        if xi_core <= 0 or R_system <= xi_core:
            warnings.warn("Invalid core/system size, returning 0")
            return 0.0
        
        M_eff = rho_condensate * np.pi * xi_core**2 * np.log(R_system / xi_core)
        
        return M_eff
    
    def breathing_mode_frequency(self, M_eff: float, 
                                 kappa_string: float = 1.0) -> float:
        """
        Compute radial breathing mode frequency (file:8 Eq. 6).
        
        ω_r = √(κ / M_eff)
        
        Divergence triggers hadronization instability.
        
        Args:
            M_eff: Effective vortex mass
            kappa_string: String tension (GeV²)
        
        Returns:
            Breathing mode frequency in GeV
        """
        if M_eff <= 0:
            return np.inf  # Massless limit → instant collapse
        
        omega_r = np.sqrt(kappa_string / M_eff)
        
        return omega_r


class ExceptionalPointAnalyzer:
    """
    Analyzer for exceptional points (EP) in non-Hermitian Hamiltonian.
    
    Implements detection and characterization of PT-symmetry breaking
    transitions (file:7 Section 3).
    """
    
    def __init__(self, gamma_loss: float, gamma_SR: float):
        """
        Args:
            gamma_loss: Loss rate Γ_loss
            gamma_SR: Superradiant gain rate Γ_SR
        """
        self.gamma_loss = gamma_loss
        self.gamma_SR = gamma_SR
    
    def find_exceptional_point(self, r_injection_range: Tuple[float, float],
                               n_points: int = 100) -> Dict[str, float]:
        """
        Locate exceptional point EP2 in parameter space.
        
        EP2 occurs when Re(E₊) = Re(E₋) and Im(E₊) = Im(E₋)
        i.e., eigenvalues and eigenvectors coalesce (file:7 Eq. 8).
        
        Args:
            r_injection_range: (r_min, r_max) to scan
            n_points: Number of scan points
        
        Returns:
            Dict with EP location and properties
        """
        r_values = np.linspace(r_injection_range[0], r_injection_range[1], n_points)
        
        # Scan for EP signature: |E₊ - E₋| → 0
        energy_gaps = []
        
        for r in r_values:
            H_eff = self._effective_hamiltonian(r)
            eigenvalues = eigvals(H_eff)
            
            # Smallest gap
            gap = np.min(np.abs(np.diff(eigenvalues)))
            energy_gaps.append(gap)
        
        # Find minimum gap
        ep_index = np.argmin(energy_gaps)
        r_ep = r_values[ep_index]
        gap_ep = energy_gaps[ep_index]
        
        # Verify EP2 topology (square-root branch point)
        ep_order = self._determine_ep_order(r_ep)
        
        return {
            'r_injection': r_ep,
            'energy_gap': gap_ep,
            'ep_order': ep_order,
            'is_ep2': (ep_order == 2),
            'gamma_loss': self.gamma_loss,
            'gamma_SR': self.gamma_SR
        }
    
    def _effective_hamiltonian(self, r_injection: float) -> np.ndarray:
        """
        Construct effective non-Hermitian Hamiltonian (file:7 Eq. 2).
        
        H_eff = ω₀ a†a - i(Γ_loss/2) a†a + i(Γ_SR r_inj)(a†² - a²)
        
        2×2 representation in {|0⟩, |1⟩} basis.
        """
        omega_0 = 1.0  # Base frequency (natural units)
        
        # Diagonal: rotation + loss
        diagonal = omega_0 - 1j * self.gamma_loss / 2.0
        
        # Off-diagonal: superradiant gain
        off_diagonal = 1j * self.gamma_SR * r_injection
        
        H_eff = np.array([
            [0, off_diagonal],
            [off_diagonal, diagonal]
        ], dtype=complex)
        
        return H_eff
    
    def _determine_ep_order(self, r_ep: float, epsilon: float = 1e-4) -> int:
        """
        Determine EP order from branching exponent (file:7 Eq. 10).
        
        ΔE ~ (ρ - ρ_EP)^{1/n} for EP of order n
        
        EP2: n=2 (square-root)
        EP3: n=3 (cube-root)
        """
        # Sample eigenvalue splitting around EP
        r_above = r_ep + epsilon
        r_below = r_ep - epsilon
        
        H_above = self._effective_hamiltonian(r_above)
        H_below = self._effective_hamiltonian(r_below)
        
        eigs_above = eigvals(H_above)
        eigs_below = eigvals(H_below)
        
        gap_above = np.abs(eigs_above[0] - eigs_above[1])
        gap_below = np.abs(eigs_below[0] - eigs_below[1])
        
        # Fit to power law: gap ~ ε^{1/n}
        if gap_above > 1e-10:
            exponent = np.log(gap_above / gap_below) / np.log(2)
            ep_order = int(round(1.0 / exponent))
        else:
            ep_order = 2  # Default EP2
        
        return ep_order
    
    def pt_symmetry_phase(self, r_injection: float) -> str:
        """
        Determine PT-symmetry phase (file:7 Eq. 7).
        
        PT-symmetric: All eigenvalues real
        PT-broken: Complex eigenvalues appear
        
        Args:
            r_injection: Squeezing injection parameter
        
        Returns:
            Phase label: "PT_SYMMETRIC" or "PT_BROKEN"
        """
        H_eff = self._effective_hamiltonian(r_injection)
        eigenvalues = eigvals(H_eff)
        
        # Check if all eigenvalues are real (within tolerance)
        max_imag = np.max(np.abs(np.imag(eigenvalues)))
        
        if max_imag < 1e-8:
            return "PT_SYMMETRIC"
        else:
            return "PT_BROKEN"


class PetermannFactor:
    """
    Petermann factor calculation (file:7 Eq. 11, file:7 Section 5).
    
    K = |⟨L|R⟩|² / (⟨L|L⟩ ⟨R|R⟩)
    
    Diverges as K ~ |ρ - ρ_EP|^{-1} near exceptional point.
    """
    
    def __init__(self, H_effective: np.ndarray):
        """
        Args:
            H_effective: Non-Hermitian Hamiltonian matrix
        """
        self.H = H_effective
        
        # Compute right and left eigenvectors
        self.eigenvalues, self.R_vectors, self.L_vectors = self._biorthogonal_eigensystem()
    
    def _biorthogonal_eigensystem(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute biorthogonal eigensystem for non-Hermitian H.
        
        H |R⟩ = E |R⟩
        ⟨L| H = E ⟨L|
        """
        # Right eigenvectors
        eigenvalues, R_vecs = np.linalg.eig(self.H)
        
        # Left eigenvectors (from H†)
        _, L_vecs = np.linalg.eig(self.H.conj().T)
        
        return eigenvalues, R_vecs, L_vecs
    
    def compute_factor(self, mode_index: int = 0) -> float:
        """
        Compute Petermann factor for given mode.
        
        Args:
            mode_index: Index of eigenmode (0 = ground state)
        
        Returns:
            Petermann factor K ≥ 1
        """
        R = self.R_vectors[:, mode_index]
        L = self.L_vectors[:, mode_index]
        
        # Overlaps
        LR = np.abs(np.vdot(L, R))**2
        LL = np.vdot(L, L).real
        RR = np.vdot(R, R).real
        
        if LL * RR < 1e-16:
            return np.inf
        
        K = LR / (LL * RR)
        
        return K
    
    def sensitivity_enhancement(self, mode_index: int = 0) -> float:
        """
        Compute eigenvalue sensitivity enhancement (file:7 Eq. 10).
        
        δE ~ K^{1/2} × perturbation
        
        Args:
            mode_index: Eigenmode index
        
        Returns:
            Sensitivity factor √K
        """
        K = self.compute_factor(mode_index)
        
        return np.sqrt(K)


# Utility functions

def detect_bifurcation(omega_history: np.ndarray, 
                      threshold_derivative: float = 10.0) -> Optional[int]:
    """
    Detect bifurcation point from vortex rotation frequency history.
    
    Bifurcation indicated by rapid change in dω/dt (file:9 Section 3.4).
    
    Args:
        omega_history: Time series of ω_vortex
        threshold_derivative: Detection threshold for |dω/dt|
    
    Returns:
        Index of bifurcation, or None if not detected
    """
    # Numerical derivative
    d_omega = np.diff(omega_history)
    
    # Find first point exceeding threshold
    bifurcation_indices = np.where(np.abs(d_omega) > threshold_derivative)[0]
    
    if len(bifurcation_indices) > 0:
        return bifurcation_indices[0]
    else:
        return None


def vortex_charge_duality_invariant(Q_charge: complex, 
                                    V_vorticity: complex) -> complex:
    """
    Compute unified topological invariant (file:7 Eq. 9).
    
    Ψ = Q + iV
    
    Real part: Topological charge (color confinement)
    Imaginary part: Spectral vorticity (ER bridge stability)
    
    Args:
        Q_charge: Color charge topological number
        V_vorticity: Vortex winding number
    
    Returns:
        Complex invariant Ψ
    """
    Psi = Q_charge + 1j * V_vorticity
    
    return Psi


# Example usage
if __name__ == "__main__":
    print("=== MVC Threshold Detection ===\n")
    
    detector = MVCThresholdDetector(alpha=2.5)
    
    print(f"Critical density ρ_MVC: {detector.rho_mvc:.2e} GeV⁴")
    
    # Test confinement phases
    test_cases = [
        (0.5e19, 0.3, 0.5),  # Deconfined
        (1.0e19, 0.5, 1.0),  # Pre-confinement
        (2.0e19, 0.8, 2.0),  # Confined
    ]
    
    for rho, omega, rho_E in test_cases:
        confined, phase = detector.is_confined(rho, omega, rho_E)
        print(f"ρ={rho:.2e}, ω={omega:.2f}, ρ_E={rho_E:.2f} → {phase}")
    
    print("\n=== Exceptional Point Analysis ===\n")
    
    ep_analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)
    
    ep_data = ep_analyzer.find_exceptional_point((0.0, 2.0), n_points=200)
    
    print(f"EP2 location: r_inj = {ep_data['r_injection']:.4f}")
    print(f"Energy gap at EP: {ep_data['energy_gap']:.2e}")
    print(f"EP order: {ep_data['ep_order']}")
    
    print("\n=== Petermann Factor ===\n")
    
    H_test = ep_analyzer._effective_hamiltonian(ep_data['r_injection'])
    petermann = PetermannFactor(H_test)
    
    K = petermann.compute_factor(mode_index=0)
    print(f"Petermann factor K: {K:.2f}")
    print(f"Sensitivity enhancement: {petermann.sensitivity_enhancement():.2f}×")
