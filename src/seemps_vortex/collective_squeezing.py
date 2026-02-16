"""
Collective Squeezing Operators
===============================

Implements two-mode squeezed states (TMST) and collective squeezing
projection mechanisms for QCD vortex networks.

Mathematical framework from:
- file:9 Eq. (1-3): Collective TMST projection
- file:6 Eq. (4.1-4.3): Two-mode squeezing operators
- file:7 Eq. (2): Superradiant amplification term
"""

import numpy as np
from scipy.linalg import expm, sqrtm
from typing import Tuple, Optional, List
import warnings


class CollectiveSqueezing:
    """
    Collective two-mode squeezing operator for vortex networks.
    
    Implements S_coll(r) = exp[r(A_R† A_L† - A_R A_L)]
    where A_R, A_L are collective right/left circulation modes.
    
    Attributes:
        r_squeeze: Squeezing parameter (dimensionless)
        N_modes: Number of local modes participating
        coupling_coefficients: Overlap factors c_i for each mode
    """
    
    def __init__(self, r_squeeze: float, N_modes: int, 
                 coupling_coefficients: Optional[np.ndarray] = None):
        """
        Initialize collective squeezing operator.
        
        Args:
            r_squeeze: Squeezing strength (0 < r < ∞)
            N_modes: Number of participating modes
            coupling_coefficients: Optional c_i weights (default: uniform)
        """
        if r_squeeze < 0:
            raise ValueError("Squeezing parameter must be non-negative")
        
        self.r = r_squeeze
        self.N_modes = N_modes
        
        # Uniform coupling by default (file:9 Eq. 1)
        if coupling_coefficients is None:
            self.c_i = np.ones(N_modes) / np.sqrt(N_modes)
        else:
            # Normalize
            self.c_i = coupling_coefficients / np.linalg.norm(coupling_coefficients)
    
    def generate_covariance_matrix(self, n_thermal: float = 0.0) -> np.ndarray:
        """
        Generate covariance matrix for TMST state (file:6 Appendix A.2).
        
        For symmetric TMST:
        σ = (n + 1/2) [ cosh(2r) I₂    sinh(2r) Z
                        sinh(2r) Z    cosh(2r) I₂ ]
        
        where Z = diag(1, -1) and n = n_thermal
        
        Args:
            n_thermal: Thermal occupation number n(T)
        
        Returns:
            4×4 covariance matrix for two-mode system
        """
        n_plus_half = n_thermal + 0.5
        
        # Block matrices
        I2 = np.eye(2)
        Z = np.diag([1, -1])
        
        cosh_2r = np.cosh(2 * self.r)
        sinh_2r = np.sinh(2 * self.r)
        
        # Upper blocks
        sigma_AA = n_plus_half * cosh_2r * I2
        sigma_AB = n_plus_half * sinh_2r * Z
        
        # Full covariance matrix (Gaussian state)
        sigma = np.block([
            [sigma_AA, sigma_AB],
            [sigma_AB, sigma_AA]
        ])
        
        return sigma
    
    def squeezing_transformation(self, quadratures: np.ndarray) -> np.ndarray:
        """
        Apply squeezing transformation to phase-space quadratures.
        
        Symplectic transformation:
        R' = S(r) R where S(r) = [ e^r   0  ]
                                   [  0  e^-r]
        
        Args:
            quadratures: Array [x_A, p_A, x_B, p_B]
        
        Returns:
            Transformed quadratures
        """
        # Squeezing matrix (symplectic form)
        S_matrix = np.diag([
            np.exp(self.r), np.exp(-self.r),
            np.exp(self.r), np.exp(-self.r)
        ])
        
        return S_matrix @ quadratures
    
    def compute_entanglement_entropy(self, n_thermal: float) -> float:
        """
        Compute entanglement entropy of TMST (file:9 Eq. 3).
        
        S_E = (n + 1/2) log[(n + 1/2)] - (n - 1/2) log|n - 1/2|
        
        where n = (n_thermal + 1/2) cosh(2r)
        
        Args:
            n_thermal: Thermal occupation
        
        Returns:
            Entanglement entropy S_E in bits
        """
        n_eff = (n_thermal + 0.5) * np.cosh(2 * self.r)
        
        if n_eff <= 0.5:
            return 0.0
        
        # von Neumann entropy for Gaussian states
        S_plus = (n_eff + 0.5) * np.log2(n_eff + 0.5)
        S_minus = (n_eff - 0.5) * np.log2(abs(n_eff - 0.5))
        
        return S_plus - S_minus


class TMSTState:
    """
    Two-Mode Squeezed Thermal State representation.
    
    Implements |ψ_TMST⟩ = S_2(r)|thermal⟩ (file:6 Eq. 4.3)
    """
    
    def __init__(self, r_squeeze: float, T_temperature: float, omega: float = 1.0):
        """
        Args:
            r_squeeze: Squeezing parameter
            T_temperature: Temperature in natural units
            omega: Mode frequency (default 1.0)
        """
        self.r = r_squeeze
        self.T = T_temperature
        self.omega = omega
        
        # Compute thermal occupation (file:6 Eq. 2.3)
        self.n_thermal = self._bose_einstein(T_temperature, omega)
        
        # Squeezing operator
        self.squeezing = CollectiveSqueezing(r_squeeze, N_modes=2)
    
    def _bose_einstein(self, T: float, omega: float) -> float:
        """Bose-Einstein occupation number."""
        if T <= 0:
            return 0.0
        
        x = omega / T
        if x > 100:  # Avoid overflow
            return 0.0
        
        return 1.0 / (np.exp(x) - 1.0)
    
    def is_entangled(self) -> Tuple[bool, float]:
        """
        Check entanglement via PPT criterion (file:6 Theorem 4.3.1).
        
        Entangled iff: n(T) < 1/2 (e^{2r} - 1)^{-1}
        
        Returns:
            (is_entangled, critical_temperature)
        """
        # Critical thermal occupation (file:6 Eq. 4.4)
        n_critical = 0.5 / (np.exp(2 * self.r) - 1.0)
        
        # Check criterion
        entangled = self.n_thermal < n_critical
        
        # Compute critical temperature (file:6 Eq. 4.7)
        if n_critical > 0:
            T_critical = self.omega / np.log(1.0 + 1.0/n_critical)
        else:
            T_critical = 0.0
        
        return entangled, T_critical
    
    def compute_log_negativity(self) -> float:
        """
        Compute log-negativity E_N (file:6 Eq. A.4).
        
        E_N = max(0, -log₂(2ν₋))
        where ν₋ = (n + 1/2) exp(-2r)
        """
        nu_minus = (self.n_thermal + 0.5) * np.exp(-2 * self.r)
        
        EN = max(0.0, -np.log2(2 * nu_minus))
        
        return EN
    
    def covariance_matrix(self) -> np.ndarray:
        """Return full covariance matrix."""
        return self.squeezing.generate_covariance_matrix(self.n_thermal)


class SuperradiantOperator:
    """
    Superradiant amplification operator (file:7 Eq. 2, file:9 Eq. 4).
    
    Implements gain term: Γ_SR r_inj a_c†² a² - Γ_loss a_c† a_c
    """
    
    def __init__(self, gamma_SR: float, gamma_loss: float, r_injection: float):
        """
        Args:
            gamma_SR: Superradiant gain rate
            gamma_loss: Radiative loss rate
            r_injection: Injection squeezing parameter
        """
        self.gamma_SR = gamma_SR
        self.gamma_loss = gamma_loss
        self.r_inj = r_injection
    
    def net_gain(self, omega_rot: float, m_mode: int) -> float:
        """
        Compute net superradiant gain (file:9 Eq. 4).
        
        Superradiance condition: ω_mode < m Ω_rot
        
        Args:
            omega_rot: Vortex rotation frequency
            m_mode: Azimuthal mode number
        
        Returns:
            Net amplification rate
        """
        # Superradiant threshold (file:7)
        omega_effective = m_mode * omega_rot
        
        if omega_effective > 0:
            gain = self.gamma_SR * self.r_inj - self.gamma_loss / 2.0
        else:
            gain = -self.gamma_loss / 2.0
        
        return gain
    
    def amplification_coefficient(self, k_wave: float, delta_x: float) -> float:
        """
        Spatial amplification factor (file:9 Eq. 5).
        
        A(Δx) = exp[Γ_SR Δx / c] for matched phase k·Δx = 0
        
        Args:
            k_wave: Wave vector
            delta_x: Spatial separation
        
        Returns:
            Amplification coefficient
        """
        # Phase matching condition
        phase_factor = np.cos(k_wave * delta_x)
        
        return np.exp(self.gamma_SR * delta_x) * phase_factor


# Utility functions

def critical_squeezing(n_thermal: float) -> float:
    """
    Compute critical squeezing parameter r_c(T) (file:6 Eq. 4.5).
    
    r_c = (1/2) ln(2n + 1)
    
    Args:
        n_thermal: Thermal occupation number
    
    Returns:
        Critical squeezing r_c
    """
    if n_thermal < 0:
        warnings.warn("Negative thermal occupation, returning 0")
        return 0.0
    
    return 0.5 * np.log(2 * n_thermal + 1)


def squeezing_spectrum(n_thermal_array: np.ndarray, 
                       r_squeeze: float) -> np.ndarray:
    """
    Generate squeezing spectrum S(ω) (file:7 Eq. 5).
    
    S(ω) = Γ_loss / [(Γ_loss/2)² + (Γ_SR r - ω)²]
    
    Args:
        n_thermal_array: Array of thermal occupations
        r_squeeze: Squeezing parameter
    
    Returns:
        Spectrum array
    """
    # Simplified version (full requires solving Langevin equations)
    spectrum = 1.0 / (1.0 + n_thermal_array * np.exp(-2 * r_squeeze))
    
    return spectrum


# Example usage
if __name__ == "__main__":
    print("=== Collective Squeezing Test ===\n")
    
    # Test TMST state at different temperatures
    r_test = 1.2  # Strong squeezing (central PbPb regime)
    
    temperatures = [0.2, 0.8, 1.5]  # pp, peripheral, central
    
    for T in temperatures:
        tmst = TMSTState(r_squeeze=r_test, T_temperature=T, omega=1.0)
        
        is_ent, T_crit = tmst.is_entangled()
        EN = tmst.compute_log_negativity()
        
        print(f"T = {T:.2f}")
        print(f"  n(T) = {tmst.n_thermal:.4f}")
        print(f"  Entangled: {is_ent}")
        print(f"  T_critical: {T_crit:.4f}")
        print(f"  Log-negativity: {EN:.4f}")
        print()
    
    # Test superradiance
    print("=== Superradiant Amplification ===\n")
    
    sr_op = SuperradiantOperator(gamma_SR=0.5, gamma_loss=0.1, r_injection=1.0)
    
    omega_rot = 0.8
    for m in [1, 2, 3]:
        gain = sr_op.net_gain(omega_rot, m)
        print(f"Mode m={m}: Net gain = {gain:.4f}")
