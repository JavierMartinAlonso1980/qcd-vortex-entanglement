"""
Two-Particle Correlation Analysis for Belle II
===============================================

Extraction and analysis of two-particle correlations in τ⁺τ⁻ events:
- Spin-spin correlations
- Angular correlations (Δφ, Δη)
- Bell inequality tests (CHSH)
- Entanglement witness measurements

Theoretical framework:
- file:2: Belle II Tau Pair Entanglement Toy MC
- file:3: Fermionic Bulk-Boundary Algorithm
- file:9: Collective vortex projection signatures
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Dict, List, Optional, Callable
import warnings


class TwoParticleCorrelator:
    """
    Compute two-particle correlation functions from Belle II data.
    
    Standard observable:
    C(Δφ, Δη) = ⟨N_pairs(Δφ, Δη)⟩ / ⟨N_pairs⟩_mixed
    
    where mixed-event normalization removes detector effects.
    """
    
    def __init__(self, n_phi_bins: int = 36, n_eta_bins: int = 20):
        """
        Args:
            n_phi_bins: Number of Δφ bins (default 36 → 10° bins)
            n_eta_bins: Number of Δη bins
        """
        self.n_phi_bins = n_phi_bins
        self.n_eta_bins = n_eta_bins
        
        # Bin edges
        self.phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)
        self.eta_edges = np.linspace(-5, 5, n_eta_bins + 1)
        
        # Bin centers
        self.phi_centers = 0.5 * (self.phi_edges[1:] + self.phi_edges[:-1])
        self.eta_centers = 0.5 * (self.eta_edges[1:] + self.eta_edges[:-1])
    
    def compute_correlation(self, phi1: np.ndarray, eta1: np.ndarray,
                           phi2: np.ndarray, eta2: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D correlation function C(Δφ, Δη).
        
        Args:
            phi1, eta1: Azimuthal/pseudorapidity of particle 1 (τ+)
            phi2, eta2: Azimuthal/pseudorapidity of particle 2 (τ-)
            weights: Optional event weights
        
        Returns:
            (correlation_matrix, phi_centers, eta_centers)
        """
        # Compute differences
        delta_phi = self._delta_phi_wrapped(phi1, phi2)
        delta_eta = eta1 - eta2
        
        # 2D histogram
        if weights is None:
            weights = np.ones(len(delta_phi))
        
        H, _, _ = np.histogram2d(
            delta_phi, delta_eta,
            bins=[self.phi_edges, self.eta_edges],
            weights=weights
        )
        
        # Normalize
        H = H / (np.sum(H) + 1e-10)
        
        return H, self.phi_centers, self.eta_centers
    
    def compute_1d_correlation(self, phi1: np.ndarray, phi2: np.ndarray,
                              weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 1D correlation C(Δφ).
        
        Returns:
            (correlation_values, phi_centers)
        """
        delta_phi = self._delta_phi_wrapped(phi1, phi2)
        
        if weights is None:
            weights = np.ones(len(delta_phi))
        
        H, _ = np.histogram(delta_phi, bins=self.phi_edges, weights=weights)
        
        # Normalize
        H = H / (np.sum(H) + 1e-10)
        
        return H, self.phi_centers
    
    @staticmethod
    def _delta_phi_wrapped(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
        """Compute Δφ with periodic boundary conditions."""
        dphi = phi1 - phi2
        dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
        dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
        return dphi
    
    def mixed_event_correlation(self, phi1_list: List[np.ndarray],
                               eta1_list: List[np.ndarray],
                               phi2_list: List[np.ndarray],
                               eta2_list: List[np.ndarray],
                               n_mix: int = 10) -> np.ndarray:
        """
        Compute mixed-event correlation for normalization.
        
        Pairs particles from different events to remove correlations.
        
        Args:
            phi1_list: List of φ arrays from different events
            eta1_list: List of η arrays
            phi2_list, eta2_list: Same for particle 2
            n_mix: Number of mixing iterations
        
        Returns:
            Mixed-event correlation matrix
        """
        N_events = len(phi1_list)
        
        mixed_sum = np.zeros((self.n_phi_bins, self.n_eta_bins))
        
        for _ in range(n_mix):
            # Random event pairing
            idx1 = np.random.randint(0, N_events)
            idx2 = np.random.randint(0, N_events)
            
            if idx1 == idx2:
                continue
            
            # Compute correlation
            H, _, _ = self.compute_correlation(
                phi1_list[idx1], eta1_list[idx1],
                phi2_list[idx2], eta2_list[idx2]
            )
            
            mixed_sum += H
        
        mixed_avg = mixed_sum / n_mix
        
        return mixed_avg
    
    def corrected_correlation(self, same_event_corr: np.ndarray,
                             mixed_event_corr: np.ndarray) -> np.ndarray:
        """
        Apply mixed-event correction.
        
        C_corrected = C_same / C_mixed
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            C_corrected = same_event_corr / (mixed_event_corr + 1e-10)
        
        C_corrected[~np.isfinite(C_corrected)] = 1.0
        
        return C_corrected


class SpinCorrelationAnalyzer:
    """
    Analyze spin-spin correlations in τ⁺τ⁻ pairs.
    
    Key observable:
    C_spin(θ₁, θ₂) = ⟨σ₁·n̂₁⟩ ⟨σ₂·n̂₂⟩
    
    where n̂ᵢ are measurement directions.
    """
    
    def __init__(self):
        pass
    
    def spin_spin_correlation(self, helicity1: np.ndarray, helicity2: np.ndarray,
                             theta1: np.ndarray, theta2: np.ndarray) -> Dict[str, float]:
        """
        Compute spin-spin correlation coefficient.
        
        C = ⟨h₁ h₂⟩ / √(⟨h₁²⟩ ⟨h₂²⟩)
        
        Args:
            helicity1, helicity2: Helicity values (±1)
            theta1, theta2: Polar angles (for weighting)
        
        Returns:
            Dictionary with correlation statistics
        """
        # Correlation
        h1h2 = helicity1 * helicity2
        
        C_mean = np.mean(h1h2)
        C_std = np.std(h1h2) / np.sqrt(len(h1h2))
        
        # Angular-weighted correlation
        weight = np.abs(np.cos(theta1) * np.cos(theta2))
        C_weighted = np.average(h1h2, weights=weight)
        
        return {
            'C_mean': C_mean,
            'C_std': C_std,
            'C_weighted': C_weighted,
            'significance': abs(C_mean) / (C_std + 1e-10)
        }
    
    def angular_correlation_function(self, helicity1: np.ndarray, helicity2: np.ndarray,
                                    delta_phi: np.ndarray,
                                    n_bins: int = 18) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute C_spin(Δφ) = ⟨h₁ h₂⟩(Δφ).
        
        Returns:
            (phi_centers, C_values, C_errors)
        """
        h1h2 = helicity1 * helicity2
        
        # Bin in Δφ
        phi_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        phi_centers = 0.5 * (phi_edges[1:] + phi_edges[:-1])
        
        C_values = np.zeros(n_bins)
        C_errors = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (delta_phi >= phi_edges[i]) & (delta_phi < phi_edges[i+1])
            
            if np.sum(mask) > 0:
                C_values[i] = np.mean(h1h2[mask])
                C_errors[i] = np.std(h1h2[mask]) / np.sqrt(np.sum(mask))
        
        return phi_centers, C_values, C_errors
    
    def fit_correlation_model(self, phi: np.ndarray, C: np.ndarray,
                             C_err: np.ndarray) -> Dict[str, float]:
        """
        Fit correlation to theoretical model (file:2).
        
        Model: C(Δφ) = A₀ + A₁ cos(Δφ) + A₂ cos(2Δφ)
        
        Returns:
            Fit parameters and χ²
        """
        def model(phi, A0, A1, A2):
            return A0 + A1 * np.cos(phi) + A2 * np.cos(2 * phi)
        
        try:
            popt, pcov = curve_fit(model, phi, C, sigma=C_err, p0=[0, 0.1, 0.1])
            
            perr = np.sqrt(np.diag(pcov))
            
            # Compute χ²
            C_fit = model(phi, *popt)
            chi2 = np.sum(((C - C_fit) / C_err)**2)
            ndof = len(phi) - len(popt)
            chi2_reduced = chi2 / ndof
            
            return {
                'A0': popt[0],
                'A1': popt[1],
                'A2': popt[2],
                'A0_err': perr[0],
                'A1_err': perr[1],
                'A2_err': perr[2],
                'chi2': chi2,
                'chi2_reduced': chi2_reduced,
                'p_value': 1 - stats.chi2.cdf(chi2, ndof)
            }
        except Exception as e:
            warnings.warn(f"Fit failed: {e}")
            return {}


class BellInequalityTester:
    """
    Test Bell inequalities (CHSH) for τ⁺τ⁻ entanglement.
    
    CHSH inequality: |S| ≤ 2 (local hidden variables)
    Quantum mechanics: |S| ≤ 2√2 ≈ 2.828
    
    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    
    where E(a,b) = ⟨σ_A(a) σ_B(b)⟩
    """
    
    def __init__(self):
        self.measurement_angles = {
            'a': 0,
            'a_prime': np.pi / 2,
            'b': np.pi / 4,
            'b_prime': -np.pi / 4
        }
    
    def compute_chsh_parameter(self, helicity_plus: np.ndarray, helicity_minus: np.ndarray,
                               phi_plus: np.ndarray, phi_minus: np.ndarray) -> Dict[str, float]:
        """
        Compute CHSH parameter S from experimental data.
        
        Args:
            helicity_plus, helicity_minus: Measured helicities
            phi_plus, phi_minus: Azimuthal angles (proxy for measurement axes)
        
        Returns:
            Dictionary with S, error, and violation significance
        """
        # Map angles to measurement settings
        def E_correlation(angle_A, angle_B):
            """Expectation value for measurement angles."""
            # Project helicities onto measurement axes
            proj_A = helicity_plus * np.cos(phi_plus - angle_A)
            proj_B = helicity_minus * np.cos(phi_minus - angle_B)
            
            return np.mean(proj_A * proj_B)
        
        # Compute four correlation functions
        E_ab = E_correlation(self.measurement_angles['a'], self.measurement_angles['b'])
        E_ab_prime = E_correlation(self.measurement_angles['a'], self.measurement_angles['b_prime'])
        E_a_prime_b = E_correlation(self.measurement_angles['a_prime'], self.measurement_angles['b'])
        E_a_prime_b_prime = E_correlation(self.measurement_angles['a_prime'], self.measurement_angles['b_prime'])
        
        # CHSH parameter
        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime
        
        # Error estimation (bootstrap)
        N = len(helicity_plus)
        S_bootstrap = []
        
        for _ in range(1000):
            idx = np.random.choice(N, size=N, replace=True)
            
            h_plus_boot = helicity_plus[idx]
            h_minus_boot = helicity_minus[idx]
            phi_plus_boot = phi_plus[idx]
            phi_minus_boot = phi_minus[idx]
            
            def E_boot(angle_A, angle_B):
                proj_A = h_plus_boot * np.cos(phi_plus_boot - angle_A)
                proj_B = h_minus_boot * np.cos(phi_minus_boot - angle_B)
                return np.mean(proj_A * proj_B)
            
            E_ab_b = E_boot(self.measurement_angles['a'], self.measurement_angles['b'])
            E_ab_prime_b = E_boot(self.measurement_angles['a'], self.measurement_angles['b_prime'])
            E_a_prime_b_b = E_boot(self.measurement_angles['a_prime'], self.measurement_angles['b'])
            E_a_prime_b_prime_b = E_boot(self.measurement_angles['a_prime'], self.measurement_angles['b_prime'])
            
            S_boot = E_ab_b - E_ab_prime_b + E_a_prime_b_b + E_a_prime_b_prime_b
            S_bootstrap.append(S_boot)
        
        S_err = np.std(S_bootstrap)
        
        # Violation significance
        classical_bound = 2.0
        quantum_bound = 2 * np.sqrt(2)
        
        violation_sigma = (abs(S) - classical_bound) / S_err
        
        return {
            'S': S,
            'S_err': S_err,
            'E_ab': E_ab,
            'E_ab_prime': E_ab_prime,
            'E_a_prime_b': E_a_prime_b,
            'E_a_prime_b_prime': E_a_prime_b_prime,
            'classical_bound': classical_bound,
            'quantum_bound': quantum_bound,
            'violates_classical': abs(S) > classical_bound,
            'violation_sigma': violation_sigma,
            'p_value': 1 - stats.norm.cdf(violation_sigma)
        }
    
    def test_clauser_horne_inequality(self, P_ab: float, P_ab_prime: float,
                                      P_a_prime_b: float, P_a_not_b_not: float) -> Dict[str, float]:
        """
        Test Clauser-Horne (CH) inequality (alternative to CHSH).
        
        CH: P(a,b) - P(a,b') - P(a',b) + P(¬a,¬b) ≤ 0
        
        Args:
            P_**: Joint detection probabilities
        
        Returns:
            CH parameter and violation status
        """
        CH = P_ab - P_ab_prime - P_a_prime_b + P_a_not_b_not
        
        return {
            'CH_parameter': CH,
            'CH_bound': 0.0,
            'violates_CH': CH > 0.0
        }


# Utility functions

def bootstrap_error(data: np.ndarray, statistic: Callable, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Compute bootstrap error for any statistic.
    
    Args:
        data: Input data array
        statistic: Function to compute (e.g., np.mean, np.std)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        (mean_value, std_error)
    """
    N = len(data)
    
    bootstrap_values = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(N, size=N, replace=True)
        bootstrap_values.append(statistic(data[idx]))
    
    return np.mean(bootstrap_values), np.std(bootstrap_values)


# Example usage
if __name__ == "__main__":
    print("=== Belle II Correlation Analysis Test ===\n")
    
    # Generate synthetic τ⁺τ⁻ events
    np.random.seed(42)
    N_events = 5000
    
    # Entangled state: correlated helicities
    helicity_plus = np.random.choice([-1, 1], size=N_events, p=[0.3, 0.7])
    helicity_minus = -helicity_plus + np.random.choice([-1, 0, 1], size=N_events, p=[0.1, 0.8, 0.1])
    
    phi_plus = np.random.uniform(-np.pi, np.pi, N_events)
    phi_minus = phi_plus + np.random.normal(0, 0.5, N_events)  # Correlated angles
    
    eta_plus = np.random.normal(0, 1.5, N_events)
    eta_minus = np.random.normal(0, 1.5, N_events)
    
    # 1. Two-particle correlation
    print("1. Two-Particle Correlation Function")
    correlator = TwoParticleCorrelator(n_phi_bins=36, n_eta_bins=20)
    
    C_2D, phi_c, eta_c = correlator.compute_correlation(phi_plus, eta_plus, phi_minus, eta_minus)
    
    print(f"   2D correlation matrix shape: {C_2D.shape}")
    print(f"   Peak correlation: {np.max(C_2D):.4f}")
    
    # 2. Spin correlation
    print("\n2. Spin-Spin Correlation")
    spin_analyzer = SpinCorrelationAnalyzer()
    
    theta_plus = np.arccos(np.random.uniform(-1, 1, N_events))
    theta_minus = np.arccos(np.random.uniform(-1, 1, N_events))
    
    spin_corr = spin_analyzer.spin_spin_correlation(helicity_plus, helicity_minus, theta_plus, theta_minus)
    
    print(f"   ⟨h₁ h₂⟩ = {spin_corr['C_mean']:.4f} ± {spin_corr['C_std']:.4f}")
    print(f"   Weighted: {spin_corr['C_weighted']:.4f}")
    print(f"   Significance: {spin_corr['significance']:.2f} σ")
    
    # 3. Bell inequality test
    print("\n3. CHSH Bell Inequality Test")
    bell_tester = BellInequalityTester()
    
    chsh_result = bell_tester.compute_chsh_parameter(helicity_plus, helicity_minus, phi_plus, phi_minus)
    
    print(f"   S parameter: {chsh_result['S']:.4f} ± {chsh_result['S_err']:.4f}")
    print(f"   Classical bound: {chsh_result['classical_bound']:.1f}")
    print(f"   Quantum bound: {chsh_result['quantum_bound']:.3f}")
    print(f"   Violates classical: {chsh_result['violates_classical']}")
    print(f"   Violation significance: {chsh_result['violation_sigma']:.2f} σ")
    print(f"   p-value: {chsh_result['p_value']:.2e}")
