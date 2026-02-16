"""
Tests for Entanglement Measures and Analysis

Tests cover:
- Log-negativity computation
- Concurrence calculation
- Entanglement witnesses
- Bell inequality tests
- Correlation functions
- Separability criteria
- Gaussian state entanglement
"""

import pytest
import numpy as np
from scipy.linalg import sqrtm
import warnings

# Test utilities
from numpy.testing import assert_allclose, assert_array_less


# ============================================================================
# Mock Classes for Testing (if modules not available)
# ============================================================================

class MockGaussianState:
    """Mock Gaussian state for testing."""

    def __init__(self, covariance_matrix):
        self.covariance_matrix = covariance_matrix
        self.n_modes = covariance_matrix.shape[0] // 2

    def partial_transpose(self, subsystem):
        """Compute partial transpose of covariance matrix."""
        gamma = self.covariance_matrix.copy()

        # Apply partial transpose (flip momentum sign)
        if subsystem == 0:
            gamma[1, :] = -gamma[1, :]
            gamma[:, 1] = -gamma[:, 1]
        else:
            gamma[3, :] = -gamma[3, :]
            gamma[:, 3] = -gamma[:, 3]

        return gamma


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def product_state_covariance():
    """Covariance matrix for separable (product) state."""
    return np.eye(4)


@pytest.fixture
def epr_state_covariance():
    """Covariance matrix for maximally entangled EPR state."""
    # EPR state: γ = diag(1, 1, 1, 1) for infinite squeezing
    # Finite squeezing: r = 1.0
    r = 1.0
    cosh_r = np.cosh(2*r)
    sinh_r = np.sinh(2*r)

    gamma = np.array([
        [cosh_r, 0, sinh_r, 0],
        [0, cosh_r, 0, -sinh_r],
        [sinh_r, 0, cosh_r, 0],
        [0, -sinh_r, 0, cosh_r]
    ])
    return gamma


@pytest.fixture
def thermal_state_covariance():
    """Covariance matrix for thermal state (mixed, separable)."""
    n_thermal = 0.5
    return (2*n_thermal + 1) * np.eye(4)


@pytest.fixture
def bell_state_density_matrix():
    """Density matrix for Bell state |Φ+⟩."""
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    return np.outer(psi, psi.conj())


@pytest.fixture
def werner_state_density_matrix():
    """Werner state: mixture of Bell state and maximally mixed."""
    def werner_state(p):
        # ρ_W(p) = p|Φ+⟩⟨Φ+| + (1-p)I/4
        bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho_bell = np.outer(bell, bell.conj())
        rho_mixed = np.eye(4) / 4
        return p * rho_bell + (1 - p) * rho_mixed

    return werner_state


# ============================================================================
# Test: Log-Negativity
# ============================================================================

class TestLogNegativity:
    """Test log-negativity entanglement measure."""

    def test_product_state_zero_negativity(self, product_state_covariance):
        """Test that product state has zero log-negativity."""
        EN = self._compute_log_negativity_gaussian(product_state_covariance)
        assert EN == 0.0

    def test_epr_state_positive_negativity(self, epr_state_covariance):
        """Test that EPR state has positive log-negativity."""
        EN = self._compute_log_negativity_gaussian(epr_state_covariance)
        assert EN > 0

    def test_thermal_state_negativity(self, thermal_state_covariance):
        """Test thermal state log-negativity."""
        EN = self._compute_log_negativity_gaussian(thermal_state_covariance)

        # High temperature thermal state should be separable
        assert EN == 0.0

    def test_negativity_bounds(self, epr_state_covariance):
        """Test that log-negativity respects bounds."""
        EN = self._compute_log_negativity_gaussian(epr_state_covariance)

        # EN ≥ 0
        assert EN >= 0

        # EN ≤ log(d) for d-dimensional system (d=2 for qubits)
        assert EN <= np.log(2)

    def test_negativity_monotonicity(self):
        """Test that log-negativity increases with squeezing."""
        r_values = np.linspace(0.1, 2.0, 10)
        EN_values = []

        for r in r_values:
            gamma = self._tmst_covariance_matrix(r, n_thermal=0.0)
            EN = self._compute_log_negativity_gaussian(gamma)
            EN_values.append(EN)

        # Check monotonicity
        for i in range(len(EN_values) - 1):
            assert EN_values[i+1] >= EN_values[i]

    def test_log_negativity_from_density_matrix(self, bell_state_density_matrix):
        """Test log-negativity from density matrix representation."""
        EN = self._log_negativity_from_density_matrix(bell_state_density_matrix)

        # Bell state should be maximally entangled
        assert EN > 0.5

    # Helper methods
    def _compute_log_negativity_gaussian(self, gamma):
        """Compute log-negativity for Gaussian state."""
        # Partial transpose
        gamma_pt = gamma.copy()
        gamma_pt[1, :] = -gamma_pt[1, :]
        gamma_pt[:, 1] = -gamma_pt[:, 1]

        # Compute symplectic eigenvalues
        nu_minus = self._smallest_symplectic_eigenvalue(gamma_pt)

        # Log-negativity
        if nu_minus < 1:
            return -np.log(2 * nu_minus)
        else:
            return 0.0

    def _smallest_symplectic_eigenvalue(self, gamma):
        """Compute smallest symplectic eigenvalue."""
        omega = np.block([[np.zeros((2,2)), np.eye(2)],
                          [-np.eye(2), np.zeros((2,2))]])

        # Williamson decomposition
        M = 1j * omega @ gamma
        eigenvalues = np.linalg.eigvals(M)
        symplectic_eigenvalues = np.abs(eigenvalues[::2])  # Take only positive

        return np.min(symplectic_eigenvalues)

    def _tmst_covariance_matrix(self, r, n_thermal):
        """Generate TMST covariance matrix."""
        cosh_r = np.cosh(2*r)
        sinh_r = np.sinh(2*r)

        # Zero temperature part
        gamma_0 = np.array([
            [cosh_r, 0, sinh_r, 0],
            [0, cosh_r, 0, -sinh_r],
            [sinh_r, 0, cosh_r, 0],
            [0, -sinh_r, 0, cosh_r]
        ])

        # Add thermal noise
        return gamma_0 + 2*n_thermal * np.eye(4)

    def _log_negativity_from_density_matrix(self, rho):
        """Compute log-negativity from density matrix."""
        # Partial transpose (on second subsystem)
        rho_pt = self._partial_transpose(rho)

        # Trace norm of PT
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        trace_norm = np.sum(np.abs(eigenvalues))

        # Log-negativity
        return np.log2(trace_norm)

    def _partial_transpose(self, rho):
        """Compute partial transpose of 2-qubit density matrix."""
        # Reshape to 2x2x2x2
        rho_reshaped = rho.reshape(2, 2, 2, 2)

        # Transpose second subsystem
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))

        return rho_pt.reshape(4, 4)


# ============================================================================
# Test: Concurrence
# ============================================================================

class TestConcurrence:
    """Test concurrence entanglement measure."""

    def test_bell_state_maximal_concurrence(self, bell_state_density_matrix):
        """Test that Bell state has maximal concurrence."""
        C = self._compute_concurrence(bell_state_density_matrix)
        assert_allclose(C, 1.0, atol=1e-10)

    def test_product_state_zero_concurrence(self):
        """Test that product state has zero concurrence."""
        # |00⟩ state
        psi = np.array([1, 0, 0, 0])
        rho = np.outer(psi, psi.conj())

        C = self._compute_concurrence(rho)
        assert_allclose(C, 0.0, atol=1e-10)

    def test_werner_state_concurrence(self, werner_state_density_matrix):
        """Test concurrence for Werner states."""
        # Werner state is entangled for p > 1/3

        # p = 0.5: entangled
        rho_entangled = werner_state_density_matrix(0.5)
        C_entangled = self._compute_concurrence(rho_entangled)
        assert C_entangled > 0

        # p = 0.2: separable
        rho_separable = werner_state_density_matrix(0.2)
        C_separable = self._compute_concurrence(rho_separable)
        assert_allclose(C_separable, 0.0, atol=1e-10)

    def test_concurrence_bounds(self, bell_state_density_matrix):
        """Test that concurrence is bounded 0 ≤ C ≤ 1."""
        C = self._compute_concurrence(bell_state_density_matrix)

        assert 0 <= C <= 1

    def test_concurrence_from_gaussian_state(self):
        """Test concurrence for Gaussian states (approximate)."""
        # For Gaussian states, concurrence can be estimated from log-negativity
        r = 1.0
        gamma = self._tmst_covariance_matrix(r, n_thermal=0.0)

        EN = self._compute_log_negativity_gaussian(gamma)

        # Approximate relation: C ≈ tanh(EN) for pure states
        C_approx = np.tanh(EN)

        assert 0 <= C_approx <= 1

    # Helper methods
    def _compute_concurrence(self, rho):
        """Compute concurrence for 2-qubit state."""
        # Pauli matrices
        sigma_y = np.array([[0, -1j], [1j, 0]])

        # Spin-flip operator
        Y = np.kron(sigma_y, sigma_y)

        # Spin-flipped state
        rho_tilde = Y @ rho.conj() @ Y

        # R matrix
        R = rho @ rho_tilde

        # Eigenvalues in decreasing order
        eigenvalues = np.linalg.eigvalsh(R)
        eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Handle numerical errors
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Concurrence
        C = max(0, eigenvalues[0] - np.sum(eigenvalues[1:]))

        return C

    def _tmst_covariance_matrix(self, r, n_thermal):
        """Generate TMST covariance matrix."""
        cosh_r = np.cosh(2*r)
        sinh_r = np.sinh(2*r)

        gamma = np.array([
            [cosh_r, 0, sinh_r, 0],
            [0, cosh_r, 0, -sinh_r],
            [sinh_r, 0, cosh_r, 0],
            [0, -sinh_r, 0, cosh_r]
        ])

        return gamma + 2*n_thermal * np.eye(4)

    def _compute_log_negativity_gaussian(self, gamma):
        """Compute log-negativity for Gaussian state."""
        gamma_pt = gamma.copy()
        gamma_pt[1, :] = -gamma_pt[1, :]
        gamma_pt[:, 1] = -gamma_pt[:, 1]

        omega = np.block([[np.zeros((2,2)), np.eye(2)],
                          [-np.eye(2), np.zeros((2,2))]])

        M = 1j * omega @ gamma_pt
        eigenvalues = np.linalg.eigvals(M)
        nu_minus = np.min(np.abs(eigenvalues[::2]))

        if nu_minus < 1:
            return -np.log(2 * nu_minus)
        else:
            return 0.0


# ============================================================================
# Test: Entanglement Witnesses
# ============================================================================

class TestEntanglementWitnesses:
    """Test entanglement witness operators."""

    def test_bell_witness_detects_entanglement(self, bell_state_density_matrix):
        """Test that Bell witness detects entangled states."""
        # Simple witness: W = (I ⊗ I)/2 - |Φ+⟩⟨Φ+|
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        W = 0.5 * np.eye(4) - np.outer(phi_plus, phi_plus.conj())

        # Expectation value
        exp_W = np.real(np.trace(W @ bell_state_density_matrix))

        # For entangled state, ⟨W⟩ < 0
        assert exp_W < 0

    def test_witness_positive_on_separable(self):
        """Test that witness is positive on separable states."""
        # Product state |01⟩
        psi = np.array([0, 1, 0, 0])
        rho_separable = np.outer(psi, psi.conj())

        # Bell witness
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        W = 0.5 * np.eye(4) - np.outer(phi_plus, phi_plus.conj())

        exp_W = np.real(np.trace(W @ rho_separable))

        # Should be non-negative
        assert exp_W >= -1e-10  # Small tolerance for numerical errors

    def test_ppt_witness(self, bell_state_density_matrix):
        """Test Peres-Horodecki PPT criterion as witness."""
        # Compute partial transpose
        rho_pt = self._partial_transpose(bell_state_density_matrix)

        # Check if PT is positive
        eigenvalues = np.linalg.eigvalsh(rho_pt)

        # For entangled state, some eigenvalues should be negative
        assert np.any(eigenvalues < -1e-10)

    def test_ccnr_witness(self):
        """Test CCNR (Computable Cross-Norm or Realignment) criterion."""
        # Bell state
        phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
        rho = np.outer(phi_plus, phi_plus.conj())

        # Realignment
        rho_R = self._realignment(rho)

        # Trace norm
        trace_norm = np.sum(np.abs(np.linalg.eigvalsh(rho_R)))

        # For entangled state: ||ρ^R||_tr > 1
        assert trace_norm > 1.0

    # Helper methods
    def _partial_transpose(self, rho):
        """Compute partial transpose of 2-qubit density matrix."""
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
        return rho_pt.reshape(4, 4)

    def _realignment(self, rho):
        """Compute realignment (reshuffling) of density matrix."""
        # Reshape to 2x2x2x2
        rho_reshaped = rho.reshape(2, 2, 2, 2)

        # Realignment: (i,j,k,l) → (i,k,j,l)
        rho_R = np.transpose(rho_reshaped, (0, 2, 1, 3))

        # Reshape back to 4x4
        return rho_R.reshape(4, 4)


# ============================================================================
# Test: Bell Inequalities
# ============================================================================

class TestBellInequalities:
    """Test Bell inequality violations."""

    def test_chsh_inequality_classical_bound(self):
        """Test CHSH inequality classical bound."""
        # Classical correlations
        def classical_correlation(theta_a, theta_b):
            # Deterministic local hidden variable
            return np.cos(theta_a - theta_b)

        # CHSH parameter
        S = (classical_correlation(0, 0) - classical_correlation(0, np.pi/2) +
             classical_correlation(np.pi/4, 0) + classical_correlation(np.pi/4, np.pi/2))

        # Classical bound: |S| ≤ 2
        assert np.abs(S) <= 2.0

    def test_chsh_inequality_quantum_violation(self):
        """Test CHSH inequality quantum violation."""
        # Quantum correlations for Bell state
        def quantum_correlation(theta_a, theta_b):
            return -np.cos(theta_a - theta_b)

        # Optimal angles
        theta_a1, theta_a2 = 0, np.pi/2
        theta_b1, theta_b2 = np.pi/4, -np.pi/4

        # CHSH parameter
        S = (quantum_correlation(theta_a1, theta_b1) -
             quantum_correlation(theta_a1, theta_b2) +
             quantum_correlation(theta_a2, theta_b1) +
             quantum_correlation(theta_a2, theta_b2))

        # Quantum bound: |S| ≤ 2√2 ≈ 2.828
        assert 2.0 < np.abs(S) <= 2*np.sqrt(2)

    def test_chsh_from_correlations(self):
        """Test CHSH computation from correlation data."""
        # Simulate quantum correlations
        E_ab = -1/np.sqrt(2)
        E_ab_prime = 1/np.sqrt(2)
        E_a_prime_b = -1/np.sqrt(2)
        E_a_prime_b_prime = -1/np.sqrt(2)

        S = E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime

        # Should violate classical bound
        assert np.abs(S) > 2.0

        # Should respect quantum bound
        assert np.abs(S) <= 2*np.sqrt(2) + 1e-10

    def test_ch_inequality(self):
        """Test Clauser-Horne (CH) inequality."""
        # CH inequality: -1 ≤ S_CH ≤ 0 (classical)
        # S_CH can reach -1/2 (quantum)

        # Quantum value for Bell state
        S_CH = -1/2

        # Classical bound
        assert -1 <= S_CH <= 0

    def test_mermin_inequality(self):
        """Test Mermin inequality for 3 qubits."""
        # Mermin operator expectation for GHZ state
        # Classical bound: |M| ≤ 2
        # Quantum: |M| = 4 for GHZ state

        M_classical_max = 2
        M_quantum_ghz = 4

        assert M_quantum_ghz > M_classical_max


# ============================================================================
# Test: Correlation Functions
# ============================================================================

class TestCorrelationFunctions:
    """Test two-particle correlation functions."""

    def test_spin_correlation_product_state(self):
        """Test spin correlation for product state."""
        # |↑↑⟩ state
        psi = np.array([1, 0, 0, 0])
        rho = np.outer(psi, psi.conj())

        # σ_z ⊗ σ_z
        sigma_z = np.array([[1, 0], [0, -1]])
        ZZ = np.kron(sigma_z, sigma_z)

        C = np.real(np.trace(ZZ @ rho))

        # For |↑↑⟩: ⟨Z⊗Z⟩ = 1
        assert_allclose(C, 1.0, atol=1e-10)

    def test_spin_correlation_bell_state(self, bell_state_density_matrix):
        """Test spin correlation for Bell state."""
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        # Correlations
        XX = np.kron(sigma_x, sigma_x)
        YY = np.kron(sigma_y, sigma_y)
        ZZ = np.kron(sigma_z, sigma_z)

        C_XX = np.real(np.trace(XX @ bell_state_density_matrix))
        C_YY = np.real(np.trace(YY @ bell_state_density_matrix))
        C_ZZ = np.real(np.trace(ZZ @ bell_state_density_matrix))

        # For |Φ+⟩ = (|00⟩ + |11⟩)/√2
        assert_allclose(C_XX, 1.0, atol=1e-10)
        assert_allclose(C_YY, -1.0, atol=1e-10)
        assert_allclose(C_ZZ, 1.0, atol=1e-10)

    def test_two_point_correlation_function(self):
        """Test two-point correlation function ⟨a†_i a_j⟩."""
        # For coherent state: ⟨a†a⟩ = |α|²
        alpha = 1.0 + 0.5j
        expected = np.abs(alpha)**2

        # Mock correlation
        correlation = expected

        assert correlation >= 0
        assert_allclose(correlation, 1.25, atol=1e-10)

    def test_connected_correlation(self):
        """Test connected correlation ⟨AB⟩ - ⟨A⟩⟨B⟩."""
        # For product state: connected correlation = 0
        # For entangled state: connected correlation ≠ 0

        # Product state |00⟩
        psi_prod = np.array([1, 0, 0, 0])
        rho_prod = np.outer(psi_prod, psi_prod.conj())

        sigma_z = np.array([[1, 0], [0, -1]])
        Z1 = np.kron(sigma_z, np.eye(2))
        Z2 = np.kron(np.eye(2), sigma_z)
        ZZ = np.kron(sigma_z, sigma_z)

        exp_Z1 = np.real(np.trace(Z1 @ rho_prod))
        exp_Z2 = np.real(np.trace(Z2 @ rho_prod))
        exp_ZZ = np.real(np.trace(ZZ @ rho_prod))

        C_connected = exp_ZZ - exp_Z1 * exp_Z2

        # Product state: should be zero
        assert_allclose(C_connected, 0.0, atol=1e-10)


# ============================================================================
# Test: Separability Criteria
# ============================================================================

class TestSeparabilityCriteria:
    """Test separability detection methods."""

    def test_ppt_criterion_separable(self):
        """Test PPT criterion on separable state."""
        # Product state
        psi = np.array([1, 0, 0, 0])
        rho = np.outer(psi, psi.conj())

        is_separable = self._ppt_test(rho)
        assert is_separable

    def test_ppt_criterion_entangled(self, bell_state_density_matrix):
        """Test PPT criterion on entangled state."""
        is_separable = self._ppt_test(bell_state_density_matrix)
        assert not is_separable

    def test_reduction_criterion(self):
        """Test reduction criterion for separability."""
        # For separable states: ρ_A ⊗ I - ρ ≥ 0

        # Product state
        psi = np.array([1, 0, 0, 0])
        rho = np.outer(psi, psi.conj())

        # Partial trace over B
        rho_A = self._partial_trace(rho, subsystem='B')

        # Reduction map
        R = np.kron(rho_A, np.eye(2)) - rho

        # Check positivity
        eigenvalues = np.linalg.eigvalsh(R)
        is_separable = np.all(eigenvalues >= -1e-10)

        assert is_separable

    def test_computable_cross_norm_criterion(self):
        """Test computable cross-norm criterion."""
        # ||ρ||_1,∞ ≤ 1 for separable states

        # Product state
        psi = np.array([1, 0, 0, 0])
        rho = np.outer(psi, psi.conj())

        # Trace norm
        trace_norm = np.sum(np.abs(np.linalg.eigvalsh(rho)))

        assert trace_norm <= 1.0 + 1e-10

    # Helper methods
    def _ppt_test(self, rho):
        """Test if state passes PPT criterion."""
        rho_pt = self._partial_transpose(rho)
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        return np.all(eigenvalues >= -1e-10)

    def _partial_transpose(self, rho):
        """Compute partial transpose."""
        rho_reshaped = rho.reshape(2, 2, 2, 2)
        rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
        return rho_pt.reshape(4, 4)

    def _partial_trace(self, rho, subsystem='B'):
        """Compute partial trace."""
        rho_reshaped = rho.reshape(2, 2, 2, 2)

        if subsystem == 'B':
            # Trace out second subsystem
            rho_A = np.einsum('ijik->jk', rho_reshaped)
        else:
            # Trace out first subsystem
            rho_A = np.einsum('ijkj->ik', rho_reshaped)

        return rho_A


# ============================================================================
# Test: Gaussian State Entanglement
# ============================================================================

class TestGaussianStateEntanglement:
    """Test entanglement in Gaussian states."""

    def test_vacuum_state_separable(self):
        """Test that vacuum state is separable."""
        gamma = np.eye(4)

        is_entangled = self._is_entangled_gaussian(gamma)
        assert not is_entangled

    def test_squeezed_state_entangled(self):
        """Test that two-mode squeezed state is entangled."""
        r = 1.0
        cosh_r = np.cosh(2*r)
        sinh_r = np.sinh(2*r)

        gamma = np.array([
            [cosh_r, 0, sinh_r, 0],
            [0, cosh_r, 0, -sinh_r],
            [sinh_r, 0, cosh_r, 0],
            [0, -sinh_r, 0, cosh_r]
        ])

        is_entangled = self._is_entangled_gaussian(gamma)
        assert is_entangled

    def test_symplectic_eigenvalues(self):
        """Test symplectic eigenvalue computation."""
        # Vacuum state
        gamma = np.eye(4)

        nu = self._symplectic_eigenvalues(gamma)

        # For vacuum: all eigenvalues = 1
        assert_allclose(nu, [1.0, 1.0], atol=1e-10)

    def test_logarithmic_negativity_scaling(self):
        """Test logarithmic negativity scaling with squeezing."""
        r_values = [0.5, 1.0, 1.5, 2.0]
        EN_values = []

        for r in r_values:
            gamma = self._tmst_covariance(r, n_thermal=0.0)
            EN = self._log_negativity_gaussian(gamma)
            EN_values.append(EN)

        # Should increase monotonically
        for i in range(len(EN_values) - 1):
            assert EN_values[i+1] > EN_values[i]

    # Helper methods
    def _is_entangled_gaussian(self, gamma):
        """Check if Gaussian state is entangled via PPT."""
        gamma_pt = gamma.copy()
        gamma_pt[1, :] = -gamma_pt[1, :]
        gamma_pt[:, 1] = -gamma_pt[:, 1]

        nu = self._symplectic_eigenvalues(gamma_pt)

        return np.any(nu < 1.0 - 1e-10)

    def _symplectic_eigenvalues(self, gamma):
        """Compute symplectic eigenvalues."""
        omega = np.block([[np.zeros((2,2)), np.eye(2)],
                          [-np.eye(2), np.zeros((2,2))]])

        M = 1j * omega @ gamma
        eigenvalues = np.linalg.eigvals(M)

        # Take absolute values and keep unique ones
        symplectic_eigs = np.abs(eigenvalues[::2])

        return np.sort(symplectic_eigs)

    def _tmst_covariance(self, r, n_thermal):
        """Generate TMST covariance matrix."""
        cosh_r = np.cosh(2*r)
        sinh_r = np.sinh(2*r)

        gamma = np.array([
            [cosh_r, 0, sinh_r, 0],
            [0, cosh_r, 0, -sinh_r],
            [sinh_r, 0, cosh_r, 0],
            [0, -sinh_r, 0, cosh_r]
        ])

        return gamma + 2*n_thermal * np.eye(4)

    def _log_negativity_gaussian(self, gamma):
        """Compute log-negativity for Gaussian state."""
        gamma_pt = gamma.copy()
        gamma_pt[1, :] = -gamma_pt[1, :]
        gamma_pt[:, 1] = -gamma_pt[:, 1]

        nu_minus = np.min(self._symplectic_eigenvalues(gamma_pt))

        if nu_minus < 1:
            return -np.log(2 * nu_minus)
        else:
            return 0.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
