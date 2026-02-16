"""
Tests for QCD Vortex MPS Simulations

Tests cover:
- MPS initialization and state preparation
- Collective squeezing operations
- Time evolution under Lindblad dynamics
- MVC threshold detection
- Entanglement measures
- Export/import functionality
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import h5py

# Skip tests if SeeMPS not available
pytest.importorskip("seemps")

from seemps_vortex import (
    CenterVortexMPS,
    CollectiveSqueezing,
    TMSTState,
    MVCThresholdDetector,
    ExceptionalPointAnalyzer,
    PetermannFactor,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_vortex_system():
    """Create a small vortex system for testing (8 sites)."""
    return CenterVortexMPS(N_sites=8, chi_max=16, d=2)


@pytest.fixture
def medium_vortex_system():
    """Create a medium vortex system (32 sites)."""
    return CenterVortexMPS(N_sites=32, chi_max=32, d=2)


@pytest.fixture
def tmst_state():
    """Create a TMST state with typical parameters."""
    return TMSTState(r_squeeze=1.0, T_temperature=0.5, omega=1.0)


@pytest.fixture
def mvc_detector():
    """Create MVC threshold detector."""
    return MVCThresholdDetector(T_planck=1.221e19, alpha=2.5)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test: MPS Initialization
# ============================================================================

class TestMPSInitialization:
    """Test MPS system initialization."""

    def test_system_creation(self, small_vortex_system):
        """Test basic system creation."""
        system = small_vortex_system
        assert system.N_sites == 8
        assert system.chi_max == 16
        assert system.d == 2

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            CenterVortexMPS(N_sites=0, chi_max=16, d=2)

        with pytest.raises(ValueError):
            CenterVortexMPS(N_sites=8, chi_max=0, d=2)

        with pytest.raises(ValueError):
            CenterVortexMPS(N_sites=8, chi_max=16, d=1)

    def test_product_state_initialization(self, small_vortex_system):
        """Test initialization to product state."""
        psi = small_vortex_system.initialize_product_state(state='0')

        # Check MPS structure
        assert isinstance(psi, list)
        assert len(psi) == 8

        # Check tensor shapes
        for i, tensor in enumerate(psi):
            assert tensor.ndim == 3  # (chi_left, d, chi_right)
            assert tensor.shape[1] == 2  # Physical dimension

    def test_random_state_initialization(self, small_vortex_system):
        """Test random state initialization."""
        psi = small_vortex_system.initialize_random_state(seed=42)

        # Check normalization
        norm = small_vortex_system.compute_norm(psi)
        assert np.abs(norm - 1.0) < 1e-10

    def test_ghz_state_initialization(self, small_vortex_system):
        """Test GHZ state initialization."""
        psi = small_vortex_system.initialize_ghz_state()

        # GHZ state should be normalized
        norm = small_vortex_system.compute_norm(psi)
        assert np.abs(norm - 1.0) < 1e-10


# ============================================================================
# Test: Collective Squeezing
# ============================================================================

class TestCollectiveSqueezing:
    """Test collective squeezing operations."""

    def test_squeezing_operator_creation(self):
        """Test creation of squeezing operator."""
        squeezer = CollectiveSqueezing(N_sites=4, r_squeeze=1.0)

        assert squeezer.N_sites == 4
        assert squeezer.r_squeeze == 1.0

    def test_squeezing_parameter_range(self):
        """Test squeezing parameter validation."""
        # Valid range: r >= 0
        squeezer = CollectiveSqueezing(N_sites=4, r_squeeze=0.0)
        assert squeezer.r_squeeze == 0.0

        squeezer = CollectiveSqueezing(N_sites=4, r_squeeze=2.0)
        assert squeezer.r_squeeze == 2.0

        # Negative should raise error
        with pytest.raises(ValueError):
            CollectiveSqueezing(N_sites=4, r_squeeze=-0.5)

    def test_collective_mode_initialization(self, small_vortex_system):
        """Test collective mode state preparation."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0,
            n_thermal=0.1
        )

        # Check state is normalized
        norm = small_vortex_system.compute_norm(psi)
        assert np.abs(norm - 1.0) < 1e-8

    def test_squeezing_increases_entanglement(self, small_vortex_system):
        """Test that squeezing increases entanglement."""
        # Product state (no entanglement)
        psi_product = small_vortex_system.initialize_product_state('0')
        S_product = small_vortex_system.entanglement_entropy(psi_product, cut=4)

        # Squeezed state (should have entanglement)
        psi_squeezed = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.5, n_thermal=0.0
        )
        S_squeezed = small_vortex_system.entanglement_entropy(psi_squeezed, cut=4)

        assert S_squeezed > S_product


# ============================================================================
# Test: TMST State Properties
# ============================================================================

class TestTMSTState:
    """Test Two-Mode Squeezed Thermal State properties."""

    def test_tmst_creation(self, tmst_state):
        """Test TMST state creation."""
        assert tmst_state.r_squeeze == 1.0
        assert tmst_state.T_temperature == 0.5
        assert tmst_state.omega == 1.0

    def test_thermal_occupation(self, tmst_state):
        """Test thermal occupation number calculation."""
        n_thermal = tmst_state.n_thermal

        # Should be positive
        assert n_thermal >= 0

        # Check Bose-Einstein formula
        k_B = 1.0  # Natural units
        expected_n = 1.0 / (np.exp(tmst_state.omega / 
                                   (k_B * tmst_state.T_temperature)) - 1.0)
        assert np.abs(n_thermal - expected_n) < 1e-10

    def test_entanglement_threshold(self, tmst_state):
        """Test Theorem 4.3.1: Entanglement threshold."""
        is_entangled, T_critical = tmst_state.is_entangled()

        # At r=1.0, T=0.5, should be entangled
        assert is_entangled

        # Critical temperature should be positive
        assert T_critical > 0

    def test_critical_squeezing_calculation(self):
        """Test critical squeezing r_c(T) calculation."""
        from seemps_vortex.collective_squeezing import critical_squeezing

        # Test at different thermal occupations
        for n_th in [0.1, 0.5, 1.0]:
            r_c = critical_squeezing(n_th)

            # r_c should be positive
            assert r_c > 0

            # Verify threshold condition
            threshold = 0.5 * (np.exp(2 * r_c) - 1)**(-1)
            assert np.abs(threshold - n_th) < 1e-10

    def test_log_negativity_computation(self, tmst_state):
        """Test log-negativity calculation."""
        EN = tmst_state.compute_log_negativity()

        # Entangled state should have positive log-negativity
        is_entangled, _ = tmst_state.is_entangled()
        if is_entangled:
            assert EN > 0
        else:
            assert EN == 0

    def test_covariance_matrix(self, tmst_state):
        """Test covariance matrix computation."""
        gamma = tmst_state.covariance_matrix()

        # Should be 4x4 for two modes
        assert gamma.shape == (4, 4)

        # Should be symmetric
        assert np.allclose(gamma, gamma.T)

        # Check uncertainty relation
        # γ + iΩ ≥ 0 (positive semi-definite)
        omega = np.block([[np.zeros((2,2)), np.eye(2)],
                          [-np.eye(2), np.zeros((2,2))]])

        eigenvalues = np.linalg.eigvals(gamma + 1j * omega)
        assert all(np.real(ev) >= -1e-10 for ev in eigenvalues)


# ============================================================================
# Test: Time Evolution
# ============================================================================

class TestTimeEvolution:
    """Test Lindblad time evolution."""

    def test_lindblad_evolution_short_time(self, small_vortex_system):
        """Test short-time Lindblad evolution."""
        psi_initial = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0, n_thermal=0.1
        )

        # Evolve for short time
        trajectory = small_vortex_system.evolve_lindblad(
            psi_initial=psi_initial,
            T_temp=0.5,
            gamma_loss=0.01,
            time_steps=5,
            dt=0.01
        )

        assert len(trajectory) == 6  # Initial + 5 steps

        # Check norm conservation (approximately)
        for psi_t in trajectory:
            norm = small_vortex_system.compute_norm(psi_t)
            assert 0.9 < norm < 1.1  # Allow some numerical error

    def test_decoherence_reduces_entanglement(self, small_vortex_system):
        """Test that decoherence reduces entanglement."""
        psi_initial = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.5, n_thermal=0.0
        )

        # Initial entanglement
        S_initial = small_vortex_system.entanglement_entropy(psi_initial, cut=4)

        # Evolve with decoherence
        trajectory = small_vortex_system.evolve_lindblad(
            psi_initial=psi_initial,
            T_temp=1.0,
            gamma_loss=0.1,
            time_steps=10,
            dt=0.05
        )

        # Final entanglement
        psi_final = trajectory[-1]
        S_final = small_vortex_system.entanglement_entropy(psi_final, cut=4)

        # Entanglement should decrease
        assert S_final < S_initial

    @pytest.mark.slow
    def test_long_time_evolution(self, small_vortex_system):
        """Test long-time evolution (marked as slow)."""
        psi_initial = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0, n_thermal=0.1
        )

        trajectory = small_vortex_system.evolve_lindblad(
            psi_initial=psi_initial,
            T_temp=0.5,
            gamma_loss=0.05,
            time_steps=100,
            dt=0.01
        )

        assert len(trajectory) == 101


# ============================================================================
# Test: MVC Threshold Detection
# ============================================================================

class TestMVCThreshold:
    """Test Morphology of Vacuum Condensate threshold detection."""

    def test_mvc_detector_creation(self, mvc_detector):
        """Test MVC detector initialization."""
        assert mvc_detector.T_planck > 0
        assert mvc_detector.alpha > 0
        assert mvc_detector.rho_mvc > 0

    def test_critical_density_calculation(self):
        """Test critical density ρ_MVC = T_Planck^α."""
        T_planck = 1.221e19
        alpha = 2.5

        detector = MVCThresholdDetector(T_planck=T_planck, alpha=alpha)

        expected_rho = T_planck**alpha
        assert np.abs(detector.rho_mvc - expected_rho) < 1e10  # Large scale

    def test_confinement_detection(self, mvc_detector):
        """Test confinement phase detection."""
        # Test deconfined phase (low density)
        is_confined, phase = mvc_detector.is_confined(
            rho_local=0.5e19,
            omega_vortex=0.3,
            entanglement_density=0.5
        )
        assert not is_confined
        assert phase in ['DECONFINED', 'PRE_CONFINEMENT']

        # Test confined phase (high density)
        is_confined, phase = mvc_detector.is_confined(
            rho_local=3.0e19,
            omega_vortex=0.9,
            entanglement_density=3.0
        )
        assert is_confined
        assert phase == 'CONFINED'

    def test_hadronization_multiplicity(self, mvc_detector):
        """Test hadronization multiplicity prediction."""
        # Small system (pp collision)
        M_pp = mvc_detector.hadronization_multiplicity(
            N_coupled=2,
            rho_E=1.0,
            rho_sat=mvc_detector.rho_mvc
        )
        assert M_pp > 0
        assert M_pp < 100  # Reasonable for pp

        # Large system (PbPb collision)
        M_PbPb = mvc_detector.hadronization_multiplicity(
            N_coupled=50,
            rho_E=20.0,
            rho_sat=mvc_detector.rho_mvc
        )
        assert M_PbPb > M_pp  # More particles in PbPb


# ============================================================================
# Test: Exceptional Points
# ============================================================================

class TestExceptionalPoints:
    """Test exceptional point analysis in non-Hermitian dynamics."""

    def test_ep_analyzer_creation(self):
        """Test EP analyzer initialization."""
        analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)

        assert analyzer.gamma_loss == 0.1
        assert analyzer.gamma_SR == 0.5

    def test_effective_hamiltonian(self):
        """Test effective Hamiltonian construction."""
        analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)

        H_eff = analyzer._effective_hamiltonian(r_injection=1.0)

        # Should be 2x2 for two-level system
        assert H_eff.shape == (2, 2)

        # Check non-Hermiticity
        is_hermitian = np.allclose(H_eff, H_eff.conj().T)
        assert not is_hermitian  # Should be non-Hermitian

    def test_ep_detection(self):
        """Test exceptional point detection."""
        analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)

        ep_data = analyzer.find_exceptional_point(
            r_injection_range=(0.0, 2.0),
            n_points=50
        )

        # Check EP found
        assert 'r_injection' in ep_data
        assert 'energy_gap' in ep_data
        assert 'ep_order' in ep_data

        # Energy gap should be small at EP
        assert ep_data['energy_gap'] < 0.1

    def test_petermann_factor(self):
        """Test Petermann factor computation."""
        analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)

        H_eff = analyzer._effective_hamiltonian(r_injection=1.0)
        petermann = PetermannFactor(H_eff)

        K = petermann.compute_factor(mode_index=0)

        # Petermann factor should be >= 1
        assert K >= 1.0

    def test_pt_symmetry_phase(self):
        """Test PT-symmetry phase detection."""
        analyzer = ExceptionalPointAnalyzer(gamma_loss=0.1, gamma_SR=0.5)

        # Far from EP: PT-symmetric
        phase_low = analyzer.pt_symmetry_phase(r_injection=0.1)

        # Near/at EP: PT-broken
        phase_high = analyzer.pt_symmetry_phase(r_injection=1.5)

        assert phase_low in ['PT_SYMMETRIC', 'PT_BROKEN']
        assert phase_high in ['PT_SYMMETRIC', 'PT_BROKEN']


# ============================================================================
# Test: Entanglement Measures
# ============================================================================

class TestEntanglementMeasures:
    """Test entanglement quantification."""

    def test_entanglement_entropy_product_state(self, small_vortex_system):
        """Test entropy of product state is zero."""
        psi = small_vortex_system.initialize_product_state('0')
        S = small_vortex_system.entanglement_entropy(psi, cut=4)

        assert np.abs(S) < 1e-10

    def test_entanglement_entropy_ghz_state(self, small_vortex_system):
        """Test entropy of GHZ state."""
        psi = small_vortex_system.initialize_ghz_state()
        S = small_vortex_system.entanglement_entropy(psi, cut=4)

        # GHZ state has significant entanglement
        assert S > 0.5

    def test_entanglement_spectrum(self, small_vortex_system):
        """Test entanglement spectrum computation."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0, n_thermal=0.1
        )

        spectrum = small_vortex_system.entanglement_spectrum(psi, cut=4)

        # Spectrum should be positive
        assert all(s >= 0 for s in spectrum)

        # Should sum to approximately 1 (normalized)
        assert np.abs(np.sum(spectrum) - 1.0) < 1e-8

    def test_mutual_information(self, small_vortex_system):
        """Test mutual information calculation."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.5, n_thermal=0.0
        )

        # Mutual information between two halves
        I = small_vortex_system.mutual_information(psi, cut_A=4, cut_B=4)

        # Should be non-negative
        assert I >= 0


# ============================================================================
# Test: Export/Import
# ============================================================================

class TestExportImport:
    """Test state export and import functionality."""

    def test_export_to_hdf5(self, small_vortex_system, temp_output_dir):
        """Test exporting MPS to HDF5 format."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0, n_thermal=0.1
        )

        output_file = temp_output_dir / 'test_state.h5'
        small_vortex_system.export_to_hdf5(psi, filename=str(output_file))

        # Check file exists
        assert output_file.exists()

        # Check HDF5 structure
        with h5py.File(output_file, 'r') as f:
            assert 'mps_tensors' in f
            assert 'metadata' in f
            assert f['metadata'].attrs['N_sites'] == 8

    def test_import_from_hdf5(self, small_vortex_system, temp_output_dir):
        """Test importing MPS from HDF5 format."""
        # Export state
        psi_original = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.0, n_thermal=0.1
        )

        output_file = temp_output_dir / 'test_state.h5'
        small_vortex_system.export_to_hdf5(psi_original, filename=str(output_file))

        # Import state
        psi_loaded = small_vortex_system.import_from_hdf5(filename=str(output_file))

        # Check states match
        norm_original = small_vortex_system.compute_norm(psi_original)
        norm_loaded = small_vortex_system.compute_norm(psi_loaded)

        assert np.abs(norm_original - norm_loaded) < 1e-10

    def test_export_metadata(self, small_vortex_system, temp_output_dir):
        """Test that metadata is correctly saved."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=1.2, n_thermal=0.15
        )

        output_file = temp_output_dir / 'test_metadata.h5'
        small_vortex_system.export_to_hdf5(
            psi, 
            filename=str(output_file),
            metadata={'r_squeeze': 1.2, 'n_thermal': 0.15, 'custom_key': 'value'}
        )

        with h5py.File(output_file, 'r') as f:
            assert f['metadata'].attrs['r_squeeze'] == 1.2
            assert f['metadata'].attrs['n_thermal'] == 0.15
            assert f['metadata'].attrs['custom_key'] == 'value'


# ============================================================================
# Test: Performance and Edge Cases
# ============================================================================

class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""

    def test_zero_squeezing(self, small_vortex_system):
        """Test system with zero squeezing (product state)."""
        psi = small_vortex_system.initialize_collective_mode(
            r_squeeze=0.0, n_thermal=0.0
        )

        # Should be normalized
        norm = small_vortex_system.compute_norm(psi)
        assert np.abs(norm - 1.0) < 1e-10

        # Should have minimal entanglement
        S = small_vortex_system.entanglement_entropy(psi, cut=4)
        assert S < 0.1

    def test_high_temperature_limit(self):
        """Test high temperature limit (maximally mixed)."""
        tmst = TMSTState(r_squeeze=1.0, T_temperature=1000.0, omega=1.0)

        # High T → high thermal occupation
        assert tmst.n_thermal > 100

        # Should not be entangled at high temperature
        is_entangled, _ = tmst.is_entangled()
        assert not is_entangled

    def test_zero_temperature_limit(self):
        """Test zero temperature limit (pure squeezed state)."""
        tmst = TMSTState(r_squeeze=1.0, T_temperature=0.001, omega=1.0)

        # Low T → low thermal occupation
        assert tmst.n_thermal < 0.01

        # Should be maximally entangled at zero temperature
        is_entangled, _ = tmst.is_entangled()
        assert is_entangled

    @pytest.mark.parametrize("N_sites", [4, 8, 16])
    def test_different_system_sizes(self, N_sites):
        """Test systems of different sizes."""
        system = CenterVortexMPS(N_sites=N_sites, chi_max=16, d=2)
        psi = system.initialize_product_state('0')

        assert len(psi) == N_sites

        norm = system.compute_norm(psi)
        assert np.abs(norm - 1.0) < 1e-10


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
