"""
Belle II Analysis Package
=========================

Tools for massive data classification and analysis of Belle II τ⁺τ⁻ entanglement
data on HPC clusters using DIRAC grid computing.

Theoretical framework:
- file:3: Fermionic Bulk-Boundary Algorithm Adaptation
- file:2: Belle II Tau Pair Entanglement Toy MC
- file:9: Collective vortex projection and TMSS generation

Modules:
--------
- grid_submission: DIRAC gbasf2 job management
- data_classifier: ML-based event classification (entangled vs thermal)
- correlation_analysis: Two-particle correlation extraction and fitting

Belle II Computing Infrastructure:
- 55 computing sites worldwide
- DIRAC middleware for job distribution
- basf2 analysis framework
- ROOT-based data storage

Author: [Your Name]
Date: February 2026
Version: 1.0.0
"""

from .grid_submission import BelleIIGridAnalysis
from .data_classifier import (
    TauPairClassifier,
    EntanglementFeatureExtractor,
    KinematicSelector
)
from .correlation_analysis import (
    TwoParticleCorrelator,
    SpinCorrelationAnalyzer,
    BellInequalityTester
)

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # Grid submission
    'BelleIIGridAnalysis',
    
    # Classification
    'TauPairClassifier',
    'EntanglementFeatureExtractor',
    'KinematicSelector',
    
    # Correlation analysis
    'TwoParticleCorrelator',
    'SpinCorrelationAnalyzer',
    'BellInequalityTester',
]

# Belle II experiment parameters
BELLE2_CONFIG = {
    'center_of_mass_energy': 10.58,  # GeV (Υ(4S) mass)
    'beam_energy_HER': 7.0,  # GeV (High Energy Ring)
    'beam_energy_LER': 4.0,  # GeV (Low Energy Ring)
    'luminosity_design': 8e35,  # cm⁻² s⁻¹
    'integrated_luminosity_2024': 424,  # fb⁻¹
    'tau_pair_cross_section': 0.919,  # nb
}

# Grid computing sites (major ones)
BELLE2_GRID_SITES = [
    'KEK-CC',  # Japan (Tier-0)
    'BNL-OSG',  # USA (Tier-1)
    'INFN-CNAF',  # Italy (Tier-1)
    'GridKa',  # Germany (Tier-1)
    'RAL-LCG2',  # UK (Tier-1)
    'IN2P3-CC',  # France (Tier-1)
    'DESY-HH',  # Germany
    'KISTI-GSDC',  # Korea
]

# Event selection criteria (file:2, file:3)
EVENT_SELECTION = {
    'electron_id_threshold': 0.9,  # electronID > 0.9
    'muon_id_threshold': 0.9,
    'min_tau_momentum': 0.5,  # GeV/c
    'max_tau_momentum': 6.0,  # GeV/c
    'cms_energy_window': (10.0, 11.0),  # GeV
    'max_missing_mass': 2.0,  # GeV/c²
}

# Entanglement classification thresholds (from file:3)
ENTANGLEMENT_THRESHOLDS = {
    'concurrence_min': 0.1,  # C > 0.1 → entangled
    'log_negativity_min': 0.05,  # E_N > 0.05
    'bell_inequality_violation': 2.0,  # S > 2 (CHSH)
    'kinematic_regime': {
        'low_pT': (0.5, 2.0),  # GeV/c
        'medium_pT': (2.0, 4.0),
        'high_pT': (4.0, 6.0),
    }
}

def print_belle2_info():
    """Print Belle II experiment and grid information."""
    print("="*70)
    print("Belle II Experiment Configuration")
    print("="*70)
    for key, value in BELLE2_CONFIG.items():
        print(f"  {key:30s}: {value}")
    print(f"\n  Number of grid sites: {len(BELLE2_GRID_SITES)}")
    print("="*70)
