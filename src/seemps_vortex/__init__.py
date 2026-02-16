"""
SeeMPS Vortex Package
=====================

Implementation of QCD center vortex dynamics using Matrix Product States.

Based on theoretical framework:
- Topological Vortex Superradiance (file:9)
- Non-Hermitian Topology & Exceptional Points (file:7)
- Inertial Dynamics of Massive Vortices (file:8)
- Entanglement Dominance at Zero-T (file:6)

Modules:
--------
- center_vortex: Core MPS vortex dynamics
- collective_squeezing: Two-mode squeezing operators
- mvc_threshold: MVC threshold detection & exceptional points
- entanglement_detection: Entanglement measures (log-negativity, entropy)

Author: [Your Name]
Date: February 2026
Version: 1.0.0
"""

from .center_vortex import CenterVortexMPS
from .collective_squeezing import (
    CollectiveSqueezing,
    TMSTState,
    SuperradiantOperator
)
from .mvc_threshold import (
    MVCThresholdDetector,
    ExceptionalPointAnalyzer,
    PetermannFactor
)
from .entanglement_detection import (
    EntanglementMeasures,
    LogNegativity,
    VonNeumannEntropy,
    ConcurrenceEstimator
)

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # Core class
    'CenterVortexMPS',
    
    # Squeezing operators
    'CollectiveSqueezing',
    'TMSTState',
    'SuperradiantOperator',
    
    # MVC threshold
    'MVCThresholdDetector',
    'ExceptionalPointAnalyzer',
    'PetermannFactor',
    
    # Entanglement detection
    'EntanglementMeasures',
    'LogNegativity',
    'VonNeumannEntropy',
    'ConcurrenceEstimator',
]

# Package-level configuration
DEFAULT_CHI_MAX = 64
DEFAULT_N_SITES = 128
DEFAULT_TOLERANCE = 1e-10

# Physical constants (in natural units ħ = c = 1)
LAMBDA_QCD = 0.217  # GeV, QCD scale
T_PLANCK = 1.221e19  # GeV, Planck temperature (for MVC calculations)
