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
- tmst_threshold: Theorem 4.3.1 analytic threshold + phase-transition events  ← NEW
- phase_diagram: Stable visualization API for TMST phase diagrams             ← NEW

Author: [Javier Manuel Martín Alonso]
Date: February 2026
Version: 1.1.0
"""

# ── Existing modules (unchanged) ──────────────────────────────────────────────
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

# ── NEW: Theorem 4.3.1 threshold module ───────────────────────────────────────
from .tmst_threshold import (
    bose_einstein,
    critical_squeezing,
    log_negativity,
    min_symplectic_eigenvalue,
    scan_phase_diagram,
    TMSTPhaseEvent,
)

# ── NEW: Phase diagram visualization API ──────────────────────────────────────
from .phase_diagram import (
    compute_phase_diagram,
    plot_phase_diagram,
)

__version__ = "1.1.0"
__author__ = "Your Name"
__license__ = "MIT"

__all__ = [
    # ── Existing ──────────────────────────────────────────────────────────────
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

    # ── NEW ───────────────────────────────────────────────────────────────────
    # TMST threshold (Theorem 4.3.1)
    'bose_einstein',
    'critical_squeezing',
    'log_negativity',
    'min_symplectic_eigenvalue',
    'scan_phase_diagram',
    'TMSTPhaseEvent',

    # Phase diagram visualization
    'compute_phase_diagram',
    'plot_phase_diagram',
]

# ── Package-level configuration (unchanged) ───────────────────────────────────
DEFAULT_CHI_MAX = 64
DEFAULT_N_SITES = 128
DEFAULT_TOLERANCE = 1e-10

# Physical constants (in natural units ħ = c = 1)
LAMBDA_QCD = 0.217      # GeV, QCD scale
T_PLANCK = 1.221e19     # GeV, Planck temperature (for MVC calculations)

