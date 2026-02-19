# test/test_tmst_injection.py
"""
Injection-style unit tests for TMST log-negativity and Theorem 4.3.1.
Fixes failing tests related to:
  - log-negativity of product vs EPR states
  - monotonicity of log-negativity with squeezing parameter
  - Gaussian entanglement detection (PPT/symplectic eigenvalues)

Run with:  pytest test/test_tmst_injection.py -v
"""

import pytest
import numpy as np

from src.seemps_vortex.tmst_threshold import (
    bose_einstein,
    critical_squeezing,
    log_negativity,
    min_symplectic_eigenvalue,
)
from test.validation_tools import make_tmst_signal, make_tmst_noise

OMEGA = 1.0


# ──────────────────────────────────────────
# 1. Threshold ON / OFF injection test
# ──────────────────────────────────────────

@pytest.mark.parametrize("T", np.linspace(0.1, 3.0, 7).tolist())
def test_tmst_separable_below_threshold(T):
    """
    r < r_c(T) → E_N = 0 (noise-dominated, separable).
    """
    r_noise, n_bar = make_tmst_noise(T, alpha_below=0.5)
    EN = float(log_negativity(r_noise, n_bar))
    assert EN < 1e-9, (
        f"T={T:.3f}: expected E_N≈0 for r={r_noise:.4f} < r_c, got {EN:.6f}"
    )


@pytest.mark.parametrize("T", np.linspace(0.1, 3.0, 7).tolist())
def test_tmst_entangled_above_threshold(T):
    """
    r > r_c(T) → E_N > 0 (entanglement-dominant).
    """
    r_signal, n_bar = make_tmst_signal(T, delta_above=0.3)
    EN = float(log_negativity(r_signal, n_bar))
    assert EN > 0.05, (
        f"T={T:.3f}: expected E_N>0 for r={r_signal:.4f} > r_c, got {EN:.6f}"
    )


# ──────────────────────────────────────────
# 2. Monotonicity test (was failing)
# ──────────────────────────────────────────

@pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
def test_log_negativity_monotone_with_squeezing(T):
    """
    For fixed T and r well above r_c(T), E_N(r) must be strictly increasing.
    This directly addresses the failing monotonicity test.
    """
    n_bar = float(bose_einstein(np.array([T]), omega=OMEGA)[0])
    rc = float(critical_squeezing(n_bar))

    rs = np.linspace(rc + 0.1, rc + 1.2, 15)
    ENs = [float(log_negativity(r, n_bar)) for r in rs]

    for i in range(len(ENs) - 1):
        assert ENs[i + 1] > ENs[i] - 1e-10, (
            f"T={T}: E_N not monotone at i={i}: "
            f"E_N(r={rs[i]:.4f})={ENs[i]:.6f} ≥ E_N(r={rs[i+1]:.4f})={ENs[i+1]:.6f}"
        )


# ──────────────────────────────────────────
# 3. Symplectic eigenvalue (Simon criterion)
# ──────────────────────────────────────────

@pytest.mark.parametrize("T", [0.1, 0.5, 1.0, 2.0])
def test_nu_minus_below_half_iff_entangled(T):
    """
    Simon criterion: ν₋ < ½ ↔ entangled (PPT violated).
    """
    n_bar = float(bose_einstein(np.array([T]), omega=OMEGA)[0])
    rc = float(critical_squeezing(n_bar))

    # Exactly at threshold: ν₋ should equal 0.5 within numerical precision
    nu_at_threshold = float(min_symplectic_eigenvalue(rc, n_bar))
    assert abs(nu_at_threshold - 0.5) < 1e-6, (
        f"At threshold, ν₋={nu_at_threshold:.8f} should equal 0.5 exactly"
    )

    # Above threshold: ν₋ < 0.5 → entangled
    r_above = rc + 0.2
    nu_above = float(min_symplectic_eigenvalue(r_above, n_bar))
    assert nu_above < 0.5, f"T={T}: expected ν₋<0.5 above threshold, got {nu_above}"

    # Below threshold: ν₋ > 0.5 → separable
    r_below = 0.5 * rc
    nu_below = float(min_symplectic_eigenvalue(r_below, n_bar))
    assert nu_below > 0.5, f"T={T}: expected ν₋>0.5 below threshold, got {nu_below}"


# ──────────────────────────────────────────
# 4. Zero temperature limit
# ──────────────────────────────────────────

def test_zero_temperature_limit():
    """
    At T=0: n̄=0, r_c=0, any r>0 gives E_N>0.
    """
    T_near_zero = 1e-9
    n_bar = float(bose_einstein(np.array([T_near_zero]), omega=OMEGA)[0])
    rc = float(critical_squeezing(n_bar))

    assert rc < 1e-4, f"r_c at T≈0 should be ≈0, got {rc}"
    EN = float(log_negativity(0.5, n_bar))
    assert EN > 0.5, f"E_N at T≈0, r=0.5 should be large, got {EN}"


# ──────────────────────────────────────────
# 5. Threshold formula self-consistency
# ──────────────────────────────────────────

@pytest.mark.parametrize("T", np.linspace(0.05, 4.0, 10).tolist())
def test_critical_squeezing_formula(T):
    """
    Theorem 4.3.1: r_c = ½·ln(2n̄+1) must satisfy ν₋(r_c, n̄) = ½ exactly.
    """
    n_bar = float(bose_einstein(np.array([T]), omega=OMEGA)[0])
    rc = float(critical_squeezing(n_bar))
    nu = float(min_symplectic_eigenvalue(rc, n_bar))
    assert abs(nu - 0.5) < 1e-10, (
        f"T={T:.3f}: Theorem 4.3.1 broken — ν₋(r_c)={nu:.12f}, expected 0.5"
    )
