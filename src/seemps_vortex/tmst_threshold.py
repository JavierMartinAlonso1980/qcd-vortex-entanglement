# src/seemps_vortex/tmst_threshold.py
"""
TMST Entanglement Threshold — Theorem 4.3.1
============================================
Extracted from mainAlice.txt for use as importable module across tests,
notebooks, and the Belle II / IBM validation pipeline.

Reference:
    Martín Alonso, J. M. (2026). "Entanglement Dominance in the Zero-Temperature Limit".
    Zenodo. https://doi.org/10.5281/zenodo.18353640
"""

from __future__ import annotations
import dataclasses
import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

# Natural units: ħ = k_B = 1
OMEGA_DEFAULT: float = 1.0

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core physics functions (from mainAlice.txt)
# ─────────────────────────────────────────────

def bose_einstein(
    temp: npt.ArrayLike,
    omega: float = OMEGA_DEFAULT,
) -> np.ndarray:
    """
    Mean thermal occupation number n̄(T).
    n̄ = 1 / (exp(ω/T) - 1)

    Returns 0 at T=0 without raising division-by-zero warnings.
    """
    temp = np.atleast_1d(np.asarray(temp, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        n = 1.0 / (np.exp(omega / temp) - 1.0)
    n[temp == 0.0] = 0.0
    return n


def critical_squeezing(n_bar: npt.ArrayLike) -> np.ndarray:
    """
    Analytic threshold from Theorem 4.3.1:
        r_c(n̄) = ½ · ln(2·n̄ + 1)

    At this squeezing the minimum symplectic eigenvalue ν₋ = ½,
    i.e. the system sits exactly on the separability boundary.
    """
    n_bar = np.asarray(n_bar, dtype=float)
    return 0.5 * np.log(2.0 * n_bar + 1.0)


def min_symplectic_eigenvalue(
    r: npt.ArrayLike,
    n_bar: npt.ArrayLike,
) -> np.ndarray:
    """
    Minimum symplectic eigenvalue of the partially-transposed CM:
        ν₋(r, n̄) = (n̄ + ½) · exp(-2r)

    Entanglement ↔ ν₋ < ½  (Simon criterion / PPT).
    """
    r = np.asarray(r, dtype=float)
    n_bar = np.asarray(n_bar, dtype=float)
    return (n_bar + 0.5) * np.exp(-2.0 * r)


def log_negativity(
    r: npt.ArrayLike,
    n_bar: npt.ArrayLike,
) -> np.ndarray:
    """
    Logarithmic Negativity E_N for a TMST:
        E_N = max(0, -log₂(2·ν₋))

    Returns 0 in the separable regime (ν₋ ≥ ½).
    """
    nu_minus = min_symplectic_eigenvalue(r, n_bar)
    val = -np.log2(2.0 * nu_minus)
    return np.maximum(0.0, val)


# ─────────────────────────────────────────────
# Phase-transition event dataclass (point 4.3)
# ─────────────────────────────────────────────

@dataclasses.dataclass
class TMSTPhaseEvent:
    """
    Structured log event emitted when the system crosses the
    entanglement threshold (Theorem 4.3.1).

    Attributes
    ----------
    r : float
        Squeezing parameter at the event point.
    T : float
        Temperature at the event point.
    n_bar : float
        Thermal occupation number.
    nu_minus : float
        Minimum symplectic eigenvalue (< 0.5 → entangled).
    log_negativity : float
        E_N at this point.
    entangled : bool
        True if ν₋ < 0.5 (EPR bridge active).
    delta_S_local : Optional[float]
        Local entropy change, if computed (for TDHCF extension).
    """
    r: float
    T: float
    n_bar: float
    nu_minus: float
    log_negativity: float
    entangled: bool
    delta_S_local: Optional[float] = None

    @property
    def event_type(self) -> str:
        return "ENTANGLEMENT_DOMINANT_ON" if self.entangled else "NOISE_DOMINATED"


def scan_phase_diagram(
    r_vals: npt.ArrayLike,
    T_vals: npt.ArrayLike,
    omega: float = OMEGA_DEFAULT,
    log_events: bool = True,
) -> tuple[np.ndarray, list[TMSTPhaseEvent]]:
    """
    Scan the (r, T) phase space and return:
    - E_N matrix of shape (len(r_vals), len(T_vals))
    - List of TMSTPhaseEvent for every (r, T) point

    Useful for building the phase diagram (point 4.4) and
    for structured logging (point 4.3).
    """
    r_vals = np.asarray(r_vals, dtype=float)
    T_vals = np.asarray(T_vals, dtype=float)
    r_grid, T_grid = np.meshgrid(r_vals, T_vals, indexing="ij")

    n_bar_grid = bose_einstein(T_grid, omega=omega)
    nu_minus_grid = min_symplectic_eigenvalue(r_grid, n_bar_grid)
    EN_grid = log_negativity(r_grid, n_bar_grid)

    events: list[TMSTPhaseEvent] = []
    for i, r in enumerate(r_vals):
        for j, T in enumerate(T_vals):
            ev = TMSTPhaseEvent(
                r=float(r),
                T=float(T),
                n_bar=float(n_bar_grid[i, j]),
                nu_minus=float(nu_minus_grid[i, j]),
                log_negativity=float(EN_grid[i, j]),
                entangled=bool(nu_minus_grid[i, j] < 0.5),
            )
            events.append(ev)
            if log_events:
                logger.debug("[%s] r=%.4f T=%.4f ν₋=%.4f E_N=%.4f",
                             ev.event_type, r, T, ev.nu_minus, ev.log_negativity)

    return EN_grid, events
