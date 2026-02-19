# src/seemps_vortex/phase_diagram.py
"""
Phase Diagram Visualization — TMST Entanglement
================================================
Stable plotting API wrapping mainAlice.txt logic.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .tmst_threshold import bose_einstein, critical_squeezing, log_negativity


def compute_phase_diagram(
    T_range: tuple[float, float] = (0.01, 5.0),
    r_range: tuple[float, float] = (0.0, 2.0),
    n_points: int = 500,
    omega: float = 1.0,
) -> dict:
    """
    Compute the TMST phase diagram over a (T, r) grid.

    Returns
    -------
    dict with keys:
        T_grid, r_grid, EN_grid, T_vals, r_crit_line
    """
    T_vals = np.linspace(T_range[0], T_range[1], n_points)
    r_vals = np.linspace(r_range[0], r_range[1], n_points)
    T_grid, r_grid = np.meshgrid(T_vals, r_vals)

    n_bar_grid = bose_einstein(T_grid, omega=omega)
    EN_grid = log_negativity(r_grid, n_bar_grid)

    n_bar_line = bose_einstein(T_vals, omega=omega)
    r_crit_line = critical_squeezing(n_bar_line)

    return {
        "T_grid": T_grid,
        "r_grid": r_grid,
        "EN_grid": EN_grid,
        "T_vals": T_vals,
        "r_crit_line": r_crit_line,
    }


def plot_phase_diagram(
    data: dict | None = None,
    save_path: str | None = "entanglement_phase_diagram.png",
    dpi: int = 300,
    show: bool = True,
    **compute_kwargs,
) -> plt.Figure:
    """
    Plot the entanglement phase diagram with the analytic red threshold line.

    Parameters
    ----------
    data : dict, optional
        Pre-computed output of compute_phase_diagram().
        If None, it will be computed using **compute_kwargs.
    save_path : str, optional
        Where to save the PNG. None to skip saving.
    """
    if data is None:
        data = compute_phase_diagram(**compute_kwargs)

    T_grid = data["T_grid"]
    r_grid = data["r_grid"]
    EN_grid = data["EN_grid"]
    T_vals = data["T_vals"]
    r_crit_line = data["r_crit_line"]

    colors = ["#f0f0f0", "#d1e5f0", "#4393c3", "#2166ac", "#053061"]
    cmap = LinearSegmentedColormap.from_list("entanglement_map", colors, N=100)

    fig, ax = plt.subplots(figsize=(10, 7))

    cf = ax.contourf(T_grid, r_grid, EN_grid, levels=50, cmap=cmap)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"Log-Negativity $E_N$ (Entanglement Bits)", fontsize=12)

    # Analytic threshold — the "red line"
    ax.plot(T_vals, r_crit_line, "r-", linewidth=2.5,
            label=r"Analytic Threshold $r_c(T)$ — Theorem 4.3.1")

    ax.fill_between(T_vals, 0, r_crit_line, color="gray", alpha=0.1, hatch="///")
    ax.text(1.0, 0.2, "NOISE DOMINATED\n(Separable)",
            fontsize=13, color="#555555", ha="center", fontweight="bold")
    ax.text(1.5, 1.55, "ENTANGLEMENT DOMINANT\n(Topological Channel Open)",
            fontsize=13, color="white", ha="center", fontweight="bold")

    ax.set_title(
        "Phase Diagram: Entanglement Dominance in TMST\n"
        "(Theorem 4.3.1 — qcd-vortex-entanglement)",
        fontsize=14,
    )
    ax.set_xlabel(r"Temperature ($k_B T / \hbar \omega$)", fontsize=12)
    ax.set_ylabel(r"Squeezing Parameter $r$", fontsize=12)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(r_grid.min(), r_grid.max())
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right", framealpha=1.0)

    if save_path:
        fig.savefig(save_path, dpi=dpi)
    if show:
        plt.show()
    return fig
