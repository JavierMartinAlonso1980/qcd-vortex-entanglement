# test/gaussian/run_toy_test.py
"""
Test rápido: simulación toy Belle II con máxima vorticidad.
Ejecutar con: py run_toy_test.py  (desde la raíz)
              pytest test/gaussian/run_toy_test.py  (desde pytest)
"""
import numpy as np
import sys
from pathlib import Path

# Ruta robusta independiente de desde dónde se ejecute
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from seemps_vortex.tmst_threshold import (
    log_negativity,
    bose_einstein,
    critical_squeezing,
    min_symplectic_eigenvalue,
)

R_MAX_VORTICITY = 1.8
T_LOW = 0.1
OMEGA = 1.0

def _run():
    n_bar = float(bose_einstein(np.array([T_LOW]), omega=OMEGA)[0])
    r_c   = float(critical_squeezing(n_bar))
    EN    = float(log_negativity(R_MAX_VORTICITY, n_bar))
    nu    = float(min_symplectic_eigenvalue(R_MAX_VORTICITY, n_bar))
    return n_bar, r_c, EN, nu

def test_toy_belle2_max_vorticity():
    """Pytest: máxima vorticidad Belle II — Theorem 4.3.1."""
    n_bar, r_c, EN, nu = _run()
    assert EN > 0.5,               f"E_N={EN:.4f} demasiado baja"
    assert R_MAX_VORTICITY > r_c,  f"r no supera r_c={r_c:.6f}"
    assert nu < 0.5,               f"nu={nu:.6f} >= 0.5 (separable)"
    assert EN > 1.0,               f"E_N={EN:.4f} < 1.0 bit esperado"

if __name__ == "__main__":
    n_bar, r_c, EN, nu = _run()
    print("=" * 50)
    print("TOY TEST — Máxima Vorticidad (Belle II)")
    print("=" * 50)
    print(f"  n_bar={n_bar:.6f}  r_c={r_c:.6f}  nu={nu:.6f}  E_N={EN:.6f}")
    test_toy_belle2_max_vorticity()
    print("✅ TODOS LOS ASSERTS PASADOS")
