"""
Conftest para tests gaussianos/CV.
Hereda automáticamente test/conftest.py (reset_random_seed,
suppress_warnings, tolerance, temp_directory).
Añade: carga directa de tmst_threshold sin disparar
       seemps_vortex/__init__.py → center_vortex → seemps.
"""
import sys
import importlib.util
from pathlib import Path

# Raíz del proyecto (sube: gaussian/ → test/ → raíz)
_PROJECT_ROOT = Path(__file__).parents[2]


def _load_module_direct(alias: str, rel_path: str):
    """Carga un .py directo sin pasar por __init__.py del paquete."""
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(
        alias, _PROJECT_ROOT / rel_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod          # registrar ANTES de exec para imports circulares
    spec.loader.exec_module(mod)
    return mod


# Pre-carga al arrancar la sesión gaussian
_load_module_direct(
    "tmst_threshold",
    "src/seemps_vortex/tmst_threshold.py",
)
