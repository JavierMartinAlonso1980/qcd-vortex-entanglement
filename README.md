# QCD Center Vortex Dynamics: Tensor Network Simulation & Belle II Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18672797.svg)](https://doi.org/10.5281/zenodo.18672797)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **Status:** Research prototype — v1.1.0. Core TMST tests and injection validation are stable. Belle II and IBM Quantum modules are experimental. Please report issues.

Complete implementation of the **collective topological vortex superradiance** theoretical framework in QCD via:
- Center vortex dynamics simulation using **SeeMPS2** (Matrix Product States)
- Massive **Belle II** data classification on HPC clusters (DIRAC grid)
- Experimental validation of squeezed states on **IBM Quantum System One**

## 📋 Description

This repository implements the algorithms described in:
- *Topological Vortex Superradiance and Geometric EPR Bridges*
- *Entanglement Dominance in Zero-Temperature Limit*
- *Belle II Fermionic Bulk-Boundary Algorithm Adaptation*

### Key Features

✅ **MPS simulation with up to 128 qubits** using SeeMPS2
✅ **Automatic MVC threshold detection** (Morphology of Vacuum Condensates)
✅ **Parallel submission to Belle II DIRAC grid** (gbasf2)
✅ **IBM quantum hardware validation** with error correction
✅ **Automatic DOI via Zenodo** for reproducibility
✅ **Theorem 4.3.1 analytic threshold** — importable module with phase-transition event logging *(v1.1.0)*
✅ **Stable phase diagram visualization API** — `compute_phase_diagram` / `plot_phase_diagram` *(v1.1.0)*
✅ **Injection-style signal/noise validation** — TMST + Qiskit 2-qubit Bell tests *(v1.1.0)*

## 🚀 Installation

### 1. Clone repository

```bash
git clone https://github.com/JavierMartinAlonso1980/qcd-vortex-entanglement.git
cd qcd-vortex-entanglement
```

### 2. Create conda environment

```bash
conda env create -f environment.yml
conda activate qcd-vortex
```

### 3. Install dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development + testing
pip install -r requirements-dev.txt

# IBM Quantum (optional — needed for Qiskit injection tests)
pip install -r requirements-quantum.txt

# Everything at once
pip install -r requirements-complete.txt
```

### 4. Configure Belle II (optional)

See detailed documentation in [`docs/BELLE2_SETUP.md`](docs/BELLE2_SETUP.md)

### 5. Configure IBM Quantum (optional)

```bash
export QISKIT_IBM_TOKEN='your_token_here'
```

## 💻 Quick Start

### Simulate Vortex Dynamics

```python
from src.seemps_vortex import CenterVortexMPS

# Initialize 128-vortex system
vortex_system = CenterVortexMPS(N_sites=128, chi_max=64)

# Prepare collective squeezed state
psi = vortex_system.initialize_collective_mode(r_squeeze=1.2, n_thermal=0.1)

# Evolve under Lindblad
trajectory = vortex_system.evolve_lindblad(psi, T_temp=0.2, gamma_loss=0.05)

# Detect confinement
is_confined, S_E, K = vortex_system.compute_mvc_threshold(trajectory[-1], rho_local=1.5)
print(f"Confined: {is_confined}, Entropy: {S_E:.3f}")
```

### Compute TMST Phase Diagram (Theorem 4.3.1)

```python
from src.seemps_vortex import compute_phase_diagram, plot_phase_diagram

# Compute and plot entanglement vs (T, r)
data = compute_phase_diagram(T_range=(0.01, 5.0), r_range=(0.0, 2.0))
plot_phase_diagram(data, save_path="phase_diagram.png")
```

### Run Injection Validation Test

```python
from src.ibm_validation import run_injection_test

# Bell state vs random noise — validates log-negativity separation
results = run_injection_test(shots=2000)
print(results["status"])
```

### Submit Belle II Job to DIRAC Grid

```python
from src.belle2_analysis import BelleIIGridAnalysis

analyzer = BelleIIGridAnalysis("tau_entanglement_2026")
job_id = analyzer.submit_tau_entanglement_job(
    steering_file="steering_tau.py",
    input_dataset="/belle/MC/.../mdst/*.root",
    n_jobs=5000
)
status = analyzer.monitor_jobs(job_id)
```

### Validate on IBM Quantum Hardware

```python
from src.ibm_validation import SqueezeStateValidator

validator = SqueezeStateValidator(backend_name="ibm_sherbrooke")

results = validator.validate_entanglement_threshold(
    r_values=np.linspace(0.1, 1.5, 10),
    T_values=[0.2, 0.8, 1.5],
    n_shots=8000
)
```

## 🧪 Testing

```bash
# Core tests (no Qiskit required)
pytest test/ -v

# Including Qiskit injection tests
pip install -r requirements-quantum.txt
pytest test/ -v

# With coverage report
pytest test/ --cov=src --cov-report=html

# Skip quantum tests
pytest test/ --ignore=test/test_injection_qiskit.py
```

See [`test/README.md`](test/README.md) for full test documentation.

## 📊 Example Notebooks

| Notebook | Description |
|---|---|
| [`01_vortex_dynamics_tutorial.ipynb`](notebooks/01_vortex_dynamics_tutorial.ipynb) | Complete MPS simulation tutorial |
| [`02_belle2_workflow.ipynb`](notebooks/02_belle2_workflow.ipynb) | Belle II grid workflow |
| [`03_ibm_quantum_validation.ipynb`](notebooks/03_ibm_quantum_validation.ipynb) | Quantum hardware experiments |

## 🏗️ Architecture

```
qcd-vortex-entanglement/
├── src/
│   ├── seemps_vortex/
│   │   ├── center_vortex.py         # Core MPS vortex dynamics
│   │   ├── collective_squeezing.py  # Two-mode squeezing operators
│   │   ├── mvc_threshold.py         # MVC threshold & exceptional points
│   │   ├── entanglement_detection.py# Entanglement measures
│   │   ├── tmst_threshold.py        # Theorem 4.3.1 + phase events (v1.1.0)
│   │   ├── phase_diagram.py         # Phase diagram visualization API (v1.1.0)
│   │   └── __init__.py
│   ├── belle2_analysis/             # Belle II + HPC pipeline
│   └── ibm_validation/
│       ├── squeezed_state_prep.py   # TMST circuit preparation
│       ├── quantum_verification.py  # Hardware verification protocols
│       ├── injection_tests.py       # Signal/noise injection tests (v1.1.0)
│       └── __init__.py
├── test/                            # Unit tests (pytest)
│   ├── conftest.py
│   ├── validation_tools.py          # Shared injection helpers (v1.1.0)
│   ├── test_vortex_mps.py
│   ├── test_entanglement.py
│   ├── test_tmst_injection.py       # TMST injection tests (v1.1.0)
│   ├── test_injection_qiskit.py     # Qiskit injection tests (v1.1.0)
│   └── README.md
├── scripts/
│   ├── hpc_submit_belle2.sh         # SLURM script for HPC
│   └── batch_mps_simulation.py
├── notebooks/                       # Jupyter tutorials
├── docs/                            # Technical documentation
├── requirements.txt
├── requirements-dev.txt
├── requirements-quantum.txt         # IBM Quantum optional deps (v1.1.0)
└── requirements-complete.txt
```

## 📈 Results

### Theorem 4.3.1 Validation (Entanglement Dominance)

Accuracy on IBM Quantum Hardware: **87.5%** (8 r values × 3 temperatures)

### Belle II Data Classification

- **55 computing sites** across 15 countries
- Throughput: **~70 kHepSPEC** at peak
- Classified events: **>6 billion** (0.8 ab⁻¹ equivalent)

## 📚 Citation

If you use this code, please cite:

```bibtex
@software{qcd_vortex_2026,
  author = {Javier Manuel Martín Alonso},
  title = {QCD Center Vortex Dynamics: Tensor Network Simulation \& Belle II Analysis},
  year = {2026},
  publisher = {Zenodo},
  version = {1.1.0},
  doi = {10.5281/zenodo.18672797},
  url = {https://github.com/JavierMartinAlonso1980/qcd-vortex-entanglement}
}
```

## 🙏 Acknowledgments

### Computational Frameworks

This project uses SeeMPS for matrix product state algorithms:

- García-Molina, P., Rodríguez-Aldavero, J.J., Gidi, J., & García-Ripoll, J.J. (2026).
  "SeeMPS: A Python-based Matrix Product State and Tensor Train Library".
  arXiv:2601.16734 [quant-ph]. https://arxiv.org/abs/2601.16734

The quantum-inspired algorithms are based on:

- García-Ripoll, J.J. (2021).
  "Quantum-inspired algorithms for multivariate analysis: from interpolation to partial differential equations".
  Quantum, 5, 431. https://doi.org/10.22331/q-2021-04-15-431

**Repository:** https://github.com/juanjosegarciaripoll/seemps2
**License:** MIT

### Software Libraries

See [requirements.txt](requirements.txt) for complete list of dependencies.

Key libraries:
- SeeMPS: Matrix Product States (García-Ripoll)
- NumPy: Array computing (Harris et al., 2020)
- SciPy: Scientific computing (Virtanen et al., 2020)
- Qiskit: Quantum computing framework (IBM Quantum)

## 🤝 Contributing

Contributions are welcome! Please:
- Open an [Issue](../../issues) for bugs or features
- Fork and submit a Pull Request for code changes
- Follow PEP 8 and add tests when applicable

Questions? Contact [jmma@movistar.es](mailto:jmma@movistar.es)

## 📄 License

MIT License — see [`LICENSE`](LICENSE)

## 🔗 Links

- **Zenodo DOI**: https://doi.org/10.5281/zenodo.18672797
- **Belle II Computing**: https://www.belle2.org/computing/
- **IBM Quantum**: https://quantum.ibm.com/
- **SeeMPS2 GitHub**: https://github.com/juanjosegarciaripoll/seemps2

## 📧 Contact

For technical questions, open an [Issue](https://github.com/JavierMartinAlonso1980/qcd-vortex-entanglement/issues).
