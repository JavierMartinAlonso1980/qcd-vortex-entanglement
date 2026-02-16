# Installation Guide

Complete installation guide for the QCD Vortex Entanglement project.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Environment Setup](#python-environment-setup)
3. [Core Dependencies](#core-dependencies)
4. [Optional Components](#optional-components)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

- **CPU**: Modern x86-64 processor (minimum 4 cores, recommended 16+)
- **RAM**: Minimum 16 GB, recommended 64 GB for large MPS simulations
- **Storage**: 50 GB free space (100+ GB for Belle II data)
- **GPU** (optional): NVIDIA GPU with CUDA 11.8+ for accelerated tensor operations

### Operating Systems

Tested on:
- **Linux**: Ubuntu 22.04 LTS, Rocky Linux 9 (recommended for HPC)
- **macOS**: 12.0+ (Apple Silicon and Intel)
- **Windows**: 10/11 with WSL2 (Linux subsystem required)

### Software Prerequisites

- **Python**: 3.10 or 3.11 (3.12+ not fully tested)
- **Git**: 2.30+
- **GCC/Clang**: For compiling native extensions
- **CMake**: 3.20+ (for SeeMPS compilation)

---

## Python Environment Setup

### Option 1: Conda (Recommended)

```bash
# Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create -n qcd-vortex python=3.10
conda activate qcd-vortex
```

### Option 2: venv

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Option 3: Poetry (Development)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
poetry shell
```

---

## Core Dependencies

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/qcd-vortex-entanglement.git
cd qcd-vortex-entanglement
```

### Step 2: Install Core Packages

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install from requirements
pip install -r requirements.txt
```

### Step 3: Install SeeMPS

SeeMPS is the core tensor network library.

#### Option A: From PyPI (Stable)

```bash
pip install seemps2
```

#### Option B: From Source (Latest)

```bash
git clone https://github.com/juanjosegarciaripoll/seemps2.git
cd seemps2
pip install -e .
cd ..
```

#### Verify SeeMPS Installation

```python
python -c "import seemps; print(seemps.__version__)"
# Expected output: 1.0.x or higher
```

### Step 4: Install Project Package

```bash
# Install in editable mode
pip install -e .
```

---

## Optional Components

### Belle II Analysis (ROOT, uproot)

For Belle II data analysis:

```bash
# Install ROOT (via conda - easiest method)
conda install -c conda-forge root

# Or install uproot for ROOT file I/O without full ROOT
pip install uproot awkward

# Additional Belle II tools
pip install particle  # PDG particle data
pip install iminuit   # Fitting
```

**Verify ROOT:**

```bash
root --version
# Expected: ROOT Version: 6.28/00 or higher
```

### IBM Quantum Access (Qiskit)

For IBM Quantum hardware validation:

```bash
# Install Qiskit Runtime
pip install qiskit-ibm-runtime>=0.19.0

# Verify installation
python -c "import qiskit; print(qiskit.__version__)"
# Expected: 1.0.0+
```

**Configure IBM Quantum account:**

```bash
# Set token as environment variable
export QISKIT_IBM_TOKEN="your_token_here"

# Or save permanently
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"
```

See [IBM_QUANTUM_ACCESS.md](IBM_QUANTUM_ACCESS.md) for detailed setup.

### MPI for Parallel Computing

For large-scale batch simulations:

```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev openmpi-bin

# CentOS/RHEL
sudo yum install openmpi openmpi-devel

# macOS (via Homebrew)
brew install open-mpi

# Install mpi4py
pip install mpi4py
```

**Verify MPI:**

```bash
mpirun --version
python -c "from mpi4py import MPI; print(f'MPI rank: {MPI.COMM_WORLD.Get_rank()}')"
```

### CUDA Support (GPU Acceleration)

For GPU-accelerated tensor operations:

```bash
# Install CuPy (CUDA 11.8 example)
pip install cupy-cuda11x

# Or for specific CUDA version
pip install cupy-cuda12x  # CUDA 12.x

# Verify GPU access
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

### Visualization Tools

```bash
# Enhanced plotting
pip install plotly seaborn

# Jupyter support
pip install jupyterlab ipywidgets

# LaTeX rendering (optional, for publication-quality plots)
# Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

---

## Verification

### Run Test Suite

```bash
# Basic tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Parallel execution
pytest tests/ -n auto
```

### Run Example Scripts

```bash
# Test MPS simulation
python examples/basic_vortex_simulation.py

# Test Belle II analysis (with synthetic data)
python examples/belle2_classification_demo.py

# Test IBM Quantum (requires account)
python examples/ibm_tmst_validation.py
```

### Verify Installation Script

```bash
python scripts/verify_installation.py
```

Expected output:

```
✓ Python version: 3.10.x
✓ NumPy: 1.26.x
✓ SciPy: 1.11.x
✓ SeeMPS: 1.0.x
✓ Qiskit: 1.0.x
✓ Project modules: OK
✓ All core dependencies installed
```

---

## Troubleshooting

### Issue: SeeMPS Installation Fails

**Symptoms:**
```
error: command 'gcc' failed
```

**Solution:**

```bash
# Install build tools
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Then reinstall
pip install seemps2 --no-cache-dir
```

### Issue: ROOT Import Error

**Symptoms:**
```python
ImportError: libCore.so: cannot open shared object file
```

**Solution:**

```bash
# Source ROOT environment
source /path/to/root/bin/thisroot.sh

# Or add to ~/.bashrc
echo "source /opt/root/bin/thisroot.sh" >> ~/.bashrc
```

**Alternative:** Use uproot instead of ROOT:

```bash
pip install uproot
```

### Issue: Qiskit Authentication Fails

**Symptoms:**
```
IBMAccountError: No active account
```

**Solution:**

```bash
# Check token is set
echo $QISKIT_IBM_TOKEN

# Re-save account
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN', overwrite=True)"
```

### Issue: MPI Import Error

**Symptoms:**
```python
ImportError: libmpi.so: cannot open shared object file
```

**Solution:**

```bash
# Load MPI module (on HPC clusters)
module load openmpi

# Or add to environment
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH
```

### Issue: Out of Memory (MPS Simulations)

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solution:**

1. Reduce bond dimension `chi_max`
2. Use checkpointing to disk
3. Enable swap space (not recommended for performance)

```python
# In your script
vortex_system = CenterVortexMPS(
    N_sites=128,
    chi_max=32  # Reduce from 64
)
```

### Issue: Slow MPS Operations

**Solution:**

1. Install MKL-optimized NumPy:

```bash
pip uninstall numpy
pip install numpy[mkl]
```

2. Enable parallel linear algebra:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

3. Use tensor slicing for large systems:

```python
# Break into smaller chunks
for chunk in range(0, N_sites, 32):
    process_chunk(psi, chunk, chunk+32)
```

### Issue: pytest Not Found

**Solution:**

```bash
pip install pytest pytest-cov pytest-xdist
```

---

## Platform-Specific Notes

### Ubuntu 22.04 LTS

All packages install cleanly via apt and pip. Recommended for production.

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential gfortran \
    libopenblas-dev liblapack-dev \
    git cmake
```

### macOS

Use Homebrew for system dependencies:

```bash
brew install python@3.10 gcc cmake openblas lapack

# For Apple Silicon (M1/M2/M3)
# Some packages may need Rosetta 2
softwareupdate --install-rosetta
```

### Windows (WSL2)

Use Windows Subsystem for Linux:

```bash
# In PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04

# Then follow Ubuntu instructions above
```

### HPC Clusters (SLURM)

Most HPC systems use environment modules:

```bash
module load python/3.10
module load gcc/11
module load openmpi/4.1
module load cuda/11.8  # if available

# Create virtual environment in home directory
python -m venv ~/.venvs/qcd-vortex
source ~/.venvs/qcd-vortex/bin/activate

pip install -r requirements.txt
```

---

## Environment Variables

Recommended environment variables for optimal performance:

```bash
# Add to ~/.bashrc or ~/.zshrc

# OpenMP threads
export OMP_NUM_THREADS=8

# MKL threads (if using Intel MKL)
export MKL_NUM_THREADS=8

# IBM Quantum token
export QISKIT_IBM_TOKEN="your_token_here"

# Belle II grid (if using DIRAC)
export DIRAC_SETUP="/cvmfs/belle.cern.ch/tools/dirac/pro/bashrc"

# Python unbuffered output (for logging)
export PYTHONUNBUFFERED=1

# Disable TensorFlow warnings (if using)
export TF_CPP_MIN_LOG_LEVEL=2
```

---

## Next Steps

After successful installation:

1. **Run Tutorial Notebooks:**
   ```bash
   cd notebooks
   jupyter lab
   ```

2. **Read Documentation:**
   - [BELLE2_SETUP.md](BELLE2_SETUP.md) - Belle II grid computing setup
   - [IBM_QUANTUM_ACCESS.md](IBM_QUANTUM_ACCESS.md) - IBM Quantum configuration

3. **Join Community:**
   - GitHub Issues: Report bugs or request features
   - Discussions: Ask questions and share results

4. **Contribute:**
   - See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines

---

## Getting Help

- **Documentation**: https://github.com/yourusername/qcd-vortex-entanglement/wiki
- **Issues**: https://github.com/yourusername/qcd-vortex-entanglement/issues
- **Email**: your.email@institution.edu

## Citation

If you use this software, please cite:

```bibtex
@software{qcd_vortex_2026,
  author = {Your Name},
  title = {QCD Vortex Entanglement Analysis Framework},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/qcd-vortex-entanglement}
}
```

---

**Last Updated:** February 16, 2026
