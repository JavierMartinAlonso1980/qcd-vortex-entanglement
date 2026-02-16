# Installation Guide - setup.py

Complete guide for installing the QCD Vortex Entanglement package.

## Files Created

1. **setup.py** - Main installation script (legacy)
2. **setup.cfg** - Additional configuration
3. **pyproject.toml** - Modern Python packaging (PEP 517/518)
4. **MANIFEST.in** - Include/exclude files in distribution
5. **requirements.txt** - Core dependencies
6. **requirements-dev.txt** - Development dependencies
7. **requirements-complete.txt** - All optional dependencies

## Installation Methods

### Method 1: Development Installation (Recommended)

For active development with editable installation:

```bash
# Basic installation
pip install -e .

# With Belle II support
pip install -e ".[belle2]"

# With IBM Quantum support
pip install -e ".[quantum]"

# Complete installation (all features)
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"
```

### Method 2: Standard Installation

```bash
# From source directory
pip install .

# With specific features
pip install ".[belle2,quantum,viz]"
```

### Method 3: Using Requirements Files

```bash
# Core dependencies only
pip install -r requirements.txt

# Development environment
pip install -r requirements-dev.txt

# Complete installation
pip install -r requirements-complete.txt
```

### Method 4: From Git Repository

```bash
# Latest version
pip install git+https://github.com/yourusername/qcd-vortex-entanglement.git

# Specific branch
pip install git+https://github.com/yourusername/qcd-vortex-entanglement.git@develop

# With extras
pip install "qcd-vortex-entanglement[belle2,quantum] @ git+https://github.com/yourusername/qcd-vortex-entanglement.git"
```

## Optional Dependencies

### Belle II Analysis (`belle2`)

```bash
pip install ".[belle2]"
```

Includes:
- uproot (ROOT file I/O)
- awkward (Columnar data)
- particle (PDG particle data)
- iminuit (Fitting)

### IBM Quantum (`quantum`)

```bash
pip install ".[quantum]"
```

Includes:
- qiskit (Quantum circuits)
- qiskit-ibm-runtime (IBM Quantum access)
- qiskit-aer (Quantum simulators)

### Visualization (`viz`)

```bash
pip install ".[viz]"
```

Includes:
- plotly (Interactive plots)
- seaborn (Statistical visualization)
- ipywidgets (Jupyter widgets)

### GPU Acceleration (`gpu`)

```bash
# For CUDA 11.x
pip install ".[gpu]"

# For CUDA 12.x
pip install cupy-cuda12x
```

### MPI Support (`mpi`)

```bash
# Requires system MPI installation first
pip install ".[mpi]"
```

### Development Tools (`dev`)

```bash
pip install ".[dev]"
```

Includes:
- pytest (Testing)
- black (Code formatting)
- flake8 (Linting)
- mypy (Type checking)
- sphinx (Documentation)

### Complete Installation (`all`)

```bash
pip install ".[all]"
```

Includes all optional dependencies.

## Verification

After installation, verify with:

```bash
# Check installation
python -c "import seemps_vortex; print(seemps_vortex.__version__)"

# Run tests
pytest tests/ -v

# Check command-line tools
vortex-simulate --help
belle2-analyze --help
ibm-validate --help
```

## Building Distribution

### Source Distribution

```bash
python setup.py sdist
```

Output: `dist/qcd-vortex-entanglement-X.Y.Z.tar.gz`

### Wheel Distribution

```bash
python setup.py bdist_wheel
```

Output: `dist/qcd_vortex_entanglement-X.Y.Z-py3-none-any.whl`

### Using build (Recommended)

```bash
pip install build
python -m build
```

Creates both sdist and wheel in `dist/`

## Uploading to PyPI

### Test PyPI

```bash
pip install twine
twine upload --repository testpypi dist/*
```

### Production PyPI

```bash
twine upload dist/*
```

## Uninstallation

```bash
pip uninstall qcd-vortex-entanglement
```

## Troubleshooting

### Issue: Module not found after installation

```bash
# Reinstall in editable mode
pip install -e .

# Or check Python path
python -c "import sys; print(sys.path)"
```

### Issue: Dependency conflicts

```bash
# Create fresh environment
conda create -n qcd-vortex python=3.10
conda activate qcd-vortex
pip install -e .
```

### Issue: Build fails

```bash
# Upgrade build tools
pip install --upgrade pip setuptools wheel

# Clean previous builds
rm -rf build/ dist/ *.egg-info
python setup.py clean --all
```

### Issue: Entry points not working

```bash
# Reinstall package
pip uninstall qcd-vortex-entanglement
pip install -e .

# Check entry points
pip show qcd-vortex-entanglement
```

## Environment Variables

Set these for optimal installation:

```bash
# Python unbuffered output
export PYTHONUNBUFFERED=1

# Pip cache directory
export PIP_CACHE_DIR=~/.cache/pip

# Compiler flags (if needed)
export CFLAGS="-O3"
export CXXFLAGS="-O3"
```

## Platform-Specific Notes

### Linux

All dependencies install cleanly via pip.

### macOS

```bash
# May need Xcode command line tools
xcode-select --install

# Some packages require Homebrew
brew install gcc cmake
```

### Windows

Use WSL2 for best compatibility:

```bash
wsl --install -d Ubuntu-22.04
# Then follow Linux instructions
```

## Docker Installation

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -e ".[all]"

CMD ["bash"]
```

Build and run:

```bash
docker build -t qcd-vortex .
docker run -it qcd-vortex
```

## Conda Installation

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Or manually
conda create -n qcd-vortex python=3.10
conda activate qcd-vortex
pip install -e .
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Install package
  run: |
    pip install -e ".[dev]"

- name: Run tests
  run: |
    pytest tests/ --cov=src
```

### GitLab CI

```yaml
test:
  script:
    - pip install -e ".[dev]"
    - pytest tests/ --cov=src
```

---

**Last Updated:** February 16, 2026
