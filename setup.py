#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QCD Vortex Entanglement Analysis Framework
==========================================

A comprehensive framework for analyzing center vortex dynamics and entanglement 
in QCD using Matrix Product States (MPS), Belle II data analysis, and IBM 
Quantum hardware validation.

Author: Your Name
License: MIT
Python: >=3.10
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py"""
    init_file = Path('src/seemps_vortex/__init__.py')
    if init_file.exists():
        content = init_file.read_text()
        match = re.search(r"^__version__\s*=\s*['"]([^'"]+)['"]", content, re.M)
        if match:
            return match.group(1)
    return '1.0.0'

# Read long description from README
readme_file = Path('README.md')
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_file = Path(filename)
    if req_file.exists():
        with open(req_file, 'r') as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith('#')]
    return []

# Core dependencies
install_requires = [
    'numpy>=1.24.0',
    'scipy>=1.10.0',
    'matplotlib>=3.7.0',
    'pandas>=2.0.0',
    'h5py>=3.8.0',
    'scikit-learn>=1.3.0',
    'xgboost>=2.0.0',
    'tqdm>=4.65.0',
    'pyyaml>=6.0',
    'joblib>=1.3.0',
]

# Optional dependencies for specific features
extras_require = {
    # Belle II analysis
    'belle2': [
        'uproot>=5.0.0',
        'awkward>=2.0.0',
        'particle>=0.23.0',
        'iminuit>=2.24.0',
        'boost-histogram>=1.4.0',
    ],

    # IBM Quantum validation
    'quantum': [
        'qiskit>=1.0.0',
        'qiskit-ibm-runtime>=0.19.0',
        'qiskit-aer>=0.13.0',
    ],

    # Tensor network library (core requirement, but optional install)
    'seemps': [
        'seemps2>=1.0.0',
    ],

    # GPU acceleration
    'gpu': [
        'cupy-cuda11x>=12.0.0',  # For CUDA 11.x
        # Use 'cupy-cuda12x' for CUDA 12.x
    ],

    # MPI for parallel computing
    'mpi': [
        'mpi4py>=3.1.0',
    ],

    # Visualization
    'viz': [
        'plotly>=5.17.0',
        'seaborn>=0.13.0',
        'ipywidgets>=8.1.0',
    ],

    # Development tools
    'dev': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'pytest-xdist>=3.3.0',
        'pytest-timeout>=2.1.0',
        'black>=23.7.0',
        'flake8>=6.1.0',
        'mypy>=1.5.0',
        'isort>=5.12.0',
        'pylint>=2.17.0',
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
        'nbsphinx>=0.9.3',
    ],

    # Documentation
    'docs': [
        'sphinx>=7.1.0',
        'sphinx-rtd-theme>=1.3.0',
        'nbsphinx>=0.9.3',
        'myst-parser>=2.0.0',
        'sphinx-autodoc-typehints>=1.24.0',
    ],

    # Jupyter notebooks
    'jupyter': [
        'jupyterlab>=4.0.0',
        'notebook>=7.0.0',
        'ipykernel>=6.25.0',
        'ipywidgets>=8.1.0',
    ],
}

# Add 'all' option to install everything
extras_require['all'] = list(set(sum(extras_require.values(), [])))

# Add 'complete' option (all except GPU and MPI)
extras_require['complete'] = list(set(
    extras_require['belle2'] + 
    extras_require['quantum'] + 
    extras_require['seemps'] +
    extras_require['viz'] +
    extras_require['jupyter']
))

# Python version requirement
python_requires = '>=3.10'

setup(
    # Basic information
    name='qcd-vortex-entanglement',
    version=get_version(),
    description='QCD Vortex Entanglement Analysis Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # Author information
    author='Your Name',
    author_email='your.email@institution.edu',

    # URLs
    url='https://github.com/yourusername/qcd-vortex-entanglement',
    project_urls={
        'Documentation': 'https://qcd-vortex-entanglement.readthedocs.io/',
        'Source': 'https://github.com/yourusername/qcd-vortex-entanglement',
        'Tracker': 'https://github.com/yourusername/qcd-vortex-entanglement/issues',
    },

    # License
    license='MIT',

    # Package discovery
    packages=find_packages(where='src'),
    package_dir={'': 'src'},

    # Include non-Python files
    include_package_data=True,
    package_data={
        'seemps_vortex': ['data/*.yaml', 'configs/*.json'],
        'belle2_analysis': ['steering/*.py', 'configs/*.yaml'],
        'ibm_validation': ['circuits/*.qpy', 'configs/*.json'],
    },

    # Dependencies
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=python_requires,

    # Entry points (command-line scripts)
    entry_points={
        'console_scripts': [
            'vortex-simulate=seemps_vortex.cli:main',
            'belle2-analyze=belle2_analysis.cli:main',
            'ibm-validate=ibm_validation.cli:main',
        ],
    },

    # Classifiers for PyPI
    classifiers=[
        # Development status
        'Development Status :: 4 - Beta',

        # Intended audience
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        # Topics
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # License
        'License :: OSI Approved :: MIT License',

        # Python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',

        # Operating systems
        'Operating System :: OS Independent',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',

        # Other
        'Natural Language :: English',
        'Framework :: Jupyter',
    ],

    # Keywords for searching
    keywords=[
        'qcd',
        'vortex',
        'entanglement',
        'matrix-product-states',
        'tensor-networks',
        'belle2',
        'quantum-computing',
        'high-energy-physics',
        'quantum-information',
        'confinement',
    ],

    # Zip safety
    zip_safe=False,

    # Test suite
    test_suite='tests',
    tests_require=[
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
    ],
)
