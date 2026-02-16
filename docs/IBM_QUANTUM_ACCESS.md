# IBM Quantum Access Setup

Complete guide for accessing IBM Quantum hardware and running validation experiments.

## Table of Contents

1. [Overview](#overview)
2. [Account Registration](#account-registration)
3. [Access Tiers](#access-tiers)
4. [Installation](#installation)
5. [Authentication](#authentication)
6. [Backend Selection](#backend-selection)
7. [Running Jobs](#running-jobs)
8. [Best Practices](#best-practices)
9. [Cost Management](#cost-management)
10. [Troubleshooting](#troubleshooting)

---

## Overview

IBM Quantum provides cloud access to real quantum computers:

- **127+ qubit systems** (Eagle r3 processors)
- **Qiskit Runtime** for optimized execution
- **Error mitigation** built-in
- **Global access** via IBM Cloud

**Quantum Systems Available:**
- IBM Sherbrooke (127 qubits)
- IBM Kyiv (127 qubits)
- IBM Brisbane (127 qubits)
- IBM Osaka (127 qubits)

**Requirements:**
- IBM Quantum account (free tier available)
- Python 3.10+
- Qiskit Runtime 0.19+

---

## Account Registration

### Step 1: Create IBM Quantum Account

#### Option A: IBM Quantum Platform (Open Plan - Free)

1. Go to https://quantum.ibm.com/
2. Click "Sign in" → "Create an IBM ID"
3. Verify email
4. Access dashboard

**Free tier includes:**
- Access to 127-qubit systems
- 10 minutes/month quantum time
- Simulators (unlimited)
- Qiskit Runtime access

#### Option B: IBM Cloud (Pay-as-you-go)

1. Go to https://cloud.ibm.com/
2. Create IBM Cloud account
3. Enable Quantum Computing service
4. Link credit card (charged per second of QPU time)

**Advantages:**
- Higher priority queue
- More quantum time
- Additional backends
- Production-grade support

### Step 2: Get API Token

1. Log in to https://quantum.ibm.com/
2. Click account icon (top right)
3. Go to "Account settings"
4. Copy API token

**Token format:**
```
YOUR_TOKEN_HERE (long alphanumeric string)
```

**Security:** Never commit tokens to git repositories!

---

## Access Tiers

### Open Plan (Free)

- **Cost**: Free
- **QPU Time**: 10 min/month
- **Backends**: All 127-qubit systems
- **Priority**: Standard
- **Support**: Community forums

**Best for:** Research, education, small experiments

### Premium Plan

- **Cost**: $1.60/second of QPU time
- **QPU Time**: Unlimited (pay per use)
- **Priority**: High
- **Support**: Email/ticket support

**Best for:** Production research, publications

### Academic Partnerships

Universities and research institutions can apply for:

- **IBM Quantum Network** membership
- Dedicated quantum time
- Technical support
- Co-authorship opportunities

**Apply:** https://www.ibm.com/quantum/network

---

## Installation

### Install Qiskit Runtime

```bash
# Activate your environment
conda activate qcd-vortex  # or your environment name

# Install Qiskit Runtime
pip install qiskit-ibm-runtime>=0.19.0

# Verify installation
python -c "import qiskit_ibm_runtime; print(qiskit_ibm_runtime.__version__)"
# Expected: 0.19.0 or higher
```

### Install Project IBM Validation Module

Already included if you installed the project:

```bash
pip install -e .
```

---

## Authentication

### Method 1: Environment Variable (Recommended)

```bash
# Linux/macOS - Add to ~/.bashrc or ~/.zshrc
export QISKIT_IBM_TOKEN="YOUR_TOKEN_HERE"

# Windows PowerShell
$env:QISKIT_IBM_TOKEN="YOUR_TOKEN_HERE"

# Verify
echo $QISKIT_IBM_TOKEN
```

**Make permanent:**

```bash
# Linux/macOS
echo 'export QISKIT_IBM_TOKEN="YOUR_TOKEN_HERE"' >> ~/.bashrc
source ~/.bashrc
```

### Method 2: Save Account (One-time Setup)

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save account
QiskitRuntimeService.save_account(
    channel='ibm_quantum',  # or 'ibm_cloud'
    token='YOUR_TOKEN_HERE',
    overwrite=True
)

print("Account saved successfully")
```

Credentials stored in: `~/.qiskit/qiskit-ibm.json`

### Method 3: Direct Token (Not Recommended)

Only for testing:

```python
service = QiskitRuntimeService(
    channel='ibm_quantum',
    token='YOUR_TOKEN_HERE'
)
```

### Verify Authentication

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Load saved account
service = QiskitRuntimeService()

# Check access
print(f"Available backends: {len(service.backends())}")
print(f"Backend names: {[b.name for b in service.backends()]}")
```

---

## Backend Selection

### List Available Backends

```python
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# All backends
for backend in service.backends():
    status = backend.status()
    print(f"{backend.name:20s} {backend.num_qubits:3d} qubits "
          f"Queue: {status.pending_jobs:3d} jobs")
```

### Filter Backends

```python
# Only operational backends with 100+ qubits
backends = service.backends(
    operational=True,
    min_num_qubits=100,
    simulator=False
)

for backend in backends:
    print(backend.name)
```

### Check Backend Properties

```python
backend = service.backend('ibm_sherbrooke')

# Queue depth
status = backend.status()
print(f"Pending jobs: {status.pending_jobs}")
print(f"Status: {status.status_msg}")

# Hardware specs
config = backend.configuration()
print(f"Qubits: {config.num_qubits}")
print(f"Max shots: {config.max_shots}")

# Qubit properties
properties = backend.properties()
print(f"T1 (qubit 0): {properties.t1(0)*1e6:.1f} μs")
print(f"T2 (qubit 0): {properties.t2(0)*1e6:.1f} μs")
print(f"Readout error (qubit 0): {properties.readout_error(0)*100:.2f}%")
```

### Automatic Backend Selection

Use project utility:

```python
from ibm_validation import get_optimal_backend

# Automatically select best backend for your needs
backend_name = get_optimal_backend(
    n_qubits=2,
    circuit_depth=50
)

print(f"Recommended backend: {backend_name}")
```

---

## Running Jobs

### Basic Job Execution

```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session

# Initialize service
service = QiskitRuntimeService()
backend = service.backend('ibm_sherbrooke')

# Create circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Run with Sampler primitive
with Session(service=service, backend=backend.name) as session:
    sampler = Sampler(session=session)

    # Configure options
    sampler.options.execution.shots = 4000
    sampler.options.optimization_level = 3

    # Submit job
    job = sampler.run([qc])
    print(f"Job ID: {job.job_id()}")

    # Wait for result
    result = job.result()

    # Extract counts
    counts = result[0].data.meas.get_counts()
    print(f"Counts: {counts}")
```

### Using Project Validation Module

```python
from ibm_validation import SqueezeStateValidator
import numpy as np

# Initialize validator
validator = SqueezeStateValidator(backend_name='ibm_sherbrooke')

# Run validation (single test point)
r_values = np.array([1.0])
T_values = np.array([0.5])

results = validator.validate_entanglement_threshold(
    r_values=r_values,
    T_values=T_values,
    n_shots=8000
)

print(f"Validation accuracy: {results['overall_accuracy']*100:.1f}%")
```

### Monitor Job Progress

```python
# Get job status
job_status = job.status()
print(f"Status: {job_status}")

# Queue position (if queued)
queue_info = job.queue_info()
if queue_info:
    print(f"Queue position: {queue_info.position}")

# Cancel job (if needed)
# job.cancel()
```

---

## Best Practices

### 1. Start Small

Test on simulators first:

```python
from qiskit_aer import AerSimulator

# Local simulator
simulator = AerSimulator()

# Test circuit
job = simulator.run(qc, shots=10000)
result = job.result()
```

### 2. Use Sessions

Sessions batch multiple jobs for efficiency:

```python
with Session(service=service, backend='ibm_sherbrooke') as session:
    sampler = Sampler(session=session)

    # Run multiple circuits in same session
    job1 = sampler.run([circuit1])
    job2 = sampler.run([circuit2])
    job3 = sampler.run([circuit3])
```

### 3. Enable Error Mitigation

```python
from qiskit_ibm_runtime import Estimator

estimator = Estimator(session=session)

# Enable error mitigation (Level 2 = maximum)
estimator.options.resilience_level = 2
estimator.options.optimization_level = 3

# Dynamical decoupling
estimator.options.dynamical_decoupling.enable = True
```

### 4. Optimize Circuits

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Transpile for backend
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=3
)

# Apply transpilation
isa_circuit = pm.run(qc)

print(f"Original depth: {qc.depth()}")
print(f"Optimized depth: {isa_circuit.depth()}")
```

### 5. Check Coherence Limits

Use project utility:

```python
from ibm_validation import check_coherence_limit, estimate_circuit_duration

# Estimate circuit duration
duration = estimate_circuit_duration(
    n_qubits=2,
    n_cx_gates=5,
    backend='ibm_sherbrooke'
)

# Check if within coherence limits
coherence_check = check_coherence_limit(
    circuit_duration=duration,
    backend='ibm_sherbrooke'
)

print(f"Circuit duration: {coherence_check['circuit_duration_us']:.1f} μs")
print(f"Within T1 limit: {coherence_check['within_T1_limit']}")
print(f"Within T2 limit: {coherence_check['within_T2_limit']}")
print(f"Recommended: {coherence_check['recommended']}")
```

---

## Cost Management

### Monitor Usage (Open Plan)

```python
# Check remaining quantum time
# Note: Not directly available via API
# Check at: https://quantum.ibm.com/account
```

### Estimate Job Cost (Premium Plan)

```python
# Estimate QPU time
n_shots = 8000
n_circuits = 10
single_shot_time = 0.0001  # seconds (typical)

total_qpu_time = n_shots * n_circuits * single_shot_time
cost_estimate = total_qpu_time * 1.60  # $1.60/second

print(f"Estimated QPU time: {total_qpu_time:.2f} seconds")
print(f"Estimated cost: ${cost_estimate:.2f}")
```

### Reduce Costs

1. **Use simulators for development**
2. **Minimize shots** (4000-8000 usually sufficient)
3. **Batch circuits** in sessions
4. **Use error mitigation** to reduce need for re-runs
5. **Test on least-busy backend**

---

## Troubleshooting

### Issue: Authentication Fails

**Symptoms:**
```
IBMNotAuthorizedError: Invalid token
```

**Solution:**

```bash
# Check token
echo $QISKIT_IBM_TOKEN

# Re-save account
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN', overwrite=True)"

# Verify
python -c "from qiskit_ibm_runtime import QiskitRuntimeService; s = QiskitRuntimeService(); print('OK')"
```

### Issue: Job Stuck in Queue

**Symptoms:**
Job remains in "QUEUED" state for hours

**Solution:**

```python
# Check queue depth
backend = service.backend('ibm_sherbrooke')
status = backend.status()
print(f"Queue: {status.pending_jobs} jobs")

# Switch to less busy backend
backends = service.backends(operational=True, min_num_qubits=100)
for b in sorted(backends, key=lambda x: x.status().pending_jobs):
    print(f"{b.name}: {b.status().pending_jobs} jobs")
```

### Issue: Job Fails with Error

**Common errors:**

1. **Circuit too deep:**
   ```
   Solution: Reduce circuit depth or optimize transpilation
   ```

2. **Measurement map error:**
   ```python
   # Ensure measurements match classical register size
   qc.measure([0, 1], [0, 1])  # Correct
   ```

3. **Session timeout:**
   ```python
   # Increase session max_time
   session = Session(service=service, backend=backend.name, max_time=3600)
   ```

### Issue: Results Don't Match Theory

**Possible causes:**
1. Decoherence - circuit too long
2. Gate errors - use error mitigation
3. Readout errors - enable mitigation

**Solution:**

```python
# Enable maximum error mitigation
estimator.options.resilience_level = 2

# Increase shots
estimator.options.execution.shots = 10000

# Check circuit fidelity estimate
from ibm_validation.squeezed_state_prep import estimate_circuit_fidelity

T1 = 150e-6  # From backend properties
T2 = 100e-6
fidelity = estimate_circuit_fidelity(r_squeeze=1.0, T1=T1, T2=T2)
print(f"Expected fidelity: {fidelity:.3f}")
```

### Issue: Module Import Errors

**Symptoms:**
```python
ImportError: cannot import name 'QiskitRuntimeService'
```

**Solution:**

```bash
# Upgrade Qiskit Runtime
pip install --upgrade qiskit-ibm-runtime

# Verify version
pip show qiskit-ibm-runtime
# Should be 0.19.0+
```

---

## Example Workflows

### Workflow 1: Test TMST Circuit

```python
from ibm_validation import TMSTCircuitBuilder, SqueezeStateValidator

# Build circuit
builder = TMSTCircuitBuilder()
qc = builder.build_tmst_circuit(r_squeeze=1.0, n_thermal=0.1)

# Validate on hardware
validator = SqueezeStateValidator(backend_name='ibm_sherbrooke')

import numpy as np
results = validator.validate_entanglement_threshold(
    r_values=np.array([1.0]),
    T_values=np.array([0.5]),
    n_shots=6000
)

print(f"Accuracy: {results['overall_accuracy']}")
```

### Workflow 2: Run Full Verification Suite

```python
from ibm_validation import HardwareVerifier

# Initialize verifier
verifier = HardwareVerifier(backend_name='ibm_sherbrooke')

# Run all tests
results = verifier.run_full_verification_suite(n_qubits=2, verbose=True)

# Export results
import json
with open('verification_results.json', 'w') as f:
    json.dump({k: v.__dict__ for k, v in results.items()}, f, indent=2)
```

### Workflow 3: Batch Processing

```python
# Process multiple parameter values
r_values = np.linspace(0.5, 2.0, 10)

for r in r_values:
    qc = builder.build_tmst_circuit(r_squeeze=r)

    with Session(service=service, backend='ibm_sherbrooke') as session:
        sampler = Sampler(session=session)
        job = sampler.run([qc], shots=4000)
        result = job.result()

        # Save result
        counts = result[0].data.meas.get_counts()
        np.save(f'result_r{r:.2f}.npy', counts)
```

---

## Additional Resources

- **IBM Quantum Docs**: https://docs.quantum.ibm.com/
- **Qiskit Tutorials**: https://qiskit.org/learn/
- **Runtime Guide**: https://qiskit.org/ecosystem/ibm-runtime/
- **Error Mitigation**: https://docs.quantum.ibm.com/run/error-mitigation-explanation

## Support

- **IBM Quantum Support**: https://quantum.ibm.com/support
- **Qiskit Slack**: https://qisk.it/join-slack
- **Stack Overflow**: Tag `qiskit`

---

**Last Updated:** February 16, 2026
