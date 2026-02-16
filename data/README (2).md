# Sample Configuration Files

This directory contains example configuration files for the QCD vortex 
entanglement analysis framework.

## Available Configs

### MPS Parameters

- `mps_default.yaml` - Default MPS simulation parameters
- `mps_large.yaml` - Large-scale MPS (high bond dimension)
- `mps_quick.yaml` - Quick tests (low bond dimension)

### Vortex Detection

- `vortex_detection.yaml` - Vortex identification parameters
- `vortex_threshold.yaml` - Threshold configurations

### Entanglement

- `entanglement_default.yaml` - Default entanglement measures
- `entanglement_bipartite.yaml` - Bipartite entanglement

### Belle II

- `belle2_analysis.yaml` - Belle II data analysis settings
- `belle2_classification.yaml` - Event classification parameters

### IBM Quantum

- `ibm_quantum_default.yaml` - IBM Quantum Platform settings
- `ibm_backend_config.yaml` - Backend selection and options

## Usage

```python
import yaml

# Load configuration
with open('data/sample_configs/mps_default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use in your code
from qcd_vortex import VortexMPS
mps = VortexMPS(**config)
```

## Creating Custom Configs

Copy a sample config and modify:

```bash
cp data/sample_configs/mps_default.yaml my_config.yaml
# Edit my_config.yaml
```

---

**Note:** These are example configurations. Modify according to your needs.
