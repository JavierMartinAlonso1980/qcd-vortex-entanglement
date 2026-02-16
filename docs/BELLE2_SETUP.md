# Belle II Grid Computing Setup

Complete guide for setting up Belle II data analysis and grid job submission via DIRAC/gbasf2.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Grid Certificate Setup](#grid-certificate-setup)
4. [DIRAC Installation](#dirac-installation)
5. [basf2 Setup](#basf2-setup)
6. [Grid Proxy Initialization](#grid-proxy-initialization)
7. [Data Access](#data-access)
8. [Job Submission](#job-submission)
9. [Monitoring](#monitoring)
10. [Troubleshooting](#troubleshooting)

---

## Overview

Belle II uses distributed computing infrastructure spanning 55+ sites worldwide:

- **DIRAC**: Middleware for job distribution
- **gbasf2**: Grid-aware basf2 wrapper for job submission
- **basf2**: Belle II Analysis Software Framework
- **CVMFS**: Software distribution via network filesystem

**Key Resources:**
- Belle II Computing: https://confluence.desy.de/display/BI/Computing
- DIRAC Docs: https://dirac.readthedocs.io/
- Grid Certificate: https://ca.cern.ch/ca/

---

## Prerequisites

### 1. Belle II Collaboration Membership

You must be a registered Belle II collaborator with:
- Valid Belle II user account
- Access to Belle II computing resources
- Completed computing training modules

**Check your status:**
```bash
ssh <username>@bastion.belle2.org
# If successful, you have access
```

### 2. Institutional Grid Certificate

Required for authentication on the grid.

**Certificate Authorities:**
- **Europe**: CERN CA, DFN-Verein CA
- **USA**: CILogon, DOEGrids
- **Japan**: KEK CA
- **Other**: See https://www.eugridpma.org/

---

## Grid Certificate Setup

### Step 1: Request Certificate

#### Option A: CERN CA (Recommended for European users)

1. Go to https://ca.cern.ch/ca/
2. Click "New Grid User Certificate"
3. Fill out form with:
   - Name (must match official ID)
   - Email (institutional)
   - Registration Authority approval
4. Generate key pair in browser
5. Download certificate (`.p12` file)

#### Option B: CILogon (USA users)

1. Go to https://cilogon.org/
2. Select your institution
3. Authenticate via institutional login
4. Download certificate

### Step 2: Install Certificate

```bash
# Create certificate directory
mkdir -p ~/.globus

# Convert .p12 to PEM format (if needed)
# Extract user certificate
openssl pkcs12 -in certificate.p12 -clcerts -nokeys -out ~/.globus/usercert.pem

# Extract private key
openssl pkcs12 -in certificate.p12 -nocerts -out ~/.globus/userkey.pem

# Set correct permissions (CRITICAL)
chmod 644 ~/.globus/usercert.pem
chmod 400 ~/.globus/userkey.pem

# Verify certificate
openssl x509 -in ~/.globus/usercert.pem -noout -text
```

**Security Notes:**
- Never share your private key (`userkey.pem`)
- Back up `.p12` file securely
- Certificate valid for 1 year (renew before expiration)

### Step 3: Register Certificate with Belle II

1. Extract certificate DN:
   ```bash
   openssl x509 -in ~/.globus/usercert.pem -noout -subject
   ```

2. Register DN at Belle II user portal:
   - Go to https://b2mmsdev.belle2.org/
   - Profile → Grid Certificate
   - Add certificate DN
   - Wait for approval (~1 business day)

---

## DIRAC Installation

Belle II uses DIRAC for grid job management.

### Option 1: Using CVMFS (Recommended)

```bash
# Source DIRAC environment
source /cvmfs/belle.cern.ch/tools/dirac/pro/bashrc

# Verify installation
dirac-version
# Expected: Belle2Dirac vXrY
```

### Option 2: Local Installation

```bash
# Download installer
wget https://github.com/DIRACGrid/DIRAC/raw/integration/Core/scripts/dirac-install.py

# Install Belle II DIRAC
python dirac-install.py -t client Belle2

# Source environment
source bashrc

# Configure for Belle II
dirac-configure \
  --Setup=Belle2-Production \
  --ConfigurationServer=dips://vodirac01.cc.kek.jp:9135/Configuration/Server \
  --SkipCAChecks
```

### Verify DIRAC Installation

```bash
dirac-info
# Should show Belle II VO and available sites
```

---

## basf2 Setup

### Option 1: CVMFS (Recommended)

```bash
# List available releases
ls /cvmfs/belle.cern.ch/el9/releases/

# Source specific release
source /cvmfs/belle.cern.ch/tools/b2setup release-08-00-00

# Verify basf2
basf2 --version
# Expected: Release: release-08-00-00
```

### Option 2: Local Build (Advanced)

Only for development work. See: https://software.belle2.org/

```bash
# Clone repository
git clone https://github.com/belle2/basf2.git
cd basf2

# Build
scons -j8

# Setup environment
source tools/setup_belle2
```

### Test basf2

```python
# Create test script: test_basf2.py
import basf2 as b2

path = b2.Path()
path.add_module('EventInfoSetter', evtNumList=[10])
path.add_module('Progress')
b2.process(path)
```

```bash
basf2 test_basf2.py
# Should process 10 events successfully
```

---

## Grid Proxy Initialization

Grid proxy authenticates you for 24 hours.

### Initialize Proxy

```bash
# Standard proxy (24 hours)
gb2_proxy_init -g belle

# Long-lived proxy (7 days, for long jobs)
gb2_proxy_init -g belle -v 168:00

# Check proxy status
gb2_proxy_info
```

**Expected output:**
```
subject      : /DC=ch/DC=cern/OU=Organic Units/...
issuer       : /DC=ch/DC=cern/OU=Organic Units/...
identity     : /DC=ch/DC=cern/OU=Organic Units/...
timeleft     : 23:59:57
DIRAC group  : belle
path         : /tmp/x509up_u1000
username     : your_username
```

### Automate Proxy Renewal

Add to crontab for automatic renewal:

```bash
# Edit crontab
crontab -e

# Add line (renew daily at 2 AM)
0 2 * * * source /cvmfs/belle.cern.ch/tools/dirac/pro/bashrc && gb2_proxy_init -g belle -v 168:00 > /dev/null 2>&1
```

---

## Data Access

### Browse Data Catalog

```bash
# List available data
gb2_ds_list "/belle/MC/release-08-00-00/DB00002481/MC15ri_a/prod00029130/*.root"

# Search for specific dataset
gb2_ds_search --dataset "*tau*pair*" --release release-08-00-00

# Get dataset info
gb2_ds_info /belle/MC/.../mdst/*.root
```

### Download Files (for local testing)

```bash
# Download single file
gb2_ds_get lfn:/belle/MC/.../mdst_000001.root

# Download entire dataset (careful, can be TBs!)
gb2_ds_get --dataset "/belle/MC/.../mdst/*.root" --max-files 10
```

### Data Locations

- **KEKCC** (KEK, Japan): Primary tape storage
- **BNL** (USA): Disk cache
- **GridKa** (Germany): Disk cache
- **CNAF** (Italy): Disk cache

---

## Job Submission

### Basic gbasf2 Job

#### 1. Create Steering File

```python
# steering_simple.py
import basf2 as b2

path = b2.Path()

# Input
path.add_module('RootInput', inputFileNames=['input.root'])

# Analysis modules
path.add_module('ParticleLoader', decayStrings=['tau+:all'])
path.add_module('ParticleSelector', 
                decayString='tau+:selected',
                cut='pt > 0.5')

# Output
path.add_module('RootOutput', 
                outputFileName='output.root',
                branchNames=['TauPairs'])

b2.process(path)
```

#### 2. Submit to Grid

```bash
# Initialize project
gbasf2 --init MyTauAnalysis

# Submit job
gbasf2 steering_simple.py \
  --input_dslist /belle/MC/.../mdst/*.root \
  --project_name MyTauAnalysis \
  --release release-08-00-00 \
  --force_submission
```

**Job options:**
- `--njobs 100`: Number of parallel jobs
- `--priority 5`: Job priority (1-10)
- `--jobtype User`: Job type
- `--cputime 24`: Max CPU hours per job

### Advanced: Project Integration

Use the project's grid submission module:

```bash
# Submit via project script
sbatch scripts/hpc_submit_belle2.sh
```

Or programmatically:

```python
from belle2_analysis import BelleIIGridAnalysis

analyzer = BelleIIGridAnalysis(
    project_name="TauEntanglement",
    basf2_release="release-08-00-00"
)

job_id = analyzer.submit_tau_entanglement_job(
    steering_file='steering_tau_classification.py',
    input_dataset='/belle/MC/.../mdst/*.root',
    n_jobs=1000,
    priority=5
)

print(f"Job submitted: {job_id}")
```

---

## Monitoring

### Check Job Status

```bash
# Get job status
gb2_job_status <job_id>

# Detailed status
gb2_job_status <job_id> --verbose

# Watch status (updates every 5 min)
watch -n 300 'gb2_job_status <job_id>'
```

### Use Monitoring Script

```bash
# Real-time monitoring with auto-resubmit
python scripts/grid_job_monitor.py \
  --job-id <job_id> \
  --check-interval 300 \
  --auto-resubmit \
  --log-file monitor.log
```

### Web Monitoring

- **DIRAC Web Portal**: https://vodirac01.cc.kek.jp:8443/DIRAC/
- **Belle II Dashboard**: https://b2-grafana.belle2.org/

### Download Results

```bash
# Download output files
gb2_job_output <job_id>

# Download logs
gb2_job_output <job_id> --logs

# Download to specific directory
gb2_job_output <job_id> --output-dir ./results/
```

---

## Troubleshooting

### Issue: Certificate Expired

**Symptoms:**
```
Error: Your proxy has expired
```

**Solution:**
```bash
# Renew certificate (if within 1 year)
gb2_proxy_init -g belle

# If certificate expired (> 1 year), request new certificate
```

### Issue: Proxy Initialization Fails

**Symptoms:**
```
Error: Cannot find valid certificate
```

**Solution:**

```bash
# Check certificate files exist
ls -la ~/.globus/
# Should show usercert.pem and userkey.pem

# Check permissions
chmod 644 ~/.globus/usercert.pem
chmod 400 ~/.globus/userkey.pem

# Verify certificate is valid
openssl x509 -in ~/.globus/usercert.pem -noout -dates
```

### Issue: "VO belle not found"

**Solution:**

```bash
# Re-configure DIRAC
dirac-configure \
  --Setup=Belle2-Production \
  --ConfigurationServer=dips://vodirac01.cc.kek.jp:9135/Configuration/Server

# Or use CVMFS version
source /cvmfs/belle.cern.ch/tools/dirac/pro/bashrc
```

### Issue: Job Stuck in "Waiting" State

**Possible causes:**
1. High grid load - wait or increase priority
2. Dataset not replicated - choose different site
3. Resource requirements too strict - reduce CPUTime

**Solution:**

```bash
# Kill and resubmit
gb2_job_kill <job_id>
gbasf2 steering.py --resubmit --job-id <job_id>
```

### Issue: Jobs Keep Failing

**Solution:**

```bash
# Download and check logs
gb2_job_output <job_id> --logs

# Common issues:
# 1. Steering file error - test locally first
# 2. Memory limit - reduce chi_max or dataset size
# 3. Timeout - increase --cputime
```

**Test steering file locally:**

```bash
# Download one input file
gb2_ds_get lfn:/belle/MC/.../mdst_000001.root

# Run locally
basf2 steering.py -i mdst_000001.root
```

### Issue: Cannot Access Data

**Symptoms:**
```
Error: File not found on any storage element
```

**Solution:**

```bash
# Check data replication
gb2_ds_info /belle/MC/.../mdst/*.root

# Request data replication (if needed)
gb2_ds_replicate /belle/MC/.../mdst/*.root --site GridKa-disk
```

---

## Best Practices

### 1. Test Locally First

Always test steering files on a single file before grid submission:

```bash
# Download test file
gb2_ds_get lfn:/belle/MC/.../mdst_000001.root

# Test
basf2 steering.py -i mdst_000001.root -n 100
```

### 2. Use Appropriate Resources

```bash
# Small test job
gbasf2 steering.py --njobs 10 --cputime 1

# Production job
gbasf2 steering.py --njobs 1000 --cputime 24 --priority 5
```

### 3. Monitor and Resubmit

Use the monitoring script to auto-resubmit failed jobs:

```bash
python scripts/grid_job_monitor.py \
  --job-id <job_id> \
  --auto-resubmit \
  --max-retries 3
```

### 4. Clean Up Old Jobs

```bash
# List your jobs
gb2_job_list

# Kill old jobs
gb2_job_kill <old_job_id>

# Clean up project
gbasf2 --cleanup MyOldProject
```

---

## Example Workflow

Complete workflow for tau pair analysis:

```bash
# 1. Initialize environment
source /cvmfs/belle.cern.ch/tools/b2setup release-08-00-00
source /cvmfs/belle.cern.ch/tools/dirac/pro/bashrc

# 2. Initialize proxy
gb2_proxy_init -g belle

# 3. Test steering file locally
gb2_ds_get lfn:/belle/MC/.../mdst_000001.root
basf2 steering_tau_analysis.py -i mdst_000001.root -n 1000

# 4. Submit to grid
gbasf2 steering_tau_analysis.py \
  --input_dslist /belle/MC/.../mdst/*.root \
  --project_name TauEntanglement_v1 \
  --release release-08-00-00 \
  --njobs 500 \
  --cputime 12 \
  --force_submission

# 5. Monitor
python scripts/grid_job_monitor.py --job-id 12345678

# 6. Download results
gb2_job_output 12345678 --output-dir ./results/

# 7. Merge output files
hadd -f merged_output.root results/*.root
```

---

## Additional Resources

- **Belle II Software Portal**: https://software.belle2.org/
- **Computing Talks**: https://confluence.desy.de/display/BI/Computing+Tutorials
- **DIRAC User Guide**: https://dirac.readthedocs.io/
- **Grid Certificate Help**: https://ca.cern.ch/ca/Help/

## Support

- **Belle II Computing Help**: belle2-distributed-computing-admins@desy.de
- **DIRAC Support**: belle2-dirac@belle2.org
- **General Questions**: belle2-software@belle2.org

---

**Last Updated:** February 16, 2026

