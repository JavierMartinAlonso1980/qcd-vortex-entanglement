#!/bin/bash
#SBATCH --job-name=belle2_tau_entanglement
#SBATCH --nodes=50
#SBATCH --ntasks-per-node=32
#SBATCH --time=48:00:00
#SBATCH --partition=high-throughput
#SBATCH --mem=128GB
#SBATCH --output=logs/belle2_%j.out
#SBATCH --error=logs/belle2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tu-email@institution.edu

# Configuración Belle II
export BELLE2_RELEASE=release-08-00-00
export BELLE2_LOCAL_DIR=/path/to/belle2/software
source ${BELLE2_LOCAL_DIR}/setup_belle2.sh ${BELLE2_RELEASE}

# Inicializar proxy DIRAC
gb2_proxy_init -g belle -valid 168:00  # 1 semana

# Ejecutar análisis Python
module load python/3.10
source venv/bin/activate

python << 'EOF'
from belle2_analysis.grid_submission import BelleIIGridAnalysis

# Inicializar manager
analyzer = BelleIIGridAnalysis(
    project_name="tau_entanglement_2026",
    basf2_release="release-08-00-00"
)

# Someter 10,000 trabajos clasificando todos los datos 2024-2025
job_id = analyzer.submit_tau_entanglement_job(
    steering_file="steering_tau_classification.py",
    input_dataset="/belle/MC/release-08-00-00/DB00002179/MC15ri/prod00028766/s00/e1003/4S/r00000/all/mdst/*/*.root",
    n_jobs=10000,
    priority=7
)

print(f"Grid Job ID: {job_id}")

# Monitorear cada hora
import time
while True:
    status = analyzer.monitor_jobs(job_id)
    print(f"Status: {status}")
    
    if status['Done'] + status['Failed'] >= 10000:
        break
    
    time.sleep(3600)  # 1 hora

# Descargar resultados
analyzer.download_results(job_id, output_dir="/scratch/tau_results")
EOF

# Post-procesamiento: agregar NTuples
hadd -f tau_entanglement_merged.root /scratch/tau_results/*/tau_entanglement_output.root

echo "Análisis Belle II completado"
