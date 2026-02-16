"""
Belle II DIRAC Grid Submission for Massive Data Classification
Integrates with gbasf2 client (web:30, web:31)
"""

import os
import subprocess
from pathlib import Path

class BelleIIGridAnalysis:
    """
    Gestor de trabajos en Belle II DIRAC grid computing.
    """
    
    def __init__(self, project_name, basf2_release="release-08-00-00"):
        self.project_name = project_name
        self.basf2_release = basf2_release
        self.grid_initialized = False
        
    def initialize_grid_proxy(self, valid_hours=24):
        """
        Inicializa proxy DIRAC (web:33).
        
        $ gb2_proxy_init -g belle
        """
        try:
            result = subprocess.run(
                ["gb2_proxy_init", "-g", "belle", "-valid", f"{valid_hours}:00"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Proxy DIRAC inicializado: {result.stdout}")
            self.grid_initialized = True
        except subprocess.CalledProcessError as e:
            print(f"Error inicializando proxy: {e.stderr}")
            return False
        
        return True
    
    def submit_tau_entanglement_job(self, steering_file, input_dataset, 
                                     n_jobs=1000, priority=5):
        """
        Somete trabajo de análisis de entrelazamiento τ+τ- a DIRAC grid.
        
        Clasifica eventos Belle II según criterio de entrelazamiento
        del archivo [file:3]: adaptación bulk-boundary a pares fermiónicos.
        
        Args:
            steering_file: Script basf2 de análisis (Python)
            input_dataset: Dataset LFN en Belle II storage
            n_jobs: Número de trabajos paralelos
            priority: Prioridad (0-10, default 5)
        
        Returns:
            str: Job ID de DIRAC
        """
        if not self.grid_initialized:
            self.initialize_grid_proxy()
        
        # Comando gbasf2 (web:31, web:33)
        cmd = [
            "gbasf2",
            steering_file,
            "-p", self.project_name,
            "-s", self.basf2_release,
            "--input_dslist", input_dataset,
            "--nJobs", str(n_jobs),
            "--priority", str(priority),
            "--force"  # Sobrescribir trabajos existentes
        ]
        
        print(f"Sometiendo {n_jobs} trabajos a DIRAC grid...")
        print(f"Dataset: {input_dataset}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Extraer Job ID de la salida
            job_id = self._parse_job_id(result.stdout)
            print(f"✓ Trabajo sometido exitosamente: Job ID {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            print(f"✗ Error en sumisión: {e.stderr}")
            return None
    
    def _parse_job_id(self, gbasf2_output):
        """Extrae Job ID de la salida de gbasf2."""
        for line in gbasf2_output.split('\n'):
            if "JobID" in line or "Job ID" in line:
                return line.split()[-1]
        return "UNKNOWN"
    
    def monitor_jobs(self, job_id=None):
        """
        Monitorea estado de trabajos en grid (web:32).
        
        $ gb2_job_status <JobID>
        """
        cmd = ["gb2_job_status"]
        if job_id:
            cmd.append(str(job_id))
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        
        return self._parse_job_status(result.stdout)
    
    def _parse_job_status(self, status_output):
        """Parsea salida de gb2_job_status."""
        status_counts = {
            'Done': 0,
            'Running': 0,
            'Waiting': 0,
            'Failed': 0
        }
        
        for line in status_output.split('\n'):
            for state in status_counts.keys():
                if state in line:
                    try:
                        count = int(line.split()[-1])
                        status_counts[state] = count
                    except (ValueError, IndexError):
                        pass
        
        return status_counts
    
    def download_results(self, job_id, output_dir="./belle2_results"):
        """
        Descarga resultados del grid storage (web:31).
        
        $ gb2_job_output <JobID> --dir <output_dir>
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        cmd = [
            "gb2_job_output",
            str(job_id),
            "--dir", output_dir
        ]
        
        print(f"Descargando resultados de Job {job_id}...")
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Resultados descargados en {output_dir}")
            return True
        except subprocess.CalledProcessError:
            print("✗ Error descargando resultados")
            return False


# Script de steering basf2 para clasificación
BASF2_STEERING_TEMPLATE = """
#!/usr/bin/env python3
\"\"\"
Belle II steering file: Tau pair entanglement classification
Implements algorithm from [file:3]: Fermionic Bulk-Boundary Adaptation
\"\"\"

import basf2 as b2
import modularAnalysis as ma
from variables import variables as vm

# Crear path de análisis
my_path = b2.create_path()

# Input data
ma.inputMdstList(
    filelist='input.list',
    path=my_path
)

# Reconstruir pares τ+τ-
ma.reconstructDecay(
    'tau+:signal -> e+ nu_e anti-nu_tau',
    'electronID > 0.9',
    path=my_path
)

ma.reconstructDecay(
    'tau-:signal -> e- anti-nu_e nu_tau',
    'electronID > 0.9',
    path=my_path
)

ma.reconstructDecay(
    'Upsilon(4S) -> tau+:signal tau-:signal',
    '',
    path=my_path
)

# Variable customizada: Concurrence (file:3 Eq. 17)
def calculate_concurrence(particle):
    \"\"\"
    C = max(0, λ1 - λ2 - λ3 - λ4)
    donde λi son autovalores de R = sqrt(sqrt(ρ) ρ_tilde sqrt(ρ))
    \"\"\"
    # Implementar cálculo según matriz densidad de spin
    # (requiere acceso a helicidades)
    pass

vm.addAlias('tauConcurrence', calculate_concurrence)

# Variables de entrelazamiento
entanglement_vars = [
    'tauConcurrence',
    'daughter(0, cosTheta)',  # ángulo polar τ+
    'daughter(1, cosTheta)',  # ángulo polar τ-
    'M',  # masa invariante
    'cosAngleBetweenMomentaInCMS(0, 1)'  # correlación angular
]

# Output NTuple
ma.variablesToNtuple(
    'Upsilon(4S)',
    entanglement_vars,
    filename='tau_entanglement_output.root',
    path=my_path
)

# Ejecutar
b2.process(my_path)
print(b2.statistics)
"""


# Script HPC para lanzar trabajos
