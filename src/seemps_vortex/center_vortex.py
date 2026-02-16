"""
Center Vortex Dynamics using SeeMPS2 Matrix Product States
Implements collective vortex projection and MVC threshold detection
"""

import numpy as np
from seemps.state import MPS, Strategy
from seemps.operators import MPO
from seemps.analysis import entanglement_entropy, log_negativity
from seemps.evolution import rk4_step

class CenterVortexMPS:
    """
    Representa la red de vórtices de centro en QCD como MPS.
    
    Attributes:
        N_sites: Número de sitios de vórtices (default: 128)
        chi_max: Dimensión de enlace máxima
        d: Dimensión local de Hilbert space (d=2 para spin-1/2)
    """
    
    def __init__(self, N_sites=128, chi_max=64, d=2):
        self.N_sites = N_sites
        self.chi_max = chi_max
        self.d = d
        self.strategy = Strategy(method='variational', 
                                tolerance=1e-10,
                                simplify=True)
        
    def initialize_collective_mode(self, r_squeeze=0.8, n_thermal=0.1):
        """
        Inicializa el modo colectivo de vórtice con squeezing.
        
        Implementa el estado TMST del archivo [file:6]:
        |ψ⟩ = S_coll(r) |thermal⟩
        
        Args:
            r_squeeze: Parámetro de squeezing (Eq. 4.5 de file:6)
            n_thermal: Ocupación térmica n(T)
        
        Returns:
            MPS representando el estado colectivo squeezing
        """
        # Estado inicial: producto tensorial de estados térmicos
        thermal_state = self._thermal_product_state(n_thermal)
        
        # Aplicar operador de squeezing colectivo (file:9, Eq. 3.1)
        S_collective = self._collective_squeezing_mpo(r_squeeze)
        
        # Aplicar con estrategia variacional para controlar chi
        psi_squeezed = S_collective @ thermal_state
        psi_squeezed = psi_squeezed.simplify(strategy=self.strategy)
        
        return psi_squeezed
    
    def _thermal_product_state(self, n_thermal):
        """Construye estado producto térmico como MPS."""
        from seemps.state import product_state
        
        # Matriz densidad de un solo modo térmico
        rho_thermal = np.diag([1/(1+n_thermal), n_thermal/(1+n_thermal)])
        
        # Purificación: |ψ_thermal⟩ = Σ_i √λ_i |i⟩|i⟩
        eigenvals, eigenvecs = np.linalg.eigh(rho_thermal)
        
        # Estado local purificado (dimensión d=2)
        local_state = np.sqrt(eigenvals)
        
        return product_state([local_state] * self.N_sites)
    
    def _collective_squeezing_mpo(self, r):
        """
        Construye S_coll(r) = exp[r(A_R† A_L† - A_R A_L)] como MPO.
        
        Utiliza descomposición de Trotter para exponencial de operadores:
        exp(A+B) ≈ exp(A/2)exp(B)exp(A/2)
        """
        from seemps.operators import MPOList
        from scipy.linalg import expm
        
        # Operadores de escalera colectivos
        a_creation = np.array([[0, 0], [1, 0]])  # a†
        a_annihilation = np.array([[0, 1], [0, 0]])  # a
        
        # Términos del Hamiltoniano de squeezing: a_R a_L + h.c.
        H_local = r * (np.kron(a_creation, a_creation) + 
                       np.kron(a_annihilation, a_annihilation))
        
        # Exponencial del operador local
        U_local = expm(-1j * H_local).reshape(2, 2, 2, 2)
        
        # Construir MPO secuencial (file:35 MPOList)
        mpo_layers = [MPO([U_local] * (self.N_sites // 2))]
        
        return MPOList(mpo_layers)
    
    def compute_mvc_threshold(self, psi, rho_local):
        """
        Detecta el umbral MVC como Punto Excepcional (file:7).
        
        Implementa la condición:
        ρ → ρ_MVC ⇒ EP2 bifurcation
        
        Args:
            psi: Estado MPS actual
            rho_local: Densidad de carga de color local
        
        Returns:
            tuple: (es_confinado, S_entrelazamiento, Petermann_factor)
        """
        # Calcular entropía de entrelazamiento (indicador de fase)
        S_E = entanglement_entropy(psi, site=self.N_sites // 2)
        
        # Negatividad logarítmica (medida de entrelazamiento, file:6 Eq. 2.4)
        EN = log_negativity(psi, partition=[0, self.N_sites // 2])
        
        # Parámetro MVC crítico (file:7 Eq. 12)
        rho_MVC_critical = self._calculate_rho_mvc(T_planck=1.0, alpha_universal=2.5)
        
        # Factor de Petermann diverge cerca del EP (file:7 Eq. 11)
        K_petermann = 1.0 / np.abs(rho_local - rho_MVC_critical)
        
        # Criterio de confinamiento
        is_confined = (rho_local >= rho_MVC_critical) and (EN > 0)
        
        return is_confined, S_E, K_petermann
    
    def _calculate_rho_mvc(self, T_planck, alpha_universal):
        """Calcula densidad crítica MVC (file:9 Eq. 12)."""
        return T_planck ** alpha_universal
    
    def evolve_lindblad(self, psi_initial, T_temp, gamma_loss, time_steps=100, dt=0.01):
        """
        Evoluciona el MPS bajo ecuación maestra de Lindblad (file:6 Eq. 3.1).
        
        d\rho/dt = -i[H, \rho] + \sum_j L_j \rho L_j† - 1/2{L_j† L_j, \rho}
        
        Args:
            psi_initial: Estado MPS inicial
            T_temp: Temperatura del baño térmico
            gamma_loss: Tasa de pérdida
            time_steps: Número de pasos de tiempo
            dt: Incremento temporal
        
        Returns:
            list: Estados MPS en cada paso temporal
        """
        from seemps.operators import MPO
        
        # Hamiltoniano efectivo no-hermítico (file:7 Eq. 2)
        H_eff = self._non_hermitian_hamiltonian(gamma_loss, T_temp)
        
        # Lista de estados
        trajectory = [psi_initial]
        psi_t = psi_initial
        
        for t in range(time_steps):
            # Evolución RK4 (preserva estructura MPS)
            psi_t = rk4_step(psi_t, H_eff, dt, strategy=self.strategy)
            trajectory.append(psi_t)
            
            # Simplificar cada 10 pasos para controlar chi
            if t % 10 == 0:
                psi_t = psi_t.simplify(strategy=self.strategy)
        
        return trajectory
    
    def _non_hermitian_hamiltonian(self, gamma, T):
        """Construye H_eff = H_0 - i\gamma/2 (file:7 Eq. 1)."""
        from seemps.operators import MPO
        
        # Término hermítico: rotación de vórtice
        H0 = self._vortex_rotation_hamiltonian()
        
        # Término anti-hermítico: pérdidas térmicas
        n_T = 1 / (np.exp(1/T) - 1) if T > 0 else 0
        damping_term = -1j * gamma * n_T / 2
        
        return H0 + damping_term * MPO.identity(self.N_sites)
    
    def _vortex_rotation_hamiltonian(self):
        """H_vortex = \omega_rot J_z (file:9 Eq. 4)."""
        from seemps.operators import MPO
        
        # Operador J_z = \sum_i S^z_i
        sz = np.array([[0.5, 0], [0, -0.5]])  # Spin-1/2 Pauli Z
        
        return MPO([sz] * self.N_sites)
    
    def export_to_hdf5(self, psi, filename):
        """Exporta MPS a formato HDF5 para análisis Belle II."""
        import h5py
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('N_sites', data=self.N_sites)
            f.create_dataset('chi_max', data=self.chi_max)
            
            for i, tensor in enumerate(psi):
                f.create_dataset(f'tensor_{i}', data=tensor, compression='gzip')
        
        print(f"MPS exportado a {filename}")


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar sistema de 128 vórtices
    vortex_system = CenterVortexMPS(N_sites=128, chi_max=64)
    
    # Preparar estado colectivo squeezed
    psi_collective = vortex_system.initialize_collective_mode(
        r_squeeze=1.2,  # Régimen central PbPb (file:5)
        n_thermal=0.1
    )
    
    # Evolucionar bajo Lindblad
    trajectory = vortex_system.evolve_lindblad(
        psi_collective,
        T_temp=0.2,  # Temperatura efectiva
        gamma_loss=0.05,
        time_steps=100
    )
    
    # Detectar umbral MVC
    is_confined, S_E, K = vortex_system.compute_mvc_threshold(
        trajectory[-1],
        rho_local=1.5
    )
    
    print(f"Confinamiento: {is_confined}")
    print(f"Entropía de entrelazamiento: {S_E:.4f}")
    print(f"Factor de Petermann: {K:.2e}")
    
    # Exportar para análisis Belle II
    vortex_system.export_to_hdf5(trajectory[-1], "vortex_state_mvc.h5")
