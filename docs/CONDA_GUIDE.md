# 🐍 Guía de Uso de Conda - environment.yml

## ¿Por Qué Usar Conda?

### Ventajas sobre pip/venv:

1. **Gestión de Python**: Especifica versión exacta de Python
2. **Dependencias del sistema**: Instala gcc, cmake, ROOT, etc.
3. **Reproducibilidad**: Misma versión en todos los sistemas
4. **Belle II**: ROOT se instala fácilmente con conda
5. **Múltiples canales**: conda-forge tiene muchos paquetes científicos

## 📦 Archivos Creados

1. **environment.yml** - Entorno completo con todo
2. **environment-minimal.yml** - Entorno mínimo básico

## 🚀 Instalación con Conda

### Opción 1: Entorno Completo (Recomendado)

```bash
# Crear entorno
conda env create -f environment.yml

# Activar
conda activate qcd-vortex

# Verificar
python -c "import seemps_vortex; print('OK')"
```

### Opción 2: Entorno Mínimo

```bash
# Crear entorno mínimo
conda env create -f environment-minimal.yml

# Activar
conda activate qcd-vortex

# Instalar extras después según necesites
conda install -c conda-forge root  # Para Belle II
pip install qiskit qiskit-ibm-runtime  # Para IBM Quantum
```

### Opción 3: Crear Manualmente

```bash
# Crear entorno vacío
conda create -n qcd-vortex python=3.10

# Activar
conda activate qcd-vortex

# Instalar desde requirements
pip install -r requirements.txt

# Instalar proyecto
pip install -e .
```

## 📝 Comandos Útiles

### Gestión del Entorno

```bash
# Listar entornos
conda env list

# Activar entorno
conda activate qcd-vortex

# Desactivar
conda deactivate

# Actualizar entorno desde yml
conda env update -f environment.yml --prune

# Eliminar entorno
conda env remove -n qcd-vortex
```

### Exportar/Compartir Entorno

```bash
# Exportar entorno exacto (incluye todas las versiones)
conda env export > environment-exact.yml

# Exportar solo las dependencias especificadas
conda env export --from-history > environment-clean.yml

# Exportar para requirements.txt (solo pip)
pip freeze > requirements-frozen.txt
```

## 🔧 Modificar environment.yml

### Añadir Paquetes

Edita `environment.yml` y añade bajo `dependencies:`:

```yaml
dependencies:
  - tu-nuevo-paquete>=1.0.0
```

O bajo `pip:` si solo está en PyPI:

```yaml
  - pip:
      - tu-paquete-pip>=2.0.0
```

Luego actualiza:

```bash
conda env update -f environment.yml --prune
```

### Especificar Versión Exacta

Para máxima reproducibilidad:

```yaml
dependencies:
  - numpy=1.26.3  # Versión exacta
  - scipy>=1.11.0,<1.12.0  # Rango
```

## 🌍 Canales de Conda

### Orden de Prioridad

El archivo usa estos canales:
1. **conda-forge**: Paquetes científicos actualizados
2. **defaults**: Canal oficial de Anaconda

### Añadir Canales

```yaml
channels:
  - conda-forge
  - defaults
  - bioconda  # Si necesitas paquetes bioinformáticos
```

## 🎯 Casos de Uso

### Para Desarrollo

```bash
conda env create -f environment.yml
conda activate qcd-vortex
pip install -e ".[dev]"
```

### Para Belle II

```bash
conda env create -f environment.yml
conda activate qcd-vortex

# ROOT ya está instalado con conda
root --version
```

### Para IBM Quantum

```bash
conda env create -f environment.yml
conda activate qcd-vortex

# Qiskit ya está instalado
python -c "import qiskit; print(qiskit.__version__)"
```

## 🔄 Actualizar Dependencias

```bash
# Actualizar todos los paquetes
conda update --all

# Actualizar paquete específico
conda update numpy

# Actualizar desde environment.yml
conda env update -f environment.yml --prune
```

## 🐛 Troubleshooting

### Problema: Conflictos de paquetes

```bash
# Solución 1: Crear entorno desde cero
conda env remove -n qcd-vortex
conda env create -f environment.yml

# Solución 2: Usar mamba (más rápido)
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

### Problema: Paquete no encontrado

```bash
# Buscar en qué canal está
conda search nombre-paquete

# Buscar en conda-forge
conda search -c conda-forge nombre-paquete

# Instalar desde canal específico
conda install -c conda-forge nombre-paquete
```

### Problema: pip vs conda mix

**Regla general:**
1. Instala primero con conda todo lo que puedas
2. Luego instala con pip lo que solo esté en PyPI
3. Instala el proyecto con pip (pip install -e .)

## 📊 Comparación de Métodos

| Método | Ventaja | Desventaja |
|--------|---------|-----------|
| **conda (environment.yml)** | Reproducible, incluye sistema | Más lento, más espacio |
| **pip (requirements.txt)** | Rápido, ligero | No gestiona Python ni sistema |
| **Ambos** | Lo mejor de ambos | Requiere ambas herramientas |

## 💡 Mejores Prácticas

1. **Versiona environment.yml** en git
2. **Excluye environment-exact.yml** (demasiado específico)
3. **Usa conda para**: Python, gcc, ROOT, numpy, scipy
4. **Usa pip para**: Paquetes solo en PyPI (seemps2, qiskit)
5. **Documenta** comandos de instalación en README.md
6. **Prueba** la instalación en entorno limpio antes de compartir

## 🎓 Recursos

- Conda docs: https://docs.conda.io/
- Conda-forge: https://conda-forge.org/
- Mamba (más rápido): https://mamba.readthedocs.io/

---

**Última actualización:** 2026-02-16
