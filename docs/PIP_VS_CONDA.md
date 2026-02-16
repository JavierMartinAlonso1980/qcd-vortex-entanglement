# 📊 pip vs Conda: ¿Cuál Usar?

## Decisión Rápida

```
┌─────────────────────────────────────┐
│ ¿Usas Conda o planeas usarlo?      │
│                                     │
│  SÍ  → Usa environment.yml         │
│  NO  → Usa requirements.txt         │
│  AMBOS → Usa ambos archivos        │
└─────────────────────────────────────┘
```

## ✅ Usa Conda (environment.yml) Si:

- Trabajas en HPC/clusters con módulos conda
- Necesitas Belle II con ROOT
- Compartes código con otros que usan conda
- Necesitas control total del entorno (Python + sistema)
- Quieres máxima reproducibilidad

## ✅ Usa pip (requirements.txt) Si:

- Prefieres virtualenv/venv
- Tu proyecto es pure Python
- Quieres instalación rápida y ligera
- No necesitas compiladores o librerías del sistema
- Solo usas paquetes de PyPI

## 🎯 Recomendación para Este Proyecto

### ¡USA AMBOS!

**Por qué:**
- Algunos usuarios prefieren conda
- Algunos usuarios prefieren pip
- No cuesta nada tener ambos
- Ambos funcionan independientemente

**Estructura final:**
```
proyecto/
├── environment.yml           ← Para usuarios de Conda
├── environment-minimal.yml   ← Conda mínimo
├── requirements.txt          ← Para usuarios de pip
├── requirements-dev.txt      ← pip desarrollo
├── requirements-complete.txt ← pip completo
├── setup.py                  ← Instalación del proyecto
├── pyproject.toml           ← Config moderna
└── ...
```

## 📋 Tabla Comparativa Detallada

| Característica | pip + requirements.txt | conda + environment.yml |
|----------------|------------------------|-------------------------|
| **Instalación** | `pip install -r requirements.txt` | `conda env create -f environment.yml` |
| **Velocidad** | ⚡ Rápido | 🐌 Más lento |
| **Espacio disco** | 💾 Pequeño | 💾💾 Grande |
| **Python version** | ❌ No gestiona | ✅ Gestiona |
| **Dependencias sistema** | ❌ No | ✅ Sí (gcc, cmake, ROOT) |
| **Canales** | PyPI | PyPI + conda-forge + otros |
| **Reproducibilidad** | Buena | Excelente |
| **Portabilidad** | Buena | Excelente |
| **Belle II (ROOT)** | ⚠️ Difícil | ✅ Fácil |
| **Actualizaciones** | ⚡ Rápidas | 🐌 Lentas |
| **Comunidad** | 🔥 Grande | 🔥 Grande (científica) |

## 🔄 Flujo de Trabajo Híbrido

Puedes usar ambos:

```bash
# 1. Crear entorno conda
conda env create -f environment.yml
conda activate qcd-vortex

# 2. Dentro del entorno, usar pip para extras
pip install paquete-extra

# 3. Instalar proyecto
pip install -e .
```

## 📄 ¿Qué Archivo Incluir en Git?

✅ **SÍ incluir:**
- environment.yml
- environment-minimal.yml
- requirements.txt
- requirements-dev.txt
- requirements-complete.txt

❌ **NO incluir:**
- environment-exact.yml (demasiado específico)
- requirements-frozen.txt (versiones exactas, no portables)

En `.gitignore`:
```
environment-exact.yml
requirements-frozen.txt
```

## 🎓 Mejores Prácticas

1. **Documenta ambos métodos** en README.md
2. **Prueba ambos** antes de publicar
3. **Usa conda** para dependencias del sistema
4. **Usa pip** para paquetes pure-Python
5. **Mantén sincronizados** ambos archivos

## 🚀 Comandos Lado a Lado

| Acción | pip | conda |
|--------|-----|-------|
| Crear entorno | `python -m venv venv` | `conda create -n nombre` |
| Activar | `source venv/bin/activate` | `conda activate nombre` |
| Instalar deps | `pip install -r requirements.txt` | `conda env create -f environment.yml` |
| Añadir paquete | `pip install paquete` | `conda install paquete` |
| Actualizar | `pip install --upgrade paquete` | `conda update paquete` |
| Exportar | `pip freeze > requirements.txt` | `conda env export > environment.yml` |
| Eliminar entorno | `rm -rf venv/` | `conda env remove -n nombre` |

## 💡 Consejo Final

**Para este proyecto:** 

Mantén **ambos** archivos (environment.yml + requirements.txt).

**En el README.md, documenta:**
```markdown
## Installation

### Option 1: Using Conda (Recommended for Belle II)
conda env create -f environment.yml
conda activate qcd-vortex

### Option 2: Using pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Así cada usuario elige su método preferido.
