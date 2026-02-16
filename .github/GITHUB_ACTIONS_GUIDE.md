# 🤖 Guía de GitHub Actions

## 📁 Archivos Creados

```
.github/
├── workflows/
│   ├── tests.yml       ← Tests automáticos (CI)
│   ├── lint.yml        ← Linting y formateo
│   ├── docs.yml        ← Construcción de docs
│   └── publish.yml     ← Publicación en PyPI
└── dependabot.yml      ← Actualizaciones automáticas
```

## 🚀 Cómo Activar GitHub Actions

### Paso 1: Subir los Archivos

```bash
# Añadir archivos a git
git add .github/

# Commit
git commit -m "Add GitHub Actions workflows"

# Push a GitHub
git push origin main
```

### Paso 2: Verificar en GitHub

1. Ve a tu repositorio en GitHub
2. Click en la pestaña **"Actions"**
3. Deberías ver los workflows listados
4. Se ejecutarán automáticamente en el próximo push

## 🔧 Configurar Secrets (Para publish.yml)

Para publicar en PyPI necesitas configurar secrets:

### Paso 1: Obtener Token de PyPI

1. Ve a https://pypi.org/manage/account/token/
2. Crea un nuevo API token
3. Copia el token

### Paso 2: Añadir Secret en GitHub

1. Ve a tu repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: [pega tu token de PyPI]
5. Click "Add secret"

Repite para `TEST_PYPI_API_TOKEN` si quieres usar Test PyPI.

## 📋 Descripción de Workflows

### tests.yml - Tests Automáticos

**Se ejecuta cuando:**
- Push a main o develop
- Pull request
- Manualmente desde Actions tab

**Qué hace:**
- Ejecuta tests en Python 3.10 y 3.11
- Ejecuta en Linux, macOS y Windows
- Genera reporte de cobertura
- Sube cobertura a Codecov

**Ver resultados:**
- Actions tab → Tests workflow

### lint.yml - Calidad de Código

**Se ejecuta cuando:**
- Push a main o develop
- Pull request

**Qué hace:**
- Verifica formateo con black
- Verifica imports con isort
- Ejecuta flake8
- Type checking con mypy

**Si falla:** Revisa el código y ejecuta localmente:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### docs.yml - Documentación

**Se ejecuta cuando:**
- Push a main
- Manualmente

**Qué hace:**
- Construye documentación con Sphinx
- Publica en GitHub Pages

**Ver docs:** `https://tu-usuario.github.io/tu-repo/`

### publish.yml - Publicación PyPI

**Se ejecuta cuando:**
- Creas un nuevo release en GitHub
- Manualmente (para Test PyPI)

**Qué hace:**
- Construye el paquete
- Verifica el paquete
- Publica en PyPI/Test PyPI

**Crear release:**
1. GitHub → Releases → Create new release
2. Tag version: v1.0.0
3. Title: Release 1.0.0
4. Description: [changelog]
5. Publish release → Workflow se ejecuta automáticamente

### dependabot.yml - Actualizaciones

**Qué hace:**
- Revisa dependencias semanalmente
- Crea PRs automáticos con actualizaciones
- Mantiene GitHub Actions actualizados

**Configurar:**
1. Reemplaza "tu-usuario-github" con tu usuario
2. GitHub detecta automáticamente el archivo
3. Empezará a crear PRs de actualizaciones

## 🎯 Badges para README.md

Añade badges a tu README.md:

```markdown
# QCD Vortex Entanglement

![Tests](https://github.com/tu-usuario/tu-repo/workflows/Tests/badge.svg)
![Lint](https://github.com/tu-usuario/tu-repo/workflows/Lint/badge.svg)
![Docs](https://github.com/tu-usuario/tu-repo/workflows/Documentation/badge.svg)
[![codecov](https://codecov.io/gh/tu-usuario/tu-repo/branch/main/graph/badge.svg)](https://codecov.io/gh/tu-usuario/tu-repo)
[![PyPI version](https://badge.fury.io/py/qcd-vortex-entanglement.svg)](https://pypi.org/project/qcd-vortex-entanglement/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

...
```

## 🔍 Ver Logs de Ejecución

1. Actions tab en GitHub
2. Click en un workflow run
3. Click en un job
4. Ver logs detallados

## ⚙️ Personalizar Workflows

### Cambiar triggers

```yaml
on:
  push:
    branches: [ main, develop, feature/* ]  # Múltiples branches
  pull_request:
  schedule:
    - cron: '0 0 * * 0'  # Cada domingo a medianoche
```

### Añadir más versiones de Python

```yaml
matrix:
  python-version: ['3.10', '3.11', '3.12']
```

### Ejecutar solo tests específicos

```yaml
- name: Run fast tests
  run: pytest tests/ -m "not slow"
```

### Cachear dependencias

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

## 🐛 Troubleshooting

### Problema: Workflow no se ejecuta

**Solución:**
- Verifica que el archivo esté en `.github/workflows/`
- Verifica la sintaxis YAML (usa yamllint)
- Verifica que el branch tenga push

### Problema: Tests fallan en CI pero pasan localmente

**Posibles causas:**
- Diferencias en entorno (paths, variables)
- Falta alguna dependencia
- Tests dependientes del sistema operativo

**Solución:**
```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y tu-dependencia
```

### Problema: Secrets no funcionan

**Solución:**
- Verifica que el secret esté configurado en Settings
- Verifica el nombre del secret (case-sensitive)
- Los secrets solo funcionan en branches protegidos para PRs de forks

### Problema: Workflow muy lento

**Optimizaciones:**
```yaml
# 1. Usar cache
- uses: actions/cache@v3

# 2. Instalar solo lo necesario
pip install -e ".[test]"  # No instalar todo

# 3. Paralelizar
strategy:
  matrix:
    shard: [1, 2, 3, 4]
```

## 📊 Monitoreo

### Ver histórico

Actions tab → Workflow → Ver todos los runs

### Notificaciones

Settings → Notifications → Actions → Configurar

### Insights

Actions tab → Ver métricas de uso

## 🎓 Recursos

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Marketplace**: https://github.com/marketplace?type=actions
- **Ejemplos**: https://github.com/actions/starter-workflows
- **Sintaxis**: https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions

## 💡 Mejores Prácticas

1. **Usar cache** para dependencias
2. **Paralelizar** tests cuando sea posible
3. **Fail fast** para detectar errores rápido
4. **Versionar** workflows (keep them in git)
5. **Documentar** cambios en workflows
6. **Monitorear** uso de minutos (cuenta gratuita tiene límite)

## 🚦 Status Checks

Habilitar checks obligatorios antes de merge:

1. Settings → Branches
2. Selecciona branch (ej: main)
3. "Require status checks to pass before merging"
4. Selecciona: Tests, Lint
5. Save

Ahora los PRs necesitan pasar tests antes de merge.

---

**Última actualización:** 2026-02-16
