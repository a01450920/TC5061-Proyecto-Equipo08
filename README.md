# Proyecto MiBici - Equipo 08

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto colaborativo de MLOps y Machine Learning para el análisis y predicción de la demanda del sistema MiBici utilizando regresión, clasificación y clustering.

## Equipo y roles

| Integrante | Rol principal |
|------------|---------------|
| Ricardo | Data Engineer y Product Owner |
| Iván | MLOps Engineer, GitHub y gestión del repositorio |
| Tania | Data Scientist, modelado y MLflow |
| Jose Antonio | Documentación, ML Canvas y arquitectura |

### Responsabilidades compartidas

Todos los integrantes participan en:
- Revisión de Pull Requests
- Validación de decisiones técnicas
- Discusión de P1, P2 y P3
- Elaboración del entregable final

## Preguntas sobre MiBici

### P1. Predicción de demanda
¿Cuántos viajes se iniciarán en una estación durante un periodo futuro?

Problema de ML:
- Tipo: Regresión
- Modelo candidato: RandomForestRegressor
- Baseline: DummyRegressor

### P2. Riesgo de desbalance
¿Una estación presenta riesgo de quedarse sin bicicletas o sin espacios disponibles?

Problema de ML:
- Tipo: Clasificación
- Modelo candidato: RandomForestClassifier
- Baseline: DummyClassifier

### P3. Perfiles de uso
¿Qué grupos de estaciones presentan patrones horarios similares de utilización?

Problema de ML:
- Tipo: Clustering
- Modelo base: KMeans
- Métrica: Silhouette Score

## Datos

Fuente:
- Datos abiertos de MiBici Guadalajara.

Archivos utilizados:
- datos_abiertos_2026_01.csv
- datos_abiertos_2026_02.csv
- datos_abiertos_2026_03.csv
- datos_abiertos_2026_04.csv
- datos_abiertos_2026_05.csv

Los datos son descargados automáticamente durante la ejecución del pipeline de ingesta.

## Instalación


```bash
python -m venv mlops-env
source mlops-env/bin/activate

pip install -r requirements.txt
```


## Ingesta de datos

```bash
python -m mibici.dataset
```

Este comando:
- Descarga los datos abiertos de MiBici.
- Procesa los archivos CSV.
- Genera `viajes_clean.parquet`.
- Genera `data_manifest.json`.

## Ejecución de experimentos

```bash
python -m mibici.modeling.train
```

Este comando:
- Ejecuta P1 (regresión).
- Ejecuta P2 (clasificación).
- Ejecuta P3 (clustering).
- Registra métricas y artefactos en MLflow.

## Flujo de trabajo Git

El equipo seguirá GitHub Flow.

### Ramas

- main: rama protegida
- feature/ingestion
- feature/features
- feature/mlflow
- feature/docs

### Convención de commits

Se utilizará Conventional Commits:

- feat: nueva funcionalidad
- fix: corrección de errores
- docs: documentación
- refactor: reorganización de código
- test: pruebas
- chore: configuración o mantenimiento

### Pull Requests

Todo cambio debe realizarse en una rama de trabajo y fusionarse mediante Pull Request hacia `main`.


## Organización del proyecto

```text
TC5061-Proyecto-Equipo08
│
├── data/
│   ├── raw/                     <- Datos abiertos originales de MiBici
│   └── processed/               <- Datos procesados y manifiestos
│
├── docs/                        <- Documentación adicional del proyecto
│
├── mibici/                      <- Código fuente principal
│   ├── __init__.py
│   ├── config.py                <- Configuración global
│   ├── dataset.py               <- Ingesta y procesamiento de datos
│   ├── features.py              <- Generación de variables para modelado
│   ├── plots.py                 <- Generación de visualizaciones
│   └── modeling/
│       ├── __init__.py
│       ├── train.py             <- Entrenamiento y registro en MLflow
│       └── predict.py           <- Inferencia y predicciones
│
├── mlruns/                      <- Experimentos registrados por MLflow
│
├── models/                      <- Modelos serializados (cuando aplique)
│
├── notebooks/
│   └── P1_P2_P3_MiBici.ipynb    <- Análisis exploratorio y experimentación
│
├── references/
│   ├── Arquitectura_Equipo8.pdf <- Arquitectura de la solución
│   └── ml_canvas.pdf            <- ML Canvas del proyecto
│
├── reports/
│   └── figures/                 <- Figuras generadas para análisis y reporte
│
├── tests/
│   └── test_data.py             <- Pruebas del pipeline de datos
│
├── .env                         <- Variables de entorno locales
├── .gitignore
├── Makefile
├── pyproject.toml
├── requirements.txt
├── README.md
└── mlflow.db                    <- Base de datos local de tracking MLflow
```

--------
ACTUALIZACIÓN 24OCT26

## Reproducción con MLflow Project

Todas las corridas se lanzan con `mlflow run`, nunca desde un notebook.
Los comandos son para PowerShell, desde la raíz del repositorio.

### 1. Entorno

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`python_env.yaml` y `requirements.txt` (versiones fijas) declaran el entorno del proyecto.
En Windows, `mlflow run` no puede crear ese entorno aislado porque necesita `pyenv`, que no
existe en ese sistema, por eso los comandos usan `--env-manager=local` (el Python del
`.venv` activo). En Linux/macOS se puede omitir esa opción.

### 2. Servidor de tracking (Terminal 1)

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

La interfaz queda en http://127.0.0.1:5000. `mlflow.db` y `mlartifacts/` están en el
`.gitignore` y nunca se suben al repositorio.

### 3. Ingesta y entrenamiento (Terminal 2)

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Ingesta (descarga, limpia y genera viajes_clean.parquet + data_manifest.json)
mlflow run . -e ingest --experiment-name mibici-equipo08 --env-manager=local

# Baselines (DummyRegressor, DummyClassifier y KMeans k=2)
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p1 -P mode=baseline
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p2 -P mode=baseline
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p3 -P mode=baseline

# Búsqueda con nested runs: un padre por pregunta, una hija por combinación,
# y las 3 mejores se reentrenan como candidatas con artefactos completos
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p1 -P mode=search
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p2 -P mode=search
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p3 -P mode=search

# Reproducir una configuración concreta (el modelo elegido de cada pregunta)
mlflow run . -e train --experiment-name mibici-equipo08 --env-manager=local -P question=p1 -P mode=single -P model=rf -P n_estimators=300
```

Parámetros de `train`: `question` (p1|p2|p3), `mode` (baseline|search|single),
`model` (rf|hgb|logreg|all), `n_estimators`, `max_depth` (-1 = sin límite) y `n_clusters`.
Rutas, fechas y semilla viven en `mibici/config.py`.

### 4. Etiquetas para filtrar en la interfaz

Cada corrida lleva `equipo`, `question`, `model_family`, `stage_of_work`
(`baseline`, `search` o `candidate`), `run_role` (`parent` o `child`) y `git_commit`.
Ejemplo de filtro: `tags.question = "p2" and tags.stage_of_work = "candidate"`.

### 5. Artefactos de cada corrida candidata

| Carpeta | Contenido |
|---|---|
| `model/` | Modelo con firma y ejemplo de entrada |
| `plots/` | P1: real vs predicho y residuales. P2: matriz de confusión y curva precisión-recall. P3: perfiles horarios por grupo |
| `data/` | `data_manifest.json` (versión de los datos); P3 también `estaciones_cluster.csv` |
| `config/` | `run_config.json` y `config.py` con los que se ejecutó |

Además se registra el dataset de entrenamiento con `mlflow.log_input`. No se guardan
datos personales de las personas usuarias (solo estaciones, horas y conteos).

### 6. Model Registry y ciclo de vida

```powershell
mlflow run . -e register --experiment-name mibici-equipo08 --env-manager=local
```

Registra la mejor candidata de cada pregunta (elegida con la métrica de **validación**,
no la de prueba) como `mibici-equipo08-p1-demanda`, `mibici-equipo08-p2-riesgo` y
`mibici-equipo08-p3-grupos`, con descripción y linaje (corrida de origen, commit y
versión de datos) en los tags de cada versión.

**Esquema: aliases.** `champion` marca la versión en producción y, opcionalmente,
`challenger` la siguiente mejor. Se eligen aliases porque las etapas (Staging/Production)
están en desuso desde MLflow 2.9, y un alias permite cambiar de versión sin modificar el
código que consume el modelo: `models:/mibici-equipo08-p1-demanda@champion`.

```python
import mlflow
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo08-p1-demanda@champion")
```

El script también genera `reports/tabla_comparativa.csv` para el reporte.
