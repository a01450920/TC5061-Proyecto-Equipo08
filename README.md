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
ACTUALIZACIÓN 21OCT26

## Reproducción del Proyecto con MLflow Project

## Reproducción con MLflow Project

1. Levantar servidor local de tracking (en una Terminal 1):
   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
   Para enlazar las ejecuciones de mlflow run directamente con el servidor activo, configura la variable de entorno en PowerShell:
   $env:MLFLOW_TRACKING_URI="[http://127.0.0.1:5000](http://127.0.0.1:5000)"

2. Ejecutar Ingesta de datos (en una Terminal 2):
   mlflow run . -e ingest --env-manager=local

3. Ejecutar Entrenamiento completo (P1, P2 y P3):
   mlflow run . -e train -P question=all -P n_estimators=300 -P run_version=v1 --env-manager=local

4. Ejecutar Entrenamiento por pregunta individual:
   - P1: mlflow run . -e train -P question=p1 -P n_estimators=300 -P max_depth=12 --env-manager=local
   - P2: mlflow run . -e train -P question=p2 -P n_estimators=100 -P max_depth=8 --env-manager=local
   - P3: mlflow run . -e train -P question=p3 -P n_clusters=4 --env-manager=local
