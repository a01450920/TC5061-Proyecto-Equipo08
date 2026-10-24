"""Configuración centralizada de rutas, parámetros temporales, semillas y MLflow."""

import os
from pathlib import Path

# Rutas del proyecto
PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Semilla de reproducibilidad
RANDOM_SEED = 42

# Rango de fechas del experimento (partición temporal, igual que el baseline)
TRAIN_START_DATE = "2026-01-01"
TRAIN_END_DATE = "2026-04-30"
TEST_START_DATE = "2026-05-01"
TEST_END_DATE = "2026-05-31"

# Validación: último mes del periodo de entrenamiento. La búsqueda de
# hiperparámetros entrena con ene-mar y evalúa en abril, así el test (mayo)
# no interviene en la selección del modelo.
VAL_START_DATE = "2026-04-01"

# MLflow: el servidor se levanta con
#   mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo08")
