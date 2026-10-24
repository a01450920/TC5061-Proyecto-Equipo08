"""Configuración centralizada de rutas, parámetros temporales y semillas."""

from pathlib import Path

# Rutas del proyecto

PROJ_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_DB_PATH = PROJ_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
MLFLOW_EXPERIMENT_NAME = "mibici-equipo08"

# Semilla de reproducibilidad
RANDOM_SEED = 42

# Rango de fechas del experimento
TRAIN_START_DATE = "2026-01-01"
TRAIN_END_DATE = "2026-04-30"
TEST_START_DATE = "2026-05-01"
TEST_END_DATE = "2026-05-31"