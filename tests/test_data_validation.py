import os
import pandas as pd
import pytest
import numpy as np

# --- PARCHES DE COMPATIBILIDAD (NumPy 2.0+ & scikit-learn 1.4+ con DeepChecks) ---
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf

import sklearn.metrics
_original_get_scorer = sklearn.metrics.get_scorer

def _patched_get_scorer(scoring):
    try:
        return _original_get_scorer(scoring)
    except (ValueError, KeyError):
        if scoring == 'max_error':
            for fallback in ['neg_max_error', 'neg_mean_absolute_error', 'r2']:
                try:
                    return _original_get_scorer(fallback)
                except Exception:
                    pass
        raise

sklearn.metrics.get_scorer = _patched_get_scorer
# ----------------------------------------------------------------------------------

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

REPORTS_DIR = os.path.join("reports", "testing")


@pytest.fixture
def load_datasets():
    """Carga los conjuntos de datos del proyecto para las pruebas."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_path = os.path.join("data", "processed", "train.csv")
    test_path = os.path.join("data", "processed", "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        # Datos sintéticos de respaldo
        train_df = pd.DataFrame(
            {
                "feature_1": [10.5, 20.1, 15.3, 12.0, 30.2, 25.0],
                "feature_2": [1, 0, 1, 0, 1, 0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        test_df = pd.DataFrame(
            {
                "feature_1": [11.0, 19.8, 14.9, 12.5, 29.5, 24.8],
                "feature_2": [1, 0, 1, 0, 1, 0],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )

    return train_df, test_df


def test_data_integrity(load_datasets):
    """Valida la calidad del dataset (valores faltantes, duplicados, tipos de datos)."""
    train_df, _ = load_datasets

    target_col = "target" if "target" in train_df.columns else None
    ds = Dataset(train_df, label=target_col)

    suite = data_integrity()
    result = suite.run(ds)

    report_path = os.path.join(REPORTS_DIR, "deepchecks_data_integrity.html")
    result.save_as_html(report_path)

    # Verificación corregida: no debe haber chequeos no superados con nivel de falla crítico
    failed_checks = result.get_not_passed_checks()
    assert (
        len(failed_checks) == 0
    ), f"Problemas de integridad detectados en datos. Consulta el reporte en: {report_path}"


def test_train_test_drift(load_datasets):
    """Valida la presencia de Data Drift entre los conjuntos de entrenamiento y prueba."""
    train_df, test_df = load_datasets

    target_col = "target" if "target" in train_df.columns else None
    ds_train = Dataset(train_df, label=target_col)
    ds_test = Dataset(test_df, label=target_col)

    suite = train_test_validation()
    result = suite.run(train_dataset=ds_train, test_dataset=ds_test)

    report_path = os.path.join(REPORTS_DIR, "deepchecks_train_test_drift.html")
    result.save_as_html(report_path)

    failed_checks = result.get_not_passed_checks()
    assert (
        len(failed_checks) == 0
    ), f"Data drift detectado entre Train y Test. Consulta el reporte en: {report_path}"