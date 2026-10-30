import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, recall_score

from mibici.features import (
    preparar_features_p1_demanda,
    preparar_features_p2_riesgo,
    preparar_features_p3_clustering,
)
from mibici.modeling.train import construir_modelo
from mibici.config import RANDOM_SEED


# ============================================================================
# 1. PRUEBAS UNITARIAS DE FEATURE ENGINEERING (P1, P2 Y P3)
# ============================================================================

def test_preparar_features_p1_demanda_caso_normal():
    """P1 Normal: Verifica la generación de características de demanda."""
    data = {
        "inicio_del_viaje": pd.date_range(start="2026-01-01 08:00:00", periods=5, freq="h"),
        "origen_id": [10, 10, 12, 10, 12]
    }
    df_raw = pd.DataFrame(data)
    df_features = preparar_features_p1_demanda(df_raw)

    assert "viajes_salientes" in df_features.columns
    assert "hora" in df_features.columns
    assert "es_fin_semana" in df_features.columns
    assert len(df_features) > 0


def test_preparar_features_p1_demanda_edge_case_vacio():
    """P1 Límite: Verifica el comportamiento ante un DataFrame vacío."""
    df_vacio = pd.DataFrame({"inicio_del_viaje": pd.to_datetime([]), "origen_id": []})
    df_resultado = preparar_features_p1_demanda(df_vacio)

    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 0


def test_preparar_features_p2_riesgo_caso_normal():
    """P2 Normal: Verifica la generación de características de riesgo y balance."""
    data = {
        "inicio_del_viaje": pd.date_range(start="2026-01-01 08:00:00", periods=10, freq="h"),
        "fin_del_viaje": pd.date_range(start="2026-01-01 08:30:00", periods=10, freq="h"),
        "origen_id": [10]*5 + [12]*5,
        "destino_id": [12]*5 + [10]*5,
    }
    df_raw = pd.DataFrame(data)
    df_features = preparar_features_p2_riesgo(df_raw)

    assert "target_riesgo_t1" in df_features.columns
    assert "balance_neto" in df_features.columns


def test_preparar_features_p2_riesgo_edge_case_vacio():
    """P2 Límite: Comprobación de seguridad ante DataFrame sin registros."""
    df_vacio = pd.DataFrame({
        "inicio_del_viaje": pd.to_datetime([]),
        "fin_del_viaje": pd.to_datetime([]),
        "origen_id": [],
        "destino_id": []
    })
    df_resultado = preparar_features_p2_riesgo(df_vacio)

    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 0


def test_preparar_features_p3_clustering_caso_normal():
    """P3 Normal: Verifica la agregación y estructura de matriz de perfiles horarios."""
    data = {
        "inicio_del_viaje": pd.date_range(start="2026-01-01 08:00:00", periods=5, freq="h"),
        "origen_id": [10, 10, 12, 10, 12]
    }
    df_raw = pd.DataFrame(data)
    df_p3 = preparar_features_p3_clustering(df_raw)

    assert "origen_id" in df_p3.columns
    assert len(df_p3) <= 2


def test_preparar_features_p3_clustering_edge_case_vacio():
    """P3 Límite: Respuesta limpia ante entrada vacía."""
    df_vacio = pd.DataFrame({"inicio_del_viaje": pd.to_datetime([]), "origen_id": []})
    df_resultado = preparar_features_p3_clustering(df_vacio)

    assert isinstance(df_resultado, pd.DataFrame)
    assert len(df_resultado) == 0


# ============================================================================
# 2. PRUEBAS UNITARIAS DE REPRODUCIBILIDAD DE MODELOS (P1, P2 Y P3)
# ============================================================================

def test_reproducibilidad_entrenamiento_p1_rf():
    """P1 Reproducibilidad: Comprueba determinismo estricto en RandomForest."""
    np.random.seed(RANDOM_SEED)
    X = pd.DataFrame({
        "origen_id": np.random.randint(1, 10, 100),
        "hora": np.random.randint(0, 24, 100),
        "dia_semana": np.random.randint(0, 7, 100),
        "es_fin_semana": np.random.randint(0, 2, 100)
    })
    y = np.random.randint(1, 50, 100)

    hp = {"n_estimators": 20, "max_depth": 5}

    modelo_1 = construir_modelo("p1", "rf", hp).fit(X, y)
    preds_1 = modelo_1.predict(X)

    modelo_2 = construir_modelo("p1", "rf", hp).fit(X, y)
    preds_2 = modelo_2.predict(X)

    np.testing.assert_array_almost_equal(preds_1, preds_2, decimal=6)


def test_reproducibilidad_entrenamiento_p2_logreg():
    """P2 Reproducibilidad: Comprueba determinismo en Regresión Logística."""
    np.random.seed(RANDOM_SEED)
    X = pd.DataFrame({
        "estacion_id": np.random.randint(1, 10, 100),
        "hora": np.random.randint(0, 24, 100),
        "entradas": np.random.randint(0, 10, 100),
        "salidas": np.random.randint(0, 10, 100),
        "balance_neto": np.random.randint(-5, 5, 100),
        "balance_acumulado": np.random.randint(-10, 10, 100)
    })
    y = np.random.randint(0, 2, 100)

    modelo_1 = construir_modelo("p2", "logreg", {}).fit(X, y)
    preds_1 = modelo_1.predict(X)

    modelo_2 = construir_modelo("p2", "logreg", {}).fit(X, y)
    preds_2 = modelo_2.predict(X)

    np.testing.assert_array_equal(preds_1, preds_2)


def test_reproducibilidad_entrenamiento_p3_kmeans():
    """P3 Reproducibilidad: Comprueba consistencia en asignación de Clusters KMeans."""
    np.random.seed(RANDOM_SEED)
    X = pd.DataFrame(np.random.rand(50, 10))

    hp = {"n_clusters": 3}

    modelo_1 = construir_modelo("p3", "kmeans", hp).fit(X)
    labels_1 = modelo_1.labels_

    modelo_2 = construir_modelo("p3", "kmeans", hp).fit(X)
    labels_2 = modelo_2.labels_

    np.testing.assert_array_equal(labels_1, labels_2)
