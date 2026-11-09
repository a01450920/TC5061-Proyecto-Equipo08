# Validación de datos: reglas de calidad explícitas (pandas) y reportes de DeepChecks para P1, P2 y P3.

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

# --- PARCHES DE COMPATIBILIDAD (NumPy 2.0+ & scikit-learn con DeepChecks) -------------------
if not hasattr(np, "Inf"):
    np.Inf = np.inf
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

import sklearn.metrics  # noqa: E402

_original_get_scorer = sklearn.metrics.get_scorer


def _patched_get_scorer(scoring):
    try:
        return _original_get_scorer(scoring)
    except (ValueError, KeyError):
        if scoring == "max_error":
            for fallback in ["neg_max_error", "neg_mean_absolute_error", "r2"]:
                try:
                    return _original_get_scorer(fallback)
                except Exception:
                    pass
        raise


sklearn.metrics.get_scorer = _patched_get_scorer
# ---------------------------------------------------------------------------------------------

from mibici.config import (  # noqa: E402
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from mibici.features import (  # noqa: E402
    preparar_features_p1_demanda,
    preparar_features_p2_riesgo,
    preparar_features_p3_clustering,
)

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
MAX_FILAS_DEEPCHECKS = 50_000


# --------------------------------------------------------------------------------------------
# Datos e Ingesta Sintética/Real
# --------------------------------------------------------------------------------------------
def _viajes_sinteticos(n: int = 12000, estaciones: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    inicio = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 151 * 24 * 60, n), unit="m"
    )
    fin = inicio + pd.to_timedelta(rng.integers(1, 60, n), unit="m")
    return pd.DataFrame({
        "inicio_del_viaje": inicio,
        "fin_del_viaje": fin,
        "origen_id": rng.integers(1, estaciones + 1, n),
        "destino_id": rng.integers(1, estaciones + 1, n),
    })


def cargar_viajes_base() -> tuple[pd.DataFrame, str]:
    if PARQUET_PATH.exists():
        viajes = pd.read_parquet(PARQUET_PATH)
        origen = "datos reales"
    else:
        viajes = _viajes_sinteticos()
        origen = "datos sintéticos"
    return viajes, origen


def dividir_train_test(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def dia(texto: str) -> date:
        return pd.to_datetime(texto).date()

    if "fecha" in tabla.columns:
        f = pd.to_datetime(tabla["fecha"]).dt.date
    elif "inicio_del_viaje" in tabla.columns:
        f = pd.to_datetime(tabla["inicio_del_viaje"]).dt.date
    else:
        corte = int(len(tabla) * 0.8)
        return tabla.iloc[:corte].reset_index(drop=True), tabla.iloc[corte:].reset_index(drop=True)

    train = tabla[(f >= dia(TRAIN_START_DATE)) & (f <= dia(TRAIN_END_DATE))]
    test = tabla[(f >= dia(TEST_START_DATE)) & (f <= dia(TEST_END_DATE))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Reglas de calidad explícitas (Pandas)
# --------------------------------------------------------------------------------------------
def validar_integridad_p1(tabla: pd.DataFrame) -> list[str]:
    requeridas = ["origen_id", "hora", "dia_semana", "es_fin_semana", "viajes_salientes", "fecha"]
    faltantes = [c for c in requeridas if c not in tabla.columns]
    if faltantes:
        return [f"Faltan columnas: {faltantes}"]
    if tabla.empty:
        return ["La tabla está vacía"]

    problemas = []
    nulos = tabla[requeridas].isna().sum()
    if nulos.any():
        problemas.append(f"Valores faltantes: {nulos[nulos > 0].to_dict()}")
    if tabla.duplicated(subset=["origen_id", "fecha", "hora"]).any():
        problemas.append("Hay filas duplicadas en la llave (origen_id, fecha, hora)")

    no_numericas = [c for c in ["origen_id", "hora", "dia_semana", "es_fin_semana", "viajes_salientes"] if not pd.api.types.is_numeric_dtype(tabla[c])]
    if no_numericas:
        problemas.append(f"Columnas no numéricas: {no_numericas}")
        return problemas

    if not tabla["hora"].between(0, 23).all():
        problemas.append("Hay horas fuera del rango 0-23")
    if not tabla["dia_semana"].between(0, 6).all():
        problemas.append("Hay días de la semana fuera del rango 0-6")
    if not tabla["es_fin_semana"].isin([0, 1]).all():
        problemas.append("es_fin_semana contiene valores distintos de 0 y 1")
    if (tabla["viajes_salientes"] < 1).any():
        problemas.append("Hay conteos de viajes menores a 1")
    return problemas


def validar_separacion_temporal(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    if train.empty or test.empty:
        return ["Entrenamiento o prueba están vacíos"]
    if "fecha" in train.columns and "fecha" in test.columns:
        if train["fecha"].max() >= test["fecha"].min():
            return ["Las fechas de entrenamiento y prueba se traslapan"]
    return []


# --------------------------------------------------------------------------------------------
# Reportes de DeepChecks
# --------------------------------------------------------------------------------------------
def _carpeta_reportes():
    base = REPORTS_DIR / "testing"
    return base if os.getenv("DEEPCHECKS_GUARDAR_EVIDENCIA") == "1" else base / "generated"


def _guardar_reporte(resultado, nombre: str):
    carpeta = _carpeta_reportes()
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    if ruta.exists():
        ruta.unlink()
    resultado.save_as_html(str(ruta))
    return ruta


def _avisar_si_hay_hallazgos(pregunta: str, titulo: str, resultado, ruta) -> None:
    no_superados = resultado.get_not_passed_checks()
    if len(no_superados) > 0:
        detalles = [f"  - Chequeo: {getattr(c, 'check_name', str(c))}" for c in no_superados]
        detalles_str = "\n".join(detalles)
        mensaje = f"\n[DeepChecks Warning - Pregunta {pregunta.upper()} | {titulo}]: {len(no_superados)} chequeo(s) no superado(s).\n{detalles_str}\nReporte completo: {ruta}"
        warnings.warn(mensaje, stacklevel=2)


# --------------------------------------------------------------------------------------------
# Pruebas con PyTest
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def viajes_base():
    return cargar_viajes_base()


def test_tabla_p1_cumple_las_reglas_de_integridad(viajes_base):
    viajes, origen = viajes_base
    p1_df = preparar_features_p1_demanda(viajes)
    train, test = dividir_train_test(p1_df)
    assert validar_integridad_p1(train) == [], f"Entrenamiento ({origen})"
    assert validar_integridad_p1(test) == [], f"Prueba ({origen})"


def test_train_y_test_no_se_traslapan_en_el_tiempo(viajes_base):
    viajes, _ = viajes_base
    p1_df = preparar_features_p1_demanda(viajes)
    train, test = dividir_train_test(p1_df)
    assert validar_separacion_temporal(train, test) == []


@pytest.mark.parametrize("pregunta, func_prep, label_col, cat_cols", [
    ("p1", preparar_features_p1_demanda, "viajes_salientes", ["origen_id", "hora", "dia_semana", "es_fin_semana"]),
    ("p2", preparar_features_p2_riesgo, "alto_riesgo", ["estacion_id", "hora"]),
    ("p3", preparar_features_p3_clustering, None, ["estacion_id"]),
])
def test_deepchecks_integridad_todas_las_preguntas(viajes_base, pregunta, func_prep, label_col, cat_cols):
    pytest.importorskip("deepchecks")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity

    viajes, _ = viajes_base
    df = func_prep(viajes)
    
    cols_eval = [c for c in df.columns if c not in ["fecha", "inicio_del_viaje", "fin_del_viaje"]]
    df_sample = df[cols_eval].sample(n=min(len(df), MAX_FILAS_DEEPCHECKS), random_state=RANDOM_SEED)
    
    cats = [c for c in cat_cols if c in df_sample.columns]
    
    dataset = Dataset(df_sample, label=label_col if label_col in df_sample.columns else None, cat_features=cats)
    resultado = data_integrity().run(dataset)

    ruta = _guardar_reporte(resultado, f"deepchecks_data_integrity_{pregunta}.html")
    assert ruta.exists() and ruta.stat().st_size > 0
    _avisar_si_hay_hallazgos(pregunta, "integridad", resultado, ruta)


@pytest.mark.parametrize("pregunta, func_prep, label_col, cat_cols", [
    ("p1", preparar_features_p1_demanda, "viajes_salientes", ["origen_id", "hora", "dia_semana", "es_fin_semana"]),
    ("p2", preparar_features_p2_riesgo, "alto_riesgo", ["estacion_id", "hora"]),
    ("p3", preparar_features_p3_clustering, None, ["estacion_id"]),
])
def test_deepchecks_drift_todas_las_preguntas(viajes_base, pregunta, func_prep, label_col, cat_cols):
    pytest.importorskip("deepchecks")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import train_test_validation

    viajes, _ = viajes_base
    df = func_prep(viajes)
    
    train_df, test_df = dividir_train_test(df)
    
    cols_eval = [c for c in df.columns if c not in ["fecha", "inicio_del_viaje", "fin_del_viaje"]]
    train_sample = train_df[cols_eval].sample(n=min(len(train_df), MAX_FILAS_DEEPCHECKS), random_state=RANDOM_SEED)
    test_sample = test_df[cols_eval].sample(n=min(len(test_df), MAX_FILAS_DEEPCHECKS), random_state=RANDOM_SEED)
    
    cats = [c for c in cat_cols if c in cols_eval]
    
    ds_train = Dataset(train_sample, label=label_col if label_col in cols_eval else None, cat_features=cats)
    ds_test = Dataset(test_sample, label=label_col if label_col in cols_eval else None, cat_features=cats)
    
    resultado = train_test_validation().run(train_dataset=ds_train, test_dataset=ds_test)

    ruta = _guardar_reporte(resultado, f"deepchecks_train_test_drift_{pregunta}.html")
    assert ruta.exists() and ruta.stat().st_size > 0
    _avisar_si_hay_hallazgos(pregunta, "drift", resultado, ruta)