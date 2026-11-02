"""Validación de datos: reglas de calidad explícitas (pandas) y reportes de DeepChecks.

Qué se valida
-------------
La tabla de demanda de P1 (estación x fecha x hora), construida con
`preparar_features_p1_demanda` a partir de `viajes_clean.parquet`. Solo contiene estación,
fecha, hora y conteo de viajes: no hay datos personales de las personas usuarias.
Si el parquet no existe (por ejemplo, en un clon recién descargado) se usan viajes
sintéticos con el mismo esquema, así que las pruebas corren en cualquier equipo.

Qué hace fallar una prueba y qué solo se reporta
------------------------------------------------
* FALLA: valores faltantes, filas duplicadas en la llave (origen_id, fecha, hora), tipos no
  numéricos, rangos imposibles (hora, día de la semana, conteo < 1) o traslape de fechas entre
  entrenamiento y prueba. Son errores reales de calidad.
* SE REPORTA: lo que detecta DeepChecks (drift, duplicados de variables, etc.). Sus condiciones
  por defecto no aplican a esta tabla: al quitar la fecha, las combinaciones de variables se
  repiten entre días por construcción, y entre enero-abril y mayo es normal que haya drift
  estacional. Los hallazgos quedan documentados en el HTML y en un aviso de pytest.

Reportes de DeepChecks
----------------------
DeepChecks no sobrescribe archivos (crea "nombre (1).html", "nombre (2).html"...), por eso aquí
se borra el reporte anterior antes de guardar. Por defecto se escriben en
`reports/testing/generated/` (ignorada por Git). Para actualizar la evidencia versionada:

    PowerShell:  $env:DEEPCHECKS_GUARDAR_EVIDENCIA = "1"
                 python -m pytest tests/test_data_validation.py -q
"""

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

# --- PARCHES DE COMPATIBILIDAD (NumPy 2.0+ & scikit-learn con DeepChecks) -------------------
# Se aplican antes de importar DeepChecks. El de `get_scorer` solo afecta a chequeos que usan
# un modelo; aquí no se usa ninguno.
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
from mibici.features import preparar_features_p1_demanda  # noqa: E402

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
ETIQUETA = "viajes_salientes"
CATEGORICAS = ["origen_id", "hora", "dia_semana", "es_fin_semana"]
COLUMNAS = CATEGORICAS + [ETIQUETA]
MAX_FILAS_DEEPCHECKS = 50_000


# --------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------
def _viajes_sinteticos(n: int = 12000, estaciones: int = 15) -> pd.DataFrame:
    """Viajes de enero a mayo de 2026 con el esquema mínimo de viajes_clean."""
    rng = np.random.default_rng(RANDOM_SEED)
    inicio = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 151 * 24 * 60, n), unit="m"
    )
    return pd.DataFrame({
        "inicio_del_viaje": inicio,
        "origen_id": rng.integers(1, estaciones + 1, n),
    })


def cargar_tabla_p1() -> tuple[pd.DataFrame, str]:
    """Tabla estación x fecha x hora y de dónde salió (datos reales o sintéticos)."""
    if PARQUET_PATH.exists():
        # Solo las dos columnas necesarias: no se leen columnas personales.
        viajes = pd.read_parquet(PARQUET_PATH, columns=["inicio_del_viaje", "origen_id"])
        origen = "datos reales"
    else:
        viajes = _viajes_sinteticos()
        origen = "datos sintéticos"
    return preparar_features_p1_demanda(viajes), origen


def dividir_train_test(tabla: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Misma partición temporal que el modelado (fechas de config.py)."""
    def dia(texto: str) -> date:
        return pd.to_datetime(texto).date()

    f = tabla["fecha"]
    train = tabla[(f >= dia(TRAIN_START_DATE)) & (f <= dia(TRAIN_END_DATE))]
    test = tabla[(f >= dia(TEST_START_DATE)) & (f <= dia(TEST_END_DATE))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Reglas de calidad explícitas
# --------------------------------------------------------------------------------------------
def validar_integridad(tabla: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas de calidad; vacía si la tabla es válida."""
    requeridas = COLUMNAS + ["fecha"]
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

    no_numericas = [c for c in COLUMNAS if not pd.api.types.is_numeric_dtype(tabla[c])]
    if no_numericas:
        problemas.append(f"Columnas no numéricas: {no_numericas}")
        return problemas  # los rangos no tienen sentido sin tipos numéricos

    if not tabla["hora"].between(0, 23).all():
        problemas.append("Hay horas fuera del rango 0-23")
    if not tabla["dia_semana"].between(0, 6).all():
        problemas.append("Hay días de la semana fuera del rango 0-6")
    if not tabla["es_fin_semana"].isin([0, 1]).all():
        problemas.append("es_fin_semana contiene valores distintos de 0 y 1")
    if (tabla[ETIQUETA] < 1).any():
        problemas.append("Hay conteos de viajes menores a 1")
    return problemas


def validar_separacion_temporal(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Entrenamiento y prueba deben existir y no traslaparse en el tiempo (sin leakage)."""
    if train.empty or test.empty:
        return ["Entrenamiento o prueba están vacíos"]
    if train["fecha"].max() >= test["fecha"].min():
        return ["Las fechas de entrenamiento y prueba se traslapan"]
    return []


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tablas():
    tabla, origen = cargar_tabla_p1()
    train, test = dividir_train_test(tabla)
    return train, test, origen


def _tabla_valida() -> pd.DataFrame:
    return pd.DataFrame({
        "origen_id": [1, 1, 2],
        "fecha": [date(2026, 1, 5)] * 3,
        "hora": [8, 9, 8],
        "dia_semana": [0, 0, 0],
        "es_fin_semana": [0, 0, 0],
        ETIQUETA: [3, 1, 2],
    })


# --------------------------------------------------------------------------------------------
# Pruebas: el dataset del proyecto
# --------------------------------------------------------------------------------------------
def test_tabla_p1_cumple_las_reglas_de_integridad(tablas):
    train, test, origen = tablas
    assert validar_integridad(train) == [], f"Entrenamiento ({origen})"
    assert validar_integridad(test) == [], f"Prueba ({origen})"


def test_train_y_test_no_se_traslapan_en_el_tiempo(tablas):
    train, test, _ = tablas
    assert validar_separacion_temporal(train, test) == []


# --------------------------------------------------------------------------------------------
# Pruebas: las propias reglas detectan problemas (casos límite)
# --------------------------------------------------------------------------------------------
def test_una_tabla_valida_no_tiene_problemas():
    assert validar_integridad(_tabla_valida()) == []


def _con_nulo(t):
    t.loc[0, ETIQUETA] = np.nan
    return t


def _con_llave_duplicada(t):
    return pd.concat([t, t.iloc[[0]]], ignore_index=True)


def _con_hora_imposible(t):
    t.loc[0, "hora"] = 25
    return t


def _con_dia_imposible(t):
    t.loc[0, "dia_semana"] = 7
    return t


def _con_fin_de_semana_invalido(t):
    t.loc[0, "es_fin_semana"] = 2
    return t


def _con_conteo_cero(t):
    t.loc[0, ETIQUETA] = 0
    return t


def _con_texto_en_numerica(t):
    t["hora"] = t["hora"].astype(str)
    return t


@pytest.mark.parametrize(
    "corromper, fragmento",
    [
        (_con_nulo, "faltantes"),
        (_con_llave_duplicada, "duplicadas"),
        (_con_hora_imposible, "horas"),
        (_con_dia_imposible, "días de la semana"),
        (_con_fin_de_semana_invalido, "es_fin_semana"),
        (_con_conteo_cero, "menores a 1"),
        (_con_texto_en_numerica, "no numéricas"),
    ],
)
def test_la_validacion_detecta_problemas(corromper, fragmento):
    problemas = validar_integridad(corromper(_tabla_valida()))
    assert any(fragmento in p for p in problemas), problemas


def test_la_validacion_detecta_columna_faltante():
    problemas = validar_integridad(_tabla_valida().drop(columns=["hora"]))
    assert "Faltan columnas" in problemas[0]


def test_la_validacion_detecta_tabla_vacia():
    assert validar_integridad(_tabla_valida().iloc[0:0]) == ["La tabla está vacía"]


def test_la_validacion_detecta_traslape_temporal():
    train, test = _tabla_valida(), _tabla_valida()  # mismas fechas: traslape
    assert validar_separacion_temporal(train, test) != []


def test_la_validacion_detecta_particiones_vacias():
    assert validar_separacion_temporal(_tabla_valida(), _tabla_valida().iloc[0:0]) != []


# --------------------------------------------------------------------------------------------
# Pruebas: reportes de DeepChecks
# --------------------------------------------------------------------------------------------
def _carpeta_reportes():
    base = REPORTS_DIR / "testing"
    return base if os.getenv("DEEPCHECKS_GUARDAR_EVIDENCIA") == "1" else base / "generated"


def _guardar_reporte(resultado, nombre: str):
    carpeta = _carpeta_reportes()
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    if ruta.exists():
        ruta.unlink()  # DeepChecks no sobrescribe: crearía "nombre (1).html"
    resultado.save_as_html(str(ruta))
    return ruta


def _muestra(df: pd.DataFrame) -> pd.DataFrame:
    """Muestra reproducible para que DeepChecks corra en tiempo razonable."""
    return df[COLUMNAS].sample(n=min(len(df), MAX_FILAS_DEEPCHECKS), random_state=RANDOM_SEED)


def _avisar_si_hay_hallazgos(titulo: str, resultado, ruta) -> None:
    no_superados = resultado.get_not_passed_checks()
    if len(no_superados) > 0:
        warnings.warn(
            f"DeepChecks ({titulo}): {len(no_superados)} chequeo(s) con condiciones no "
            f"superadas. Revisa el reporte: {ruta}",
            stacklevel=2,
        )


def test_deepchecks_integridad_genera_reporte(tablas):
    """Corre la suite data_integrity sobre el entrenamiento y guarda el reporte HTML."""
    pytest.importorskip("deepchecks")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import data_integrity

    train, _, _ = tablas
    dataset = Dataset(_muestra(train), label=ETIQUETA, cat_features=CATEGORICAS)
    resultado = data_integrity().run(dataset)

    ruta = _guardar_reporte(resultado, "deepchecks_data_integrity.html")
    assert ruta.exists() and ruta.stat().st_size > 0
    _avisar_si_hay_hallazgos("integridad", resultado, ruta)


def test_deepchecks_drift_genera_reporte(tablas):
    """Corre train_test_validation (drift entre ene-abr y mayo) y guarda el reporte HTML."""
    pytest.importorskip("deepchecks")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.suites import train_test_validation

    train, test, _ = tablas
    ds_train = Dataset(_muestra(train), label=ETIQUETA, cat_features=CATEGORICAS)
    ds_test = Dataset(_muestra(test), label=ETIQUETA, cat_features=CATEGORICAS)
    resultado = train_test_validation().run(train_dataset=ds_train, test_dataset=ds_test)

    ruta = _guardar_reporte(resultado, "deepchecks_train_test_drift.html")
    assert ruta.exists() and ruta.stat().st_size > 0
    _avisar_si_hay_hallazgos("drift", resultado, ruta)
    