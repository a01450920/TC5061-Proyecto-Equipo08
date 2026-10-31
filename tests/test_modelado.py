"""Pruebas unitarias del modelado: rejilla, elección de candidatas y particiones (train.py),
y utilidades del Model Registry (register.py)."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling import register, train


# ---------------------------------------------------------------------------
# _profundidad: -1 / 0 / None significan "sin límite" (scikit-learn no acepta -1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entrada, esperado",
    [(None, None), (-1, None), (0, None), (10, 10), (20, 20)],
)
def test_profundidad(entrada, esperado):
    assert train._profundidad(entrada) == esperado


def test_max_depth_menos_uno_llega_como_sin_limite_al_modelo():
    modelo = train.construir_modelo("p1", "rf", {"n_estimators": 10, "max_depth": -1})
    assert modelo.max_depth is None


# ---------------------------------------------------------------------------
# rejilla
# ---------------------------------------------------------------------------
def test_rejilla_p3_prueba_k_de_2_a_6():
    ks = [hp["n_clusters"] for hp in train.rejilla("p3", "kmeans", 300)]
    assert ks == [2, 3, 4, 5, 6]


def test_rejilla_logreg_no_tiene_hiperparametros():
    assert train.rejilla("p2", "logreg", 300) == [{}]


@pytest.mark.parametrize("familia", ["rf", "hgb"])
def test_rejilla_arboles_combina_estimadores_y_profundidades(familia):
    rej = train.rejilla("p1", familia, 300)
    assert len(rej) == 6  # {100, 300} estimadores x 3 profundidades
    assert {hp["n_estimators"] for hp in rej} == {100, 300}
    assert None in {hp["max_depth"] for hp in rej}


def test_rejilla_no_repite_combinaciones_si_n_estimators_es_100():
    """Caso límite: con n_estimators=100 los dos valores coinciden y no se duplican."""
    rej = train.rejilla("p1", "rf", 100)
    assert len(rej) == 3
    assert len({(hp["n_estimators"], hp["max_depth"]) for hp in rej}) == 3


# ---------------------------------------------------------------------------
# nombre_corrida
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pregunta, familia, hp, esperado",
    [
        ("p1", "rf", {"n_estimators": 100, "max_depth": None}, "p1_rf_n100_dNone"),
        ("p2", "logreg", {}, "p2_logreg"),
        ("p3", "kmeans", {"n_clusters": 4}, "p3_kmeans_k4"),
    ],
)
def test_nombre_corrida(pregunta, familia, hp, esperado):
    assert train.nombre_corrida(pregunta, familia, hp) == esperado


# ---------------------------------------------------------------------------
# elegir_candidatas
# ---------------------------------------------------------------------------
def _res(familia, valor, n=1):
    return {"familia": familia, "hp": {"n_estimators": n}, "valor": valor, "run_id": f"{familia}{valor}"}


def test_candidatas_p1_minimiza_y_toma_la_mejor_de_cada_familia():
    resultados = [_res("rf", 2.0), _res("rf", 1.0), _res("rf", 1.5), _res("hgb", 1.8), _res("hgb", 1.2)]
    elegidas = train.elegir_candidatas("p1", resultados)

    assert [r["valor"] for r in elegidas] == [1.0, 1.2, 1.5]  # MAE: menor es mejor
    assert {"rf", "hgb"} <= {r["familia"] for r in elegidas}


def test_candidatas_p2_maximiza_recall():
    resultados = [_res("rf", 0.60), _res("hgb", 0.80), _res("logreg", 0.70), _res("rf", 0.65)]
    elegidas = train.elegir_candidatas("p2", resultados)

    assert [r["valor"] for r in elegidas] == [0.80, 0.70, 0.65]  # recall: mayor es mejor
    assert {r["familia"] for r in elegidas} == {"hgb", "logreg", "rf"}


def test_candidatas_no_pasan_del_maximo():
    """Caso límite: 4 familias distintas, pero solo se devuelven N_CANDIDATAS."""
    resultados = [_res(f, v) for f, v in [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]]
    elegidas = train.elegir_candidatas("p2", resultados)

    assert len(elegidas) == train.N_CANDIDATAS
    assert [r["valor"] for r in elegidas] == [0.9, 0.8, 0.7]


def test_candidatas_con_menos_resultados_que_el_maximo():
    """Caso límite: solo hay 2 combinaciones; devuelve esas 2 sin fallar."""
    elegidas = train.elegir_candidatas("p3", [_res("kmeans", 0.5), _res("kmeans", 0.4)])
    assert len(elegidas) == 2


def test_metrica_de_seleccion_ligada_al_negocio():
    """P1 minimiza MAE, P2 maximiza recall, P3 maximiza silhouette; P1 y P2 usan validación."""
    assert train.METRICA_SELECCION["p1"] == ("val_mae", "min")
    assert train.METRICA_SELECCION["p2"] == ("val_recall", "max")
    assert train.METRICA_SELECCION["p3"] == ("silhouette_score", "max")


# ---------------------------------------------------------------------------
# construir_modelo: combinaciones válidas e inválidas
# ---------------------------------------------------------------------------
def test_construir_modelo_combinacion_no_soportada():
    with pytest.raises(ValueError):
        train.construir_modelo("p1", "logreg", {})


def test_baselines_son_dummy():
    assert type(train.construir_modelo("p1", "dummy", {})).__name__ == "DummyRegressor"
    assert type(train.construir_modelo("p2", "dummy", {})).__name__ == "DummyClassifier"


def test_p2_balancea_clases():
    """P2 tiene pocos casos de riesgo: el modelo debe compensar el desbalance."""
    assert train.construir_modelo("p2", "rf", {"n_estimators": 5}).class_weight == "balanced"


# ---------------------------------------------------------------------------
# cargar_datos: particiones temporales sin traslape (sin data leakage)
# ---------------------------------------------------------------------------
def _viajes_sinteticos(inicio="2026-01-01", fin="2026-05-31", n=6000, estaciones=8):
    rng = np.random.default_rng(0)
    dias = (pd.Timestamp(fin) - pd.Timestamp(inicio)).days + 1
    ini = pd.Timestamp(inicio) + pd.to_timedelta(rng.integers(0, dias * 24 * 60, n), unit="m")
    return pd.DataFrame({
        "inicio_del_viaje": ini,
        "fin_del_viaje": ini + pd.to_timedelta(rng.integers(3, 40, n), unit="m"),
        "origen_id": rng.integers(1, estaciones + 1, n),
        "destino_id": rng.integers(1, estaciones + 1, n),
    })


def _simular_parquet(monkeypatch, tmp_path, df):
    falso = tmp_path / "viajes_clean.parquet"
    falso.write_bytes(b"")
    monkeypatch.setattr(train, "PARQUET_PATH", falso)
    monkeypatch.setattr(train.pd, "read_parquet", lambda ruta: df)


@pytest.mark.parametrize("pregunta", ["p1", "p2"])
def test_particiones_validacion_y_entrenamiento_son_complementarias(monkeypatch, tmp_path, pregunta):
    _simular_parquet(monkeypatch, tmp_path, _viajes_sinteticos())
    d = train.cargar_datos(pregunta)

    # ene-mar (sub) + abril (val) = ene-abr (train); mayo (test) queda aparte
    assert len(d.X_sub) + len(d.X_val) == len(d.X_train)
    assert len(d.X_test) > 0
    assert len(d.X_sub) == len(d.y_sub) and len(d.X_test) == len(d.y_test)
    assert all(dt == "float64" for dt in d.X_train.dtypes)  # evita avisos de firma en MLflow


def test_p2_target_es_binario_entero(monkeypatch, tmp_path):
    _simular_parquet(monkeypatch, tmp_path, _viajes_sinteticos())
    d = train.cargar_datos("p2")
    assert set(d.y_train.unique()) <= {0, 1}


def test_p3_devuelve_matriz_de_perfiles(monkeypatch, tmp_path):
    _simular_parquet(monkeypatch, tmp_path, _viajes_sinteticos())
    d = train.cargar_datos("p3")

    assert len(d.X_all) == len(d.estaciones)
    assert "origen_id" not in d.X_all.columns
    assert np.allclose(d.X_all.sum(axis=1), 100.0)  # cada fila es un perfil porcentual


def test_particion_vacia_lanza_error(monkeypatch, tmp_path):
    """Caso límite: sin datos de abril no hay conjunto de validación."""
    solo_enero = _viajes_sinteticos(inicio="2026-01-01", fin="2026-01-31")
    _simular_parquet(monkeypatch, tmp_path, solo_enero)
    with pytest.raises(ValueError):
        train.cargar_datos("p1")


def test_falta_el_parquet_da_mensaje_claro(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "PARQUET_PATH", tmp_path / "no_existe.parquet")
    with pytest.raises(FileNotFoundError):
        train.cargar_datos("p1")


# ---------------------------------------------------------------------------
# obtener_version_datos: versión de los datos a partir del manifiesto
# ---------------------------------------------------------------------------
def test_version_datos_toma_los_primeros_12_caracteres(monkeypatch, tmp_path):
    manifiesto = tmp_path / "data_manifest.json"
    manifiesto.write_text('{"hash_sha256": "abcdef0123456789abcdef"}', encoding="utf-8")
    monkeypatch.setattr(train, "MANIFEST_PATH", manifiesto)
    assert train.obtener_version_datos() == "abcdef012345"


def test_version_datos_sin_manifiesto(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "MANIFEST_PATH", tmp_path / "no_existe.json")
    assert train.obtener_version_datos() == "desconocido"


@pytest.mark.parametrize("contenido", ["esto no es json", "[]", '{"hash_sha256": 123}', ""])
def test_version_datos_con_manifiesto_danado_no_rompe_el_entrenamiento(monkeypatch, tmp_path, contenido):
    """Caso límite: un manifiesto corrupto debe dar 'desconocido', no una excepción."""
    manifiesto = tmp_path / "data_manifest.json"
    manifiesto.write_text(contenido, encoding="utf-8")
    monkeypatch.setattr(train, "MANIFEST_PATH", manifiesto)
    assert train.obtener_version_datos() == "desconocido"


# ---------------------------------------------------------------------------
# register.py: nombres del registro y cálculo de mejora
# ---------------------------------------------------------------------------
def test_nombres_del_model_registry():
    assert {k: v["nombre"] for k, v in register.MODELOS.items()} == {
        "p1": "mibici-equipo08-p1-demanda",
        "p2": "mibici-equipo08-p2-riesgo",
        "p3": "mibici-equipo08-p3-grupos",
    }


def test_register_selecciona_con_validacion_no_con_prueba():
    """La métrica de selección de P1 y P2 es de validación: el test no interviene."""
    assert register.MODELOS["p1"]["seleccion"].startswith("val_")
    assert register.MODELOS["p2"]["seleccion"].startswith("val_")


@pytest.mark.parametrize(
    "menor_es_mejor, base, cand, esperado",
    [
        (True, 1.0, 0.8, "+0.2000 (+20.0%)"),    # MAE baja de 1.0 a 0.8: mejora
        (False, 0.5, 0.75, "+0.2500 (+50.0%)"),  # recall sube de 0.5 a 0.75: mejora
        (True, 1.0, 1.2, "-0.2000 (-20.0%)"),    # MAE sube: empeora
    ],
)
def test_mejora_respecto_al_baseline(menor_es_mejor, base, cand, esperado):
    assert register.mejora(menor_es_mejor, base, cand) == esperado


def test_mejora_con_baseline_cero_no_divide():
    """Caso límite: baseline = 0 no debe producir división entre cero."""
    assert register.mejora(False, 0.0, 0.3) == "+0.3000"