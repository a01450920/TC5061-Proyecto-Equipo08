"""Pruebas unitarias de la ingesta (mibici/dataset.py): limpieza de viajes y hash de datos."""

import hashlib

import pandas as pd
import pytest

from mibici.dataset import generar_hash_sha256, limpiar_y_cargar_viajes

COLUMNAS = [
    "Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento",
    "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id",
]
INICIO = pd.Timestamp("2026-01-05 08:00:00")


def _viaje(viaje_id, minutos, origen=1, destino=2, nacimiento=1990):
    """Una fila de CSV con la duración (en minutos) y el año de nacimiento indicados."""
    fin = INICIO + pd.Timedelta(minutes=minutos)
    return [viaje_id, 100 + viaje_id, "M", nacimiento, str(INICIO), str(fin), origen, destino]


def _escribir_csv(ruta, filas, encoding="utf-8"):
    pd.DataFrame(filas, columns=COLUMNAS).to_csv(ruta, index=False, encoding=encoding)
    return ruta


# ---------------------------------------------------------------------------
# limpiar_y_cargar_viajes
# ---------------------------------------------------------------------------
def test_limpieza_conserva_un_viaje_normal_y_calcula_columnas(tmp_path):
    csv = _escribir_csv(tmp_path / "a.csv", [_viaje(1, minutos=10, nacimiento=1990)])
    df = limpiar_y_cargar_viajes([csv])

    assert len(df) == 1
    assert df.loc[0, "duracion_minutos"] == pytest.approx(10.0)
    assert df.loc[0, "edad"] == 36  # edad al año del viaje (2026 - 1990), sin leakage
    assert {"viaje_id", "origen_id", "destino_id", "inicio_del_viaje"} <= set(df.columns)


def test_limpieza_elimina_anomalias(tmp_path):
    filas = [
        _viaje(1, minutos=10),                       # normal: se conserva
        _viaje(2, minutos=0.5, origen=3, destino=3), # viaje cancelado
        _viaje(3, minutos=0.5),                      # demasiado corto
        _viaje(4, minutos=300),                      # demasiado largo
        _viaje(5, minutos=10, nacimiento=1900),      # edad de 126 años
    ]
    df = limpiar_y_cargar_viajes([_escribir_csv(tmp_path / "a.csv", filas)])

    assert df["viaje_id"].tolist() == [1]


@pytest.mark.parametrize("minutos", [1.0, 240.0])
def test_limpieza_conserva_los_limites_de_duracion(tmp_path, minutos):
    """Caso límite: 1 y 240 minutos exactos NO son anomalía (los cortes son < 1 y > 240)."""
    df = limpiar_y_cargar_viajes([_escribir_csv(tmp_path / "a.csv", [_viaje(1, minutos)])])
    assert len(df) == 1


def test_limpieza_une_varios_archivos(tmp_path):
    a = _escribir_csv(tmp_path / "a.csv", [_viaje(1, 10)])
    b = _escribir_csv(tmp_path / "b.csv", [_viaje(2, 20)])
    df = limpiar_y_cargar_viajes([a, b])

    assert sorted(df["viaje_id"]) == [1, 2]
    assert df.index.tolist() == [0, 1]  # índice reiniciado


def test_limpieza_lee_archivos_en_latin1(tmp_path):
    """Caso límite: si el CSV no es UTF-8 se reintenta con latin-1."""
    csv = _escribir_csv(tmp_path / "latin.csv", [_viaje(1, 10)], encoding="latin-1")
    df = limpiar_y_cargar_viajes([csv])

    assert len(df) == 1
    assert "edad" in df.columns


def test_limpieza_resultado_vacio_si_todo_es_anomalo(tmp_path):
    df = limpiar_y_cargar_viajes([_escribir_csv(tmp_path / "a.csv", [_viaje(1, 500)])])
    assert df.empty


# ---------------------------------------------------------------------------
# generar_hash_sha256
# ---------------------------------------------------------------------------
def test_hash_coincide_con_sha256_conocido(tmp_path):
    archivo = tmp_path / "x.bin"
    archivo.write_bytes(b"abc")
    assert generar_hash_sha256(archivo) == hashlib.sha256(b"abc").hexdigest()


def test_hash_de_archivo_mayor_a_un_bloque(tmp_path):
    """Caso límite: el archivo se lee por bloques de 8192 bytes."""
    contenido = bytes(range(256)) * 200  # 51 200 bytes
    archivo = tmp_path / "grande.bin"
    archivo.write_bytes(contenido)
    assert generar_hash_sha256(archivo) == hashlib.sha256(contenido).hexdigest()


def test_hash_cambia_si_cambia_el_archivo(tmp_path):
    archivo = tmp_path / "x.bin"
    archivo.write_bytes(b"datos v1")
    h1 = generar_hash_sha256(archivo)
    archivo.write_bytes(b"datos v2")
    assert generar_hash_sha256(archivo) != h1


def test_hash_de_archivo_vacio(tmp_path):
    archivo = tmp_path / "vacio.bin"
    archivo.write_bytes(b"")
    assert generar_hash_sha256(archivo) == hashlib.sha256(b"").hexdigest()
