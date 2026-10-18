"""Módulo de ingesta y procesamiento de datos crudos a Parquet con versionado."""

import hashlib
import json
from pathlib import Path
import pandas as pd
from mibici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

import requests
from pathlib import Path

from mibici.config import RAW_DATA_DIR

URLS_MIBICI_2026 = [
    "https://mibici.net/site/assets/files/120163/datos_abiertos_2026_01.csv",
    "https://mibici.net/site/assets/files/121761/datos_abiertos_2026_02.csv",
    "https://mibici.net/site/assets/files/123236/datos_abiertos_2026_03.csv",
    "https://mibici.net/site/assets/files/124927/datos_abiertos_2026_04.csv",
    "https://mibici.net/site/assets/files/125943/datos_abiertos_2026_05.csv",
]


def descargar_datos_mibici():
    """
    Descarga automáticamente los datasets abiertos de MiBici
    de enero a mayo de 2026 en data/raw.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS_MIBICI_2026:
        nombre_archivo = url.split("/")[-1]
        destino = RAW_DATA_DIR / nombre_archivo

        if destino.exists():
            print(f"Archivo ya existe: {nombre_archivo}")
            continue

        print(f"Descargando {nombre_archivo}...")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()

        response.raise_for_status()

        with open(destino, "wb") as f:
            f.write(response.content)

        print(f"Guardado en: {destino}")
        
def generar_hash_sha256(filepath: Path) -> str:
    """Calcula el hash SHA256 de un archivo para garantizar la trazabilidad."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def limpiar_y_cargar_viajes(csv_files: list[Path]) -> pd.DataFrame:
    """Carga, unifica y normaliza los archivos CSV de viajes sin data leakage."""
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="latin-1")

        # Corregir encabezados
        df.columns = df.columns.str.replace(
            "AÃ±o_de_nacimiento", "Año_de_nacimiento"
        )
        dfs.append(df)

    df_total = pd.concat(dfs, ignore_index=True)

    # Conversión estandarizada de fechas
    df_total["Inicio_del_viaje"] = pd.to_datetime(df_total["Inicio_del_viaje"])
    df_total["Fin_del_viaje"] = pd.to_datetime(df_total["Fin_del_viaje"])

    # Duración exacta en minutos
    df_total["Duracion_minutos"] = (
        (df_total["Fin_del_viaje"] - df_total["Inicio_del_viaje"]).dt.total_seconds()
        / 60.0
    ).round(2)

    # Edad corregida respetando el año del viaje (SIN LEAKAGE)
    df_total["Edad"] = (
        df_total["Inicio_del_viaje"].dt.year - df_total["Año_de_nacimiento"]
    )

    # Normalizar nombres a minúsculas snake_case
    column_map = {
        "Viaje_Id": "viaje_id",
        "Usuario_Id": "usuario_id",
        "Genero": "genero",
        "Edad": "edad",
        "Inicio_del_viaje": "inicio_del_viaje",
        "Fin_del_viaje": "fin_del_viaje",
        "Origen_Id": "origen_id",
        "Destino_Id": "destino_id",
        "Duracion_minutos": "duracion_minutos",
    }
    df_total = df_total.rename(columns=column_map)

    # Limpieza de valores atípicos
    cond_cancelado = (df_total["duracion_minutos"] < 1) & (
        df_total["origen_id"] == df_total["destino_id"]
    )
    cond_edad_alta = df_total["edad"] > 100
    cond_duracion_extrema = (df_total["duracion_minutos"] < 1) | (
        df_total["duracion_minutos"] > 240
    )

    anomalias = cond_cancelado | cond_edad_alta | cond_duracion_extrema
    return df_total[~anomalias].reset_index(drop=True)


def ejecutar_ingesta():
    """Ejecuta el pipeline completo de ingesta y genera el manifiesto de versión."""
    
    descargar_datos_mibici()
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    archivos_csv = sorted(list(RAW_DATA_DIR.glob("datos_abiertos_*.csv")))
    if not archivos_csv:
        print(f"No se encontraron archivos CSV en {RAW_DATA_DIR}")
        return

    print("Procesando archivos de viajes...")
    df_clean = limpiar_y_cargar_viajes(archivos_csv)

    archivo_salida = PROCESSED_DATA_DIR / "viajes_clean.parquet"
    df_clean.to_parquet(archivo_salida, index=False)

    # Crear manifiesto de versionado
    manifiesto = {
        "registros_totales": len(df_clean),
        "columnas": list(df_clean.columns),
        "hash_sha256": generar_hash_sha256(archivo_salida),
    }

    with open(PROCESSED_DATA_DIR / "data_manifest.json", "w") as f:
        json.dump(manifiesto, f, indent=4)

    print(f"¡Ingesta completada! Dataset guardado en {archivo_salida}")


if __name__ == "__main__":
    ejecutar_ingesta()
