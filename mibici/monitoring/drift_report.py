"""
Monitoreo de drift para P1 (demanda) utilizando Evidently.

Compara:
    - Reference dataset: enero-abril 2026
    - Current dataset: mayo 2026

Genera:
    reports/drift/evidently_report.html
    reports/drift/drift_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


from mibici.config import (
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    DRIFT_REFERENCE_START_DATE,
    DRIFT_REFERENCE_END_DATE,
    DRIFT_CURRENT_START_DATE,
    DRIFT_CURRENT_END_DATE,
)
from mibici.features import preparar_features_p1_demanda


PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"

OUTPUT_DIR = REPORTS_DIR / "drift"
HTML_REPORT = OUTPUT_DIR / "evidently_report.html"
SUMMARY_FILE = OUTPUT_DIR / "drift_summary.json"


COLUMNAS_MONITOREADAS = [
    "origen_id",
    "hora",
    "dia_semana",
    "es_fin_semana",
    "viajes_salientes",
]


def cargar_dataset_p1() -> pd.DataFrame:
    """
    Carga el parquet procesado y construye la tabla de demanda P1
    """
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"No existe {PARQUET_PATH}. Ejecuta primero:\nmlflow run . -e ingest")

    df = pd.read_parquet(PARQUET_PATH)
    p1 = preparar_features_p1_demanda(df)

    return p1


def construir_particiones(
    p1: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Usa las fechas definidas para el experimento del proyecto
    """

    reference_start = pd.to_datetime(DRIFT_REFERENCE_START_DATE).date()
    reference_end = pd.to_datetime(DRIFT_REFERENCE_END_DATE).date()

    current_start = pd.to_datetime(DRIFT_CURRENT_START_DATE).date()
    current_end = pd.to_datetime(DRIFT_CURRENT_END_DATE).date()

    reference = p1[(p1["fecha"] >= reference_start) & (p1["fecha"] <= reference_end)].copy()
    current = p1[(p1["fecha"] >= current_start) & (p1["fecha"] <= current_end)].copy()

    reference = reference[COLUMNAS_MONITOREADAS]
    current = current[COLUMNAS_MONITOREADAS]
        
    reference = reference.astype({"viajes_salientes": "float64",})
    current = current.astype({"viajes_salientes": "float64",})
    
    if reference.empty:
        raise ValueError("El dataset de referencia quedó vacío.")

    if current.empty:
        raise ValueError("El dataset actual quedó vacío.")

    return reference, current


def generar_reporte(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Ejecuta el análisis de drift con los tipos de variables
    definidos explícitamente.
    """
    schema = DataDefinition(
        numerical_columns=[
            "hora",
            "viajes_salientes",
        ],
        categorical_columns=[
            "origen_id",
            "dia_semana",
            "es_fin_semana",
        ],
    )

    reference_dataset = Dataset.from_pandas(
        reference,
        data_definition=schema,
    )

    current_dataset = Dataset.from_pandas(
        current,
        data_definition=schema,
    )
    report = Report(
        metrics=[DataDriftPreset(),]
    )

    return report.run(
        current_data=current_dataset,
        reference_data=reference_dataset,
    )

def guardar_resumen(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    """
    Guarda información básica para la documentación.
    """
    resumen = {
        "reference_period": {
            "start": DRIFT_REFERENCE_START_DATE,
            "end": DRIFT_REFERENCE_END_DATE,
            "rows": int(len(reference)),
        },
        "current_period": {
            "start": DRIFT_CURRENT_START_DATE,
            "end": DRIFT_CURRENT_END_DATE,
            "rows": int(len(current)),
        },
        "columns": COLUMNAS_MONITOREADAS,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8",) as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False,)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True,)
    
    print("Cargando datos...")

    p1 = cargar_dataset_p1()
    reference, current = construir_particiones(p1)

    print(f"Reference dataset: {len(reference):,} filas")
    print(f"Current dataset:   {len(current):,} filas")
    print("Generando reporte de Evidently...")

    resultado = generar_reporte(reference, current,)
    resultado.save_html(str(HTML_REPORT))
    
    guardar_resumen(reference, current,)
    print("\nReporte generado correctamente:")
    print(f"HTML:   {HTML_REPORT}")
    print(f"JSON:   {SUMMARY_FILE}")


if __name__ == "__main__":
    main()