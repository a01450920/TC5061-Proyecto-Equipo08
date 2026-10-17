"""Módulo de Feature Engineering corregido sin data leakage para P1, P2 y P3."""

import numpy as np
import pandas as pd


def preparar_features_p1_demanda(df: pd.DataFrame) -> pd.DataFrame:
    """Genera agregación de demanda saliente (viajes) por estación y hora."""
    df_temp = df.copy()
    df_temp["fecha"] = df_temp["inicio_del_viaje"].dt.date
    df_temp["hora"] = df_temp["inicio_del_viaje"].dt.hour

    p1 = (
        df_temp.groupby(["origen_id", "fecha", "hora"])
        .size()
        .reset_index(name="viajes_salientes")
    )
    p1["dia_semana"] = pd.to_datetime(p1["fecha"]).dt.dayofweek
    p1["es_fin_semana"] = p1["dia_semana"].isin([5, 6]).astype(int)

    return p1


def preparar_features_p2_riesgo(
    df: pd.DataFrame, umbral_critico: int = 5
) -> pd.DataFrame:
    """Genera dataset de clasificación binaria P2 de riesgo para t+1 sin data leakage."""
    df_temp = df.copy()
    df_temp["fecha_inicio"] = df_temp["inicio_del_viaje"].dt.date
    df_temp["hora_inicio"] = df_temp["inicio_del_viaje"].dt.hour
    df_temp["fecha_fin"] = df_temp["fin_del_viaje"].dt.date
    df_temp["hora_fin"] = df_temp["fin_del_viaje"].dt.hour

    salidas = (
        df_temp.groupby(["origen_id", "fecha_inicio", "hora_inicio"])
        .size()
        .reset_index(name="salidas")
    )
    salidas.columns = ["estacion_id", "fecha", "hora", "salidas"]

    entradas = (
        df_temp.groupby(["destino_id", "fecha_fin", "hora_fin"])
        .size()
        .reset_index(name="entradas")
    )
    entradas.columns = ["estacion_id", "fecha", "hora", "entradas"]

    flujo = pd.merge(
        entradas, salidas, on=["estacion_id", "fecha", "hora"], how="outer"
    ).fillna(0)
    flujo = flujo.sort_values(
        by=["estacion_id", "fecha", "hora"]
    ).reset_index(drop=True)

    flujo["balance_neto"] = flujo["entradas"] - flujo["salidas"]
    flujo["balance_acumulado"] = flujo.groupby(["estacion_id", "fecha"])[
        "balance_neto"
    ].cumsum()

    # Estado crítico en tiempo t
    flujo["estado_critico"] = (
        np.abs(flujo["balance_acumulado"]) >= umbral_critico
    ).astype(int)

    # Target para t+1 (lo que predeciremos)
    flujo["target_riesgo_t1"] = flujo.groupby(["estacion_id", "fecha"])[
        "estado_critico"
    ].shift(-1)

    return flujo.dropna(subset=["target_riesgo_t1"]).reset_index(drop=True)


def preparar_features_p3_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Genera matriz de perfil porcentual horario por estación para agrupamiento."""
    df_temp = df.copy()
    df_temp["hora"] = df_temp["inicio_del_viaje"].dt.hour

    perfil = pd.crosstab(df_temp["origen_id"], df_temp["hora"])
    perfil_pct = perfil.div(perfil.sum(axis=1), axis=0) * 100.0
    perfil_pct.columns = [f"pct_hora_{h}" for h in perfil_pct.columns]

    return perfil_pct.reset_index()
