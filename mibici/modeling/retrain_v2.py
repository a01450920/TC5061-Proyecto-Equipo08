"""
Paso 4: Entrena una versión alternativa (v2) del modelo P1 incorporando 
el periodo acumulado (hasta Mayo 2026) y la registra en MLflow Model Registry como @challenger.
"""

from __future__ import annotations
import os
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mibici.config as cfg
from mibici.config import PROCESSED_DATA_DIR, RANDOM_SEED
from mibici.features import preparar_features_p1_demanda

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
MODEL_NAME = "mibici-equipo08-p1-demanda"


def ejecutar_reentrenamiento_v2():
    client = MlflowClient()

    print("=== PASO 4: Entrenando Versión Alternativa (v2) de P1 (Enero - Mayo 2026) ===")
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"No existe {PARQUET_PATH}. Ejecuta la ingesta primero.")

    # 1. Cargar datos completos y procesar features
    df = pd.read_parquet(PARQUET_PATH)
    p1 = preparar_features_p1_demanda(df)

    # Definir variables explicativas y objetivo
    target_col = "viajes_salientes"
    ignore_cols = ["fecha", "estacion_id", target_col]
    feature_cols = [col for col in p1.columns if col not in ignore_cols]

    # 2. Partición temporal (Entrenamiento: Ene-May 2026 | Evaluación: Junio 2026)
    fecha_col = pd.to_datetime(p1["fecha"])

    train_mask = fecha_col <= "2026-05-31"
    eval_mask = (fecha_col >= "2026-06-01") & (fecha_col <= "2026-06-30")

    train_data = p1[train_mask]
    eval_data = p1[eval_mask]

    X_train = train_data[feature_cols].astype("float64")
    y_train = train_data[target_col]

    X_eval = eval_data[feature_cols].astype("float64")
    y_eval = eval_data[target_col]

    print(f"Dataset de Reentrenamiento (Ene-May 2026): {len(X_train):,} filas")
    print(f"Dataset de Evaluación Actual (Junio 2026): {len(X_eval):,} filas")

    # 3. Reutilizar la corrida activa del CLI o crear una nueva si se corre directo
    active_run = mlflow.active_run()
    if active_run:
        run_context = active_run
    else:
        mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
        run_context = mlflow.start_run(run_name="p1_reentrenado_v2_junio")

    with run_context:
        # Entrenar RandomForest
        modelo_v2 = RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=2, min_samples_leaf=2,
            random_state=RANDOM_SEED, n_jobs=-1
        )
        modelo_v2.fit(X_train, y_train)

        # Evaluar en Junio 2026
        pred_junio = modelo_v2.predict(X_eval)
        mae = float(mean_absolute_error(y_eval, pred_junio))
        rmse = float(mean_squared_error(y_eval, pred_junio) ** 0.5)

        print(f"Métricas en Junio 2026 -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Tags y Parámetros
        mlflow.set_tags({
            "equipo": "equipo08",
            "question": "p1",
            "model_family": "rf_retrained",
            "stage_of_work": "candidate_v2",
            "run_role": "challenger",
        })
        mlflow.log_params({
            "n_estimators": 300,
            "train_period": "2026-01-01_to_2026-05-31",
            "eval_period": "2026-06-01_to_2026-06-30",
            "reason": "Reentrenamiento acumulado incluyendo mayo"
        })
        mlflow.log_metrics({"eval_junio_mae": mae, "eval_junio_rmse": rmse})

        # Registrar Modelo en MLflow
        info = mlflow.sklearn.log_model(
            sk_model=modelo_v2,
            name="model",
            serialization_format="cloudpickle"
        )

        # 4. Asignar versión en Model Registry con alias challenger
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
        print(f"\n¡Éxito! Modelo v2 registrado en MLflow Model Registry.")
        print(f"Modelo: {MODEL_NAME} | Versión: {mv.version} | Alias: 'challenger'")


if __name__ == "__main__":
    ejecutar_reentrenamiento_v2()