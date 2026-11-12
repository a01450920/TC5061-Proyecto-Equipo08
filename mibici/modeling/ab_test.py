import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici.config import PROCESSED_DATA_DIR, REPORTS_DIR
from mibici.features import preparar_features_p1_demanda

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
OUTPUT_REPORT = REPORTS_DIR / "ab_testing_results.json"
MODEL_NAME = "mibici-equipo08-p1-demanda"


def ejecutar_ab_testing():
    print("=== PASO 5: A/B Testing de Modelos en Periodo Actual (Mayo 2026) ===")

    # 1. Cargar dataset procesado y generar features de P1
    df = pd.read_parquet(PARQUET_PATH)
    p1 = preparar_features_p1_demanda(df)

    # 2. Definir target y variables explicativas
    target_col = "viajes_salientes"
    ignore_cols = ["fecha", "estacion_id", target_col]
    feature_cols = [col for col in p1.columns if col not in ignore_cols]

#3.Filtrar el periodo de evaluación actual (Junio 2026)
    fecha_col = pd.to_datetime(p1["fecha"])
    eval_mask = (fecha_col >= "2026-06-01") & (fecha_col <= "2026-06-30")

    eval_data = p1[eval_mask]
    X_eval = eval_data[feature_cols].astype("float64")
    y_eval = eval_data[target_col]

    print(f"Registros de evaluación (Mayo 2026): {len(X_eval):,} filas")

    # 4. Cargar Champion y Challenger desde MLflow Model Registry
    model_champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    model_challenger = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger")

    # 5. Inferencias
    pred_champ = model_champion.predict(X_eval)
    pred_chall = model_challenger.predict(X_eval)

    # 6. Calcular Métricas A/B
    metrics_champ = {
        "mae": float(mean_absolute_error(y_eval, pred_champ)),
        "rmse": float(mean_squared_error(y_eval, pred_champ) ** 0.5),
        "r2": float(r2_score(y_eval, pred_champ)),
    }

    metrics_chall = {
        "mae": float(mean_absolute_error(y_eval, pred_chall)),
        "rmse": float(mean_squared_error(y_eval, pred_chall) ** 0.5),
        "r2": float(r2_score(y_eval, pred_chall)),
    }

    # 7. Determinar Veredicto
    mejora_mae = metrics_champ["mae"] - metrics_chall["mae"]
    es_mejor = mejora_mae > 0
    ganador = "challenger" if es_mejor else "champion"

    print("\n--- RESULTADOS A/B TESTING ---")
    print(f"Champion (v1)   -> MAE: {metrics_champ['mae']:.4f} | RMSE: {metrics_champ['rmse']:.4f} | R2: {metrics_champ['r2']:.4f}")
    print(f"Challenger (v2) -> MAE: {metrics_chall['mae']:.4f} | RMSE: {metrics_chall['rmse']:.4f} | R2: {metrics_chall['r2']:.4f}")
    print(f"\nVeredicto: Promover {ganador.upper()} (Diferencia MAE: {mejora_mae:+.4f})")

    # 8. Guardar JSON de Resultados
    resultados = {
        "evaluation_period": "2026-05-01 a 2026-05-31",
        "sample_size": len(X_eval),
        "champion": metrics_champ,
        "challenger": metrics_chall,
        "verdict": {
            "winner": ganador,
            "promote_challenger": es_mejor,
            "mae_delta": float(mejora_mae),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4)

    print(f"Reporte A/B Testing guardado exitosamente en: {OUTPUT_REPORT}")


if __name__ == "__main__":
    ejecutar_ab_testing()