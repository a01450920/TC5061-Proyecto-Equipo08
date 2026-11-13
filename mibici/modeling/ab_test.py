"""Paso 5: A/B testing champion vs challenger sobre el periodo actual (junio 2026)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mibici.config as cfg
from mibici.config import PROCESSED_DATA_DIR, REPORTS_DIR
from mibici.features import preparar_features_p1_demanda

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
OUT_DIR = REPORTS_DIR / "ab_testing"
MODEL_NAME = "mibici-equipo08-p1-demanda"
EVAL_START, EVAL_END = "2026-06-01", "2026-06-30"


def metricas(y, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def bootstrap_dif(err_champ, err_chall, n=1000, seed=42):
    """IC 95% de la diferencia media de error (champion - challenger).
    > 0 favorece al challenger."""
    rng = np.random.default_rng(seed)
    d = err_champ - err_chall
    medias = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(d.mean()), [float(x) for x in np.percentile(medias, [2.5, 97.5])]


def bootstrap_dif_bloques(err_champ, err_chall, dias, n=2000, seed=42):
    """IC 95% de la diferencia media de error (champion - challenger),
    remuestreando días completos (bootstrap por bloques). > 0 favorece al challenger."""
    d = pd.DataFrame({"dia": dias, "d": err_champ - err_chall})
    g = d.groupby("dia")["d"].agg(["sum", "count"])
    sumas, cuentas = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    medias = sumas[idx].sum(axis=1) / cuentas[idx].sum(axis=1)
    return float(sumas.sum() / cuentas.sum()), [float(x) for x in np.percentile(medias, [2.5, 97.5])]


def veredicto(lo, hi):
    if lo > 0:
        return "challenger"
    if hi < 0:
        return "champion"
    return "sin diferencia significativa (se mantiene champion)"


def ejecutar_ab_testing():
    print(f"=== PASO 5: A/B Testing en periodo actual ({EVAL_START} a {EVAL_END}) ===")
    mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    p1 = preparar_features_p1_demanda(pd.read_parquet(PARQUET_PATH))
    target_col = "viajes_salientes"
    feature_cols = [c for c in p1.columns if c not in ["fecha", "estacion_id", target_col]]

    fecha = pd.to_datetime(p1["fecha"])
    eval_data = p1[(fecha >= EVAL_START) & (fecha <= EVAL_END)]
    if eval_data.empty:
        raise RuntimeError("Sin datos de junio. Corre: python -m mibici.dataset")
    X_eval = eval_data[feature_cols].astype("float64")
    y_eval = eval_data[target_col].to_numpy()
    dias = eval_data["fecha"].to_numpy()
    print(f"Registros de evaluación: {len(X_eval):,}")

    ver_champ = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    ver_chall = client.get_model_version_by_alias(MODEL_NAME, "challenger").version
    model_champ = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    model_chall = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@challenger")

    pred_champ = model_champ.predict(X_eval)
    pred_chall = model_chall.predict(X_eval)
    m_champ, m_chall = metricas(y_eval, pred_champ), metricas(y_eval, pred_chall)

    # IC de la diferencia en error absoluto (MAE) y en error cuadrático (base del RMSE)
    dif_mae, (lo_mae, hi_mae) = bootstrap_dif_bloques(
        np.abs(y_eval - pred_champ), np.abs(y_eval - pred_chall), dias)
    dif_mse, (lo_mse, hi_mse) = bootstrap_dif_bloques(
        (y_eval - pred_champ) ** 2, (y_eval - pred_chall) ** 2, dias)
    ganador = veredicto(lo_mae, hi_mae)

    print("\n--- RESULTADOS A/B TESTING ---")
    print(f"Champion   (v{ver_champ}) -> {m_champ}")
    print(f"Challenger (v{ver_chall}) -> {m_chall}")
    print(f"Dif. MAE (champion - challenger): {dif_mae:+.4f}  IC95%: [{lo_mae:+.4f}, {hi_mae:+.4f}]")
    print(f"Dif. error cuadrático medio:       {dif_mse:+.4f}  IC95%: [{lo_mse:+.4f}, {hi_mse:+.4f}]")
    print(f"Veredicto (según MAE): {ganador}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pd.DataFrame({f"champion_v{ver_champ}": m_champ,
                          f"challenger_v{ver_chall}": m_chall}).T
    tabla.to_csv(OUT_DIR / "comparacion_metricas.csv")

    ax = tabla[["mae", "rmse"]].plot.bar(rot=0, figsize=(6, 4))
    ax.set_title("A/B testing, junio 2026 (menor es mejor)")
    plt.tight_layout(); plt.savefig(OUT_DIR / "ab_metricas.png", dpi=150); plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (nombre, pred) in zip(axs, [(f"champion v{ver_champ}", pred_champ),
                                         (f"challenger v{ver_chall}", pred_chall)]):
        ax.scatter(y_eval, pred, s=4, alpha=0.3)
        lim = [0, max(y_eval.max(), pred.max())]
        ax.plot(lim, lim, "r--")
        ax.set_title(nombre); ax.set_xlabel("Real")
    axs[0].set_ylabel("Predicho")
    plt.tight_layout(); plt.savefig(OUT_DIR / "ab_real_vs_predicho.png", dpi=150); plt.close()

    resultados = {
        "evaluation_period": f"{EVAL_START} a {EVAL_END}",
        "sample_size": int(len(X_eval)),
        "champion": {"version": int(ver_champ), **m_champ},
        "challenger": {"version": int(ver_chall), **m_chall},
        "mae_delta_champion_minus_challenger": dif_mae,
        "mae_ic95": [lo_mae, hi_mae],
        "mse_delta_champion_minus_challenger": dif_mse,
        "mse_ic95": [lo_mse, hi_mse],
        "winner": ganador,
    }
    (OUT_DIR / "ab_testing_results.json").write_text(
        json.dumps(resultados, indent=4), encoding="utf-8")

    with mlflow.start_run(run_name="ab_test_p1_junio"):
        mlflow.set_tags({"equipo": "equipo08", "question": "p1", "stage_of_work": "ab_test"})
        mlflow.log_params({"champion_version": ver_champ, "challenger_version": ver_chall,
                           "eval_period": resultados["evaluation_period"],
                           "n_eval": len(X_eval)})
        mlflow.log_metrics({**{f"champion_{k}": v for k, v in m_champ.items()},
                            **{f"challenger_{k}": v for k, v in m_chall.items()},
                            "mae_delta": dif_mae})
        mlflow.log_artifacts(str(OUT_DIR))
    print(f"Resultados guardados en {OUT_DIR}")


if __name__ == "__main__":
    ejecutar_ab_testing()
