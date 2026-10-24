"""Registra la mejor candidata de cada pregunta en el Model Registry y la lleva a producción.

Esquema de ciclo de vida: ALIASES (champion = versión en producción,
challenger = siguiente mejor, opcional). Las etapas clásicas (Staging/Production)
están en desuso desde MLflow 2.9; los aliases son el mecanismo vigente.

La mejor candidata se elige con la métrica de VALIDACIÓN (no la de test), de modo
que el test siga siendo una estimación honesta del desempeño en producción.

Uso (con el servidor de tracking activo):
    mlflow run . -e register --experiment-name mibici-equipo08
    python -m mibici.modeling.register --challenger      # también registra el segundo mejor
"""

from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc
import pandas as pd

from mibici.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REPORTS_DIR

# métrica de selección, si menor es mejor, métrica de prueba para comparar con el baseline
MODELOS = {
    "p1": {
        "nombre": "mibici-equipo08-p1-demanda",
        "seleccion": "val_mae", "menor_es_mejor": True, "prueba": "test_mae",
        "descripcion": (
            "P1 - Regresión de demanda: predice los viajes salientes por estación, fecha y hora "
            "(MiBici Guadalajara, ene-may 2026). Métrica de negocio: MAE de viajes por hora."
        ),
    },
    "p2": {
        "nombre": "mibici-equipo08-p2-riesgo",
        "seleccion": "val_recall", "menor_es_mejor": False, "prueba": "test_recall",
        "descripcion": (
            "P2 - Clasificación de riesgo de desbalance: predice si una estación estará en estado "
            "crítico (|balance acumulado| >= 5) en la hora siguiente. Métrica de negocio: recall."
        ),
    },
    "p3": {
        "nombre": "mibici-equipo08-p3-grupos",
        "seleccion": "silhouette_score", "menor_es_mejor": False, "prueba": "silhouette_score",
        "descripcion": (
            "P3 - Agrupamiento de estaciones por perfil horario de uso con KMeans. "
            "Métrica: silhouette score."
        ),
    },
}


def buscar(experiment_id: str, pregunta: str, stage: str, orden: str) -> pd.DataFrame:
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.question = '{pregunta}' and tags.stage_of_work = '{stage}'",
        order_by=[orden],
    )


def registrar_version(client: MlflowClient, nombre: str, fila: pd.Series, pregunta: str, rol: str):
    """Registra el modelo de una corrida (con linaje en tags) y devuelve la versión."""
    run_id = fila["run_id"]
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", nombre)
    except Exception as e:  # MLflow 3 guarda modelos como LoggedModel: usar la URI de respaldo
        respaldo = fila.get("tags.logged_model_uri")
        if not isinstance(respaldo, str):
            raise RuntimeError(f"No se pudo registrar {nombre} desde la corrida {run_id}: {e}") from e
        mv = mlflow.register_model(respaldo, nombre)

    linaje = {
        "experimento": MLFLOW_EXPERIMENT_NAME,
        "run_id_origen": run_id,
        "git_commit": str(fila.get("tags.git_commit")),
        "data_hash_version": str(fila.get("params.data_hash_version")),
        "model_family": str(fila.get("tags.model_family")),
        "rol": rol,
    }
    for k, v in linaje.items():
        client.set_model_version_tag(nombre, mv.version, k, v)
    client.update_model_version(
        nombre, mv.version,
        description=(f"{rol} de {pregunta.upper()}: {linaje['model_family']} | "
                     f"corrida {run_id[:8]} | commit {linaje['git_commit'][:7]} | "
                     f"datos {linaje['data_hash_version']}"),
    )
    return mv


def alias_actual(client: MlflowClient, nombre: str, alias: str):
    try:
        return client.get_model_version_by_alias(nombre, alias)
    except Exception:
        return None


def mejora(menor_es_mejor: bool, base: float, cand: float) -> str:
    dif = (base - cand) if menor_es_mejor else (cand - base)
    rel = f" ({dif / abs(base) * 100:+.1f}%)" if base else ""
    return f"{dif:+.4f}{rel}"


def exportar_tabla(experiment_id: str) -> None:
    """Tabla baseline vs candidatas para el reporte (reports/tabla_comparativa.csv)."""
    columnas = {
        "tags.question": "pregunta", "tags.stage_of_work": "etapa", "tags.model_family": "modelo",
        "params.n_estimators": "n_estimators", "params.max_depth": "max_depth",
        "params.n_clusters": "n_clusters", "metrics.val_mae": "val_mae", "metrics.test_mae": "test_mae",
        "metrics.test_rmse": "test_rmse", "metrics.val_recall": "val_recall",
        "metrics.test_recall": "test_recall", "metrics.test_precision": "test_precision",
        "metrics.test_f1": "test_f1", "metrics.test_pr_auc": "test_pr_auc",
        "metrics.silhouette_score": "silhouette", "metrics.davies_bouldin": "davies_bouldin",
        "run_id": "run_id",
    }
    df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.stage_of_work = 'baseline' or tags.stage_of_work = 'candidate'",
    )
    if df.empty:
        return
    tabla = df.reindex(columns=list(columnas)).rename(columns=columnas).sort_values(["pregunta", "etapa"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = REPORTS_DIR / "tabla_comparativa.csv"
    tabla.to_csv(destino, index=False)
    print(f"\nTabla comparativa guardada en {destino}")


def main(challenger: bool) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"No existe el experimento '{MLFLOW_EXPERIMENT_NAME}'. Corre primero el entrenamiento.")

    for pregunta, cfg in MODELOS.items():
        nombre = cfg["nombre"]
        sentido = "ASC" if cfg["menor_es_mejor"] else "DESC"
        orden = "metrics." + cfg["seleccion"] + " " + sentido
        cands = buscar(exp.experiment_id, pregunta, "candidate", orden)
        cands = cands.dropna(subset=["metrics." + cfg["seleccion"]]) if not cands.empty else cands
        if cands.empty:
            print(f"[{pregunta}] sin candidatas: corre mlflow run . -e train -P question={pregunta}")
            continue

        base = buscar(exp.experiment_id, pregunta, "baseline", "attributes.start_time DESC")
        mejor = cands.iloc[0]
        print(f"\n=== {pregunta.upper()} -> {nombre} ===")
        col_sel = "metrics." + cfg["seleccion"]
        print(f"Elegida: {mejor['tags.model_family']} (run {mejor['run_id'][:8]}) "
              f"{cfg['seleccion']}={mejor[col_sel]:.4f}")
        col = "metrics." + cfg["prueba"]
        if not base.empty and col in base and pd.notna(base.iloc[0][col]) and pd.notna(mejor.get(col)):
            print(f"{cfg['prueba']}: baseline {base.iloc[0][col]:.4f} -> candidata {mejor[col]:.4f} "
                  f"| mejora {mejora(cfg['menor_es_mejor'], base.iloc[0][col], mejor[col])}")

        champion = alias_actual(client, nombre, "champion")
        if champion is not None and champion.run_id == mejor["run_id"]:
            print(f"Ya es champion (versión {champion.version}); no se vuelve a registrar.")
        else:
            mv = registrar_version(client, nombre, mejor, pregunta, "champion")
            client.update_registered_model(nombre, description=cfg["descripcion"])
            client.set_registered_model_alias(nombre, "champion", mv.version)
            print(f"Registrada versión {mv.version} con alias 'champion'.")

        if challenger and len(cands) > 1:
            seg = cands.iloc[1]
            actual = alias_actual(client, nombre, "challenger")
            if actual is None or actual.run_id != seg["run_id"]:
                mv2 = registrar_version(client, nombre, seg, pregunta, "challenger")
                client.set_registered_model_alias(nombre, "challenger", mv2.version)
                print(f"Versión {mv2.version} con alias 'challenger'.")

        uri = f"models:/{nombre}@champion"
        modelo = mlflow.pyfunc.load_model(uri)
        print(f"Verificado: se carga {uri} (firma: {modelo.metadata.signature is not None})")

    exportar_tabla(exp.experiment_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registro de modelos MiBici en el Model Registry")
    parser.add_argument("--challenger", action="store_true", help="registra también el segundo mejor")
    main(parser.parse_args().challenger)
