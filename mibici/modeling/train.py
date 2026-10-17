"""Script principal para entrenar modelos base de P1, P2 y P3 y registrar resultados en MLflow."""

import os
import git
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, recall_score, silhouette_score

from mibici.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from mibici.features import (
    preparar_features_p1_demanda,
    preparar_features_p2_riesgo,
    preparar_features_p3_clustering,
)


def obtener_commit_hash():
    """Obtiene el hash del commit actual de Git para trazabilidad en MLflow."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "sin_commit_git"


def ejecutar_experimento_base():
    # Asegurar que exista la carpeta de figuras
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar dataset limpio en Parquet
    parquet_path = PROCESSED_DATA_DIR / "viajes_clean.parquet"
    if not parquet_path.exists():
        print(f"Error: No existe el archivo {parquet_path}. Ejecuta primero dataset.py")
        return

    df_viajes = pd.read_parquet(parquet_path)
    commit_hash = obtener_commit_hash()

    # Configurar MLflow
    mlflow.set_experiment("mibici-equipo08")

    # =========================================================================
    # PREGUNTA 1 (P1): REGRESIÓN DE DEMANDA FUTURA
    # =========================================================================
    print("\n--- Ejecutando Run P1: Regresión de Demanda ---")
    p1_df = preparar_features_p1_demanda(df_viajes)

    # Partición Temporal (SIN LEAKAGE)
    p1_train = p1_df[
        (p1_df["fecha"] >= pd.to_datetime(TRAIN_START_DATE).date())
        & (p1_df["fecha"] <= pd.to_datetime(TRAIN_END_DATE).date())
    ]
    p1_test = p1_df[
        (p1_df["fecha"] >= pd.to_datetime(TEST_START_DATE).date())
        & (p1_df["fecha"] <= pd.to_datetime(TEST_END_DATE).date())
    ]

    features_p1 = ["origen_id", "hora", "dia_semana", "es_fin_semana"]
    X_train_p1, y_train_p1 = p1_train[features_p1], p1_train["viajes_salientes"]
    X_test_p1, y_test_p1 = p1_test[features_p1], p1_test["viajes_salientes"]

    model_p1 = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEED)
    model_p1.fit(X_train_p1, y_train_p1)
    preds_p1 = model_p1.predict(X_test_p1)

    test_mae = mean_absolute_error(y_test_p1, preds_p1)

    # Generar artefacto de gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_p1.values[:100], label="Real", alpha=0.7)
    ax.plot(preds_p1[:100], label="Predicción", alpha=0.7)
    ax.set_title("P1: Comparativa Demanda Real vs Predicción (Muestra)")
    ax.legend()
    fig_path_p1 = FIGURES_DIR / "p1_demanda_pred.png"
    plt.savefig(fig_path_p1)
    plt.close()

    with mlflow.start_run(run_name="P1_Baseline_Regresion"):
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("pregunta", "P1")
        mlflow.set_tag("git_commit", commit_hash)

        mlflow.log_param("train_start", TRAIN_START_DATE)
        mlflow.log_param("train_end", TRAIN_END_DATE)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("seed", RANDOM_SEED)

        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_artifact(fig_path_p1)
        print(f"P1 completado | test_mae: {test_mae:.4f}")

    # =========================================================================
    # PREGUNTA 2 (P2): CLASIFICACIÓN DE RIESGO
    # =========================================================================
    print("\n--- Ejecutando Run P2: Clasificación de Riesgo ---")
    p2_df = preparar_features_p2_riesgo(df_viajes)

    p2_train = p2_df[
        (p2_df["fecha"] >= pd.to_datetime(TRAIN_START_DATE).date())
        & (p2_df["fecha"] <= pd.to_datetime(TRAIN_END_DATE).date())
    ]
    p2_test = p2_df[
        (p2_df["fecha"] >= pd.to_datetime(TEST_START_DATE).date())
        & (p2_df["fecha"] <= pd.to_datetime(TEST_END_DATE).date())
    ]

    features_p2 = ["estacion_id", "hora", "entradas", "salidas", "balance_neto", "balance_acumulado"]
    X_train_p2, y_train_p2 = p2_train[features_p2], p2_train["target_riesgo_t1"]
    X_test_p2, y_test_p2 = p2_test[features_p2], p2_test["target_riesgo_t1"]

    model_p2 = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    model_p2.fit(X_train_p2, y_train_p2)
    preds_p2 = model_p2.predict(X_test_p2)

    test_recall = recall_score(y_test_p2, preds_p2, zero_division=0)

    # Artefacto
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(preds_p2).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"])
    ax.set_title("P2: Distribución de Predicciones de Riesgo (0 vs 1)")
    fig_path_p2 = FIGURES_DIR / "p2_riesgo_dist.png"
    plt.savefig(fig_path_p2)
    plt.close()

    with mlflow.start_run(run_name="P2_Baseline_Clasificacion"):
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("pregunta", "P2")
        mlflow.set_tag("git_commit", commit_hash)

        mlflow.log_param("train_start", TRAIN_START_DATE)
        mlflow.log_param("train_end", TRAIN_END_DATE)
        mlflow.log_param("model_type", "RandomForestClassifier")

        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_artifact(fig_path_p2)
        print(f"P2 completado | test_recall: {test_recall:.4f}")

    # =========================================================================
    # PREGUNTA 3 (P3): CLUSTERING DE PERFILES HORARIOS
    # =========================================================================
    print("\n--- Ejecutando Run P3: Clustering de Estaciones ---")
    p3_df = preparar_features_p3_clustering(df_viajes)
    X_p3 = p3_df.drop(columns=["origen_id"])

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X_p3)

    sil_score = silhouette_score(X_p3, clusters)

    # Artefacto
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(3):
        ax.plot(X_p3[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
    ax.set_title("P3: Perfil Horario Promedio por Cluster")
    ax.legend()
    fig_path_p3 = FIGURES_DIR / "p3_cluster_profiles.png"
    plt.savefig(fig_path_p3)
    plt.close()

    with mlflow.start_run(run_name="P3_Baseline_Clustering"):
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("pregunta", "P3")
        mlflow.set_tag("git_commit", commit_hash)

        mlflow.log_param("n_clusters", 3)
        mlflow.log_param("model_type", "KMeans")

        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.log_artifact(fig_path_p3)
        print(f"P3 completado | silhouette_score: {sil_score:.4f}")


if __name__ == "__main__":
    ejecutar_experimento_base()
