"""Módulo de entrenamiento de modelos base (P1, P2 y P3) con tracking en MLflow."""

import json
import git
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import argparse #agregado para semana 6
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, recall_score, silhouette_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler


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


def obtener_commit_hash() -> str:
    """Obtiene el hash SHA-1 del commit actual de Git para garantizar trazabilidad en MLflow.

    Returns:
        str: Cadena con el hash del commit o 'sin_commit_git' si no se detecta repositorio.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "sin_commit_git"


def obtener_version_datos() -> str:
    """Lee el hash SHA256 del manifiesto de datos generado durante la ingesta.

    Returns:
        str: Subcadena del hash SHA256 del archivo de datos procesados.
    """
    manifest_path = PROCESSED_DATA_DIR / "data_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                return manifest.get("hash_sha256", "desconocido")[:12]
        except Exception:
            pass
    return "desconocido"

def registrar_run_mlflow(
    run_name: str,
    pregunta: str,
    commit_hash: str,
    params: dict,
    metrics: dict,
    artifacts: list | None = None,
):
    """
    Registra un experimento en MLflow.

    Parameters
    ----------
    run_name : str
        Nombre del run.
    pregunta : str
        P1, P2 o P3.
    commit_hash : str
        Hash del commit actual.
    params : dict
        Parámetros del modelo.
    metrics : dict
        Métricas calculadas.
    artifacts : list | None
        Lista de rutas de artefactos.
    """
    with mlflow.start_run(run_name=run_name,nested=True):

        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("pregunta", pregunta)
        mlflow.set_tag("git_commit", commit_hash)

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        if artifacts:
            for artifact in artifacts:
                mlflow.log_artifact(str(artifact))


def ejecutar_experimento_base(run_version: str = "v1"):
    """Ejecuta los modelos base de P1, P2 y P3 registrando hiperparámetros, métricas y artefactos en MLflow.

    Args:
        run_version (str): Identificador de versión secuencial para la corrida (ej. 'v1', 'v2').
    """
    # Asegurar que exista la carpeta de figuras
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar dataset limpio en Parquet
    parquet_path = PROCESSED_DATA_DIR / "viajes_clean.parquet"
    if not parquet_path.exists():
        print(f"Error: No existe el archivo {parquet_path}. Ejecuta primero 'python -m mibici.dataset'")
        return

    df_viajes = pd.read_parquet(parquet_path)
    commit_hash = obtener_commit_hash()
    data_hash = obtener_version_datos()

    # Configurar el experimento central en MLflow
    #mlflow.set_experiment("mibici-equipo08")#Corrección 21OCT26

    # =========================================================================
    # PREGUNTA 1 (P1): REGRESIÓN DE DEMANDA FUTURA
    # =========================================================================
    print(f"\n--- Ejecutando Run P1_{run_version} ---")
    p1_df = preparar_features_p1_demanda(df_viajes)

    # Partición temporal estricta (SIN DATA LEAKAGE)
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

    
    #<--------------------------------------------------------------->#
    # Entrenamiento del modelo dummy con DummyRegressor como baseline
    print(f"P1_{run_version}: Baseline - Dummy Regressor ---")
    dummy_p1 = DummyRegressor(strategy="mean")
    dummy_p1.fit(X_train_p1, y_train_p1)
    preds_dummy_p1 = dummy_p1.predict(X_test_p1)
    
    dummy_mae = mean_absolute_error(y_test_p1, preds_dummy_p1,)
    
    # Generar artefacto de gráfica para Dummy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_p1.values[:100], label="Real", alpha=0.7,)
    ax.plot(preds_dummy_p1[:100], label="Dummy Baseline", alpha=0.7,)
    ax.set_title("P1: Demanda Real vs Dummy Baseline")
    ax.legend()
    fig_path_dummy_p1 = (FIGURES_DIR / f"p1_dummy_baseline_{run_version}.png")
    plt.savefig(fig_path_dummy_p1)
    plt.close()

    #Registrar run Dummy en MLflow
    registrar_run_mlflow(
        run_name=f"P1_{run_version}_DummyRegressor",
        pregunta="P1",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "model_type": "DummyRegressor",
            "strategy": "mean",
        },
        metrics={
            "test_mae": dummy_mae,
        },
        artifacts=[fig_path_dummy_p1]
    )
    
    print(f"P1 Dummy completado | test_mae: {dummy_mae:.4f}")
    
    #<----------------------------------------------------------->#
    # Entrenamiento del modelo de regresión Random Forest para P1
    print(f"P1_{run_version}: Regresión de Demanda ---")
    model_p1 = RandomForestRegressor(n_estimators=300, max_depth=None,min_samples_split=2,min_samples_leaf=2, random_state=RANDOM_SEED)
    model_p1.fit(X_train_p1, y_train_p1)
    preds_p1 = model_p1.predict(X_test_p1)

    test_mae = mean_absolute_error(y_test_p1, preds_p1)

    # Generar artefacto de gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_p1.values[:100], label="Real", alpha=0.7)
    ax.plot(preds_p1[:100], label="Predicción", alpha=0.7)
    ax.set_title("P1: Comparativa Demanda Real vs Predicción (Muestra)")
    ax.legend()
    fig_path_p1 = FIGURES_DIR / f"p1_demanda_pred_{run_version}.png"
    plt.savefig(fig_path_p1)
    plt.close()

    #Registrar run en MLflow
    registrar_run_mlflow(
        run_name=f"P1_{run_version}_RandomForest",
        pregunta="P1",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "model_type": "RandomForestRegressor",
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split":2,
            "min_samples_leaf":2,
            "seed": RANDOM_SEED,
        },
        metrics={
            "test_mae": test_mae,
        },
        artifacts=[fig_path_p1],
    )

    print(f"P1 completado | test_mae: {test_mae:.4f}")

    # =========================================================================
    # PREGUNTA 2 (P2): CLASIFICACIÓN DE RIESGO
    # =========================================================================
    print(f"\n--- Ejecutando Run P2_{run_version} ---")
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


    #<--------------------------------------------------------------->#
    # Entrenamiento del modelo dummy con DummyClassifier como baseline
    print(f"P2_{run_version}: Baseline - Dummy Classifier ---")
    dummy_p2 = DummyClassifier(strategy="most_frequent")
    dummy_p2.fit(X_train_p2, y_train_p2)
    preds_dummy_p2 = dummy_p2.predict(X_test_p2)
    dummy_recall = recall_score(y_test_p2, preds_dummy_p2, zero_division=0,)
    
    # Generar artefacto de gráfica para Dummy
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(preds_dummy_p2).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"],)
    ax.set_title("P2: Distribución de Predicciones Dummy")
    fig_path_dummy_p2 = (FIGURES_DIR / f"p2_dummy_baseline_{run_version}.png")
    plt.savefig(fig_path_dummy_p2)
    plt.close()

    #Registrar run Dummy en MLflow
    registrar_run_mlflow(
        run_name=f"P2_{run_version}_DummyClassifier",
        pregunta="P2",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "model_type": "DummyClassifier",
            "strategy": "most_frequent",
        },
        metrics={
            "test_recall": dummy_recall,
        },
        artifacts=[fig_path_dummy_p2]
    )

    print(f"P2 Dummy completado | test_recall: {dummy_recall:.4f}")

    #<----------------------------------------------------------->#
    # Entrenamiento del modelo de regresión Random Forest para P2
    print(f"P2_{run_version}: Clasificación de Riesgo ---")
    model_p2 = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
    model_p2.fit(X_train_p2, y_train_p2)
    preds_p2 = model_p2.predict(X_test_p2)

    test_recall = recall_score(y_test_p2, preds_p2, zero_division=0)

    # Artefacto
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(preds_p2).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"])
    ax.set_title("P2: Distribución de Predicciones de Riesgo (0 vs 1)")
    fig_path_p2 = FIGURES_DIR / f"p2_riesgo_dist_{run_version}.png"
    plt.savefig(fig_path_p2)
    plt.close()

    #Registrar run en MLflow
    registrar_run_mlflow(
        run_name=f"P2_{run_version}_RandomForest",
        pregunta="P2",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "train_start": TRAIN_START_DATE,
            "train_end": TRAIN_END_DATE,
            "model_type": "RandomForestClassifier",
            "n_estimators": 50,
            "seed": RANDOM_SEED,
        },
        metrics={
            "test_recall": test_recall,
        },
        artifacts=[fig_path_p2],
    )

    print(f"P2 completado | test_recall: {test_recall:.4f}")

    # =========================================================================
    # PREGUNTA 3 (P3): CLUSTERING DE PERFILES HORARIOS
    # =========================================================================
    print(f"\n--- Ejecutando Run P3_{run_version}: Clustering de Estaciones ---")
    p3_df = preparar_features_p3_clustering(df_viajes)
    X_p3 = p3_df.drop(columns=["origen_id"])

    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X_p3)

    sil_score = silhouette_score(X_p3, clusters)

    # Artefacto
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(3):
        ax.plot(X_p3[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
    ax.set_title("P3: Perfil Horario Promedio por Cluster")
    ax.legend()
    fig_path_p3 = FIGURES_DIR / f"p3_cluster_profiles_{run_version}.png"
    plt.savefig(fig_path_p3)
    plt.close()

    #Registrar run en MLflow
    registrar_run_mlflow(
        run_name=f"P3_{run_version}_KMeans",
        pregunta="P3",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "n_clusters": 2,
            "model_type": "KMeans",
        },
        metrics={
            "silhouette_score": sil_score,
        },
        artifacts=[fig_path_p3],
    )

    print(f"P3 completado | silhouette_score: {sil_score:.4f}")

#Modificación semana 6
#----------------------

def ejecutar_por_pregunta(
    question: str = "p1",
    model: str = "rf",
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_clusters: int = 2,
    run_version: str = "v1"
):
    """Ejecuta una pregunta específica (P1, P2 o P3) según los parámetros de MLproject."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "viajes_clean.parquet"
    
    if not parquet_path.exists():
        print(f"Error: No existe el archivo {parquet_path}. Ejecuta primero 'python -m mibici.dataset'")
        return

    df_viajes = pd.read_parquet(parquet_path)
    commit_hash = obtener_commit_hash()
    data_hash = obtener_version_datos()
    #mlflow.set_experiment("mibici-equipo08") Corrección 21OCT26

    question = question.lower()

    # --- EJECUCIÓN P1 ---
    if question == "p1":
        print(f"\n--- Ejecutando Run P1_{run_version} ---")
        p1_df = preparar_features_p1_demanda(df_viajes)
        p1_train = p1_df[(p1_df["fecha"] >= pd.to_datetime(TRAIN_START_DATE).date()) & (p1_df["fecha"] <= pd.to_datetime(TRAIN_END_DATE).date())]
        p1_test = p1_df[(p1_df["fecha"] >= pd.to_datetime(TEST_START_DATE).date()) & (p1_df["fecha"] <= pd.to_datetime(TEST_END_DATE).date())]

        features_p1 = ["origen_id", "hora", "dia_semana", "es_fin_semana"]
        X_train_p1, y_train_p1 = p1_train[features_p1], p1_train["viajes_salientes"]
        X_test_p1, y_test_p1 = p1_test[features_p1], p1_test["viajes_salientes"]

        depth_val = None if max_depth is None or max_depth <= 0 else max_depth
        
        model_p1 = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth_val, min_samples_split=2, min_samples_leaf=2, random_state=RANDOM_SEED)
        model_p1.fit(X_train_p1, y_train_p1)
        preds_p1 = model_p1.predict(X_test_p1)
        test_mae = mean_absolute_error(y_test_p1, preds_p1)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(y_test_p1.values[:100], label="Real", alpha=0.7)
        ax.plot(preds_p1[:100], label="Predicción", alpha=0.7)
        ax.set_title("P1: Comparativa Demanda Real vs Predicción")
        ax.legend()
        fig_path_p1 = FIGURES_DIR / f"p1_demanda_pred_{run_version}.png"
        plt.savefig(fig_path_p1)
        plt.close()

        registrar_run_mlflow(
            run_name=f"P1_{run_version}_{model.upper()}",
            pregunta="P1",
            commit_hash=commit_hash,
            params={
                "data_hash_version": data_hash,
                "model_type": model,
                "n_estimators": n_estimators,
                "max_depth": str(depth_val),
                "seed": RANDOM_SEED,
            },
            metrics={"test_mae": test_mae},
            artifacts=[fig_path_p1],
        )
        print(f"P1 completado | test_mae: {test_mae:.4f}")

    # --- EJECUCIÓN P2 ---
    elif question == "p2":
        print(f"\n--- Ejecutando Run P2_{run_version} ---")
        p2_df = preparar_features_p2_riesgo(df_viajes)
        p2_train = p2_df[(p2_df["fecha"] >= pd.to_datetime(TRAIN_START_DATE).date()) & (p2_df["fecha"] <= pd.to_datetime(TRAIN_END_DATE).date())]
        p2_test = p2_df[(p2_df["fecha"] >= pd.to_datetime(TEST_START_DATE).date()) & (p2_df["fecha"] <= pd.to_datetime(TEST_END_DATE).date())]

        features_p2 = ["estacion_id", "hora", "entradas", "salidas", "balance_neto", "balance_acumulado"]
        X_train_p2, y_train_p2 = p2_train[features_p2], p2_train["target_riesgo_t1"]
        X_test_p2, y_test_p2 = p2_test[features_p2], p2_test["target_riesgo_t1"]

        model_p2 = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED)
        model_p2.fit(X_train_p2, y_train_p2)
        preds_p2 = model_p2.predict(X_test_p2)
        test_recall = recall_score(y_test_p2, preds_p2, zero_division=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        pd.Series(preds_p2).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"])
        ax.set_title("P2: Distribución de Predicciones de Riesgo")
        fig_path_p2 = FIGURES_DIR / f"p2_riesgo_dist_{run_version}.png"
        plt.savefig(fig_path_p2)
        plt.close()

        registrar_run_mlflow(
            run_name=f"P2_{run_version}_{model.upper()}",
            pregunta="P2",
            commit_hash=commit_hash,
            params={
                "data_hash_version": data_hash,
                "model_type": model,
                "n_estimators": n_estimators,
                "seed": RANDOM_SEED,
            },
            metrics={"test_recall": test_recall},
            artifacts=[fig_path_p2],
        )
        print(f"P2 completado | test_recall: {test_recall:.4f}")

    # --- EJECUCIÓN P3 ---
    elif question == "p3":
        print(f"\n--- Ejecutando Run P3_{run_version} ---")
        p3_df = preparar_features_p3_clustering(df_viajes)
        X_p3 = p3_df.drop(columns=["origen_id"])

        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        clusters = kmeans.fit_predict(X_p3)
        sil_score = silhouette_score(X_p3, clusters)

        fig, ax = plt.subplots(figsize=(8, 4))
        for c in range(n_clusters):
            ax.plot(X_p3[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
        ax.set_title("P3: Perfil Horario Promedio por Cluster")
        ax.legend()
        fig_path_p3 = FIGURES_DIR / f"p3_cluster_profiles_{run_version}.png"
        plt.savefig(fig_path_p3)
        plt.close()

        registrar_run_mlflow(
            run_name=f"P3_{run_version}_KMeans",
            pregunta="P3",
            commit_hash=commit_hash,
            params={
                "data_hash_version": data_hash,
                "n_clusters": n_clusters,
                "model_type": "KMeans",
            },
            metrics={"silhouette_score": sil_score},
            artifacts=[fig_path_p3],
        )
        print(f"P3 completado | silhouette_score: {sil_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento MLOps MiBici")
    parser.add_argument("--question", type=str, default="all")
    parser.add_argument("--model", type=str, default="rf")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--run_version", type=str, default="v1")

    args = parser.parse_args()

    if args.question.lower() == "all":
        ejecutar_experimento_base(run_version="v1")
    else:
        ejecutar_por_pregunta(
            question=args.question,
            model=args.model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_clusters=args.n_clusters,
            run_version="v1",
        )
