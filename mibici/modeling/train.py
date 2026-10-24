"""Módulo de entrenamiento de modelos base (P1, P2 y P3) con tracking en MLflow."""

import json
import git
import matplotlib.pyplot as plt
import mlflow
import mlflow.data #agregado para semana 6
from mlflow.models import infer_signature
import pandas as pd
import argparse #agregado para semana 6
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, recall_score, silhouette_score
from sklearn.dummy import DummyClassifier, DummyRegressor


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


#Modificación semana 6
#----------------------

def ejecutar_por_pregunta(
    question: str = "p1",
    n_estimators: int = 300,
    max_depth: int | None = None,
    n_clusters: int = 2,
    run_version: str = "v1",
):
    """
    Ejecuta únicamente una pregunta específica del proyecto (P1, P2 o P3).
    Puede utilizarse desde MLProject o desde línea de comandos.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "viajes_clean.parquet"

    if not parquet_path.exists():
        print(
            f"Error: No existe el archivo {parquet_path}. "
            "Ejecuta primero 'python -m mibici.dataset'"
        )
        return

    df_viajes = pd.read_parquet(parquet_path)

    commit_hash = obtener_commit_hash()
    data_hash = obtener_version_datos()

    question = question.lower()

    # ============================================================
    # PREGUNTA 1 (P1): REGRESIÓN DE DEMANDA FUTURA
    # ============================================================

    if question == "p1":
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

        #ejecutar_p1_baseline(X_train_p1, y_train_p1, X_test_p1, y_test_p1, commit_hash, data_hash, run_version,)
        ejecutar_p1(X_train=X_train_p1, y_train=y_train_p1, X_test=X_test_p1, y_test=y_test_p1, commit_hash=commit_hash,
                    data_hash=data_hash, n_estimators=n_estimators, max_depth=max_depth, run_version=run_version,
                    )

    # ============================================================
    # PREGUNTA 2 (P2): CLASIFICACIÓN DE RIESGO
    # ============================================================

    elif question == "p2":

        p2_df = preparar_features_p2_riesgo(df_viajes)

        # Partición temporal estricta (SIN DATA LEAKAGE)
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

        #ejecutar_p2_baseline(X_train_p2, y_train_p2, X_test_p2, y_test_p2, commit_hash, data_hash, run_version,)
        ejecutar_p2(X_train=X_train_p2, y_train=y_train_p2, X_test=X_test_p2, y_test=y_test_p2, commit_hash=commit_hash,
            data_hash=data_hash, n_estimators=n_estimators, run_version=run_version,
            )

    # ============================================================
    # PREGUNTA 3 (P3): CLUSTERING DE PERFILES HORARIOS
    # ============================================================

    elif question == "p3":

        p3_df = preparar_features_p3_clustering(df_viajes)
        X_p3 = p3_df.drop(columns=["origen_id"])

        ejecutar_p3(X=X_p3, commit_hash=commit_hash, data_hash=data_hash, n_clusters=n_clusters, run_version=run_version,)

    else:
        raise ValueError(
            f"Pregunta no soportada: {question}. "
            "Opciones válidas: p1, p2, p3"
        )


def ejecutar_p1_baseline(X_train, y_train, X_test, y_test, commit_hash, data_hash, run_version="v1"):
    model_family="DummyRegressor"
    dummy = DummyRegressor(strategy="mean")
    
    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    fig_path = FIGURES_DIR / f"p1_dummy_baseline_{run_version}.png"

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(y_test.values[:100], label="Real")
    ax.plot(preds[:100], label="Dummy")
    ax.legend()

    plt.savefig(fig_path)
    plt.close()

    registrar_run_mlflow(
        run_name=f"P1_{run_version}_{model_family}",
        pregunta="P1",
        commit_hash=commit_hash,
        params={"data_hash_version": data_hash, "model_type": model_family},
        metrics={"test_mae": mae},
        artifacts=[fig_path]
    )

    return mae


def ejecutar_p2_baseline( X_train,y_train, X_test, y_test, commit_hash, data_hash, run_version="v1"):
    
    model_family="DummyClassifier"
    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(X_train, y_train)
    preds = dummy.predict(X_test)
    recall = recall_score(y_test, preds, zero_division=0)

    registrar_run_mlflow(
        run_name=f"P2_{run_version}_DummyClassifier",
        pregunta="P2",
        commit_hash=commit_hash,
        params={"model_type":model_family, "data_hash_version":data_hash},
        metrics={"test_recall":recall}
    )

    return recall


def ejecutar_p1(X_train, y_train, X_test, y_test, commit_hash, data_hash, n_estimators=300, max_depth=None, run_version="v1"):
    
    model_family="RandomForestRegressor"
    # 1. Corrida padre para la búsqueda de hiperparámetros (grid search) con RandomForestRegressor
    with mlflow.start_run(run_name=f"P1_Search_{model_family.upper()}", nested=False):
        #Etiquetas obligatorias de equipo y filtrado solicitadas
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("question", "p1")
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("git_commit", commit_hash)
        
        #Variables para rastrear el mejor modelo
        best_mae = float("inf")

        best_model = None
        best_n_est = None
        best_depth = None

        #Definir el grid de hiperparámetros a probar (al menos tres variantes adicionales al baseline)
        estimators_grid = [100, 200, n_estimators]
        depths_grid = [None, 10, 20]

        for n_est in estimators_grid:
            for d_val in depths_grid:
                #2. CORRIDAS HIJAS por cada combinación
                with mlflow.start_run(run_name=f"p1_rf_est_{n_est}_depth_{d_val}", nested=True):
                    mlflow.set_tag("equipo", "equipo08")
                    mlflow.set_tag("question", "p1")
                    mlflow.set_tag("model_family", model_family)
                    mlflow.set_tag("stage_of_work", "search")
                    mlflow.set_tag("git_commit", commit_hash)

                    depth_val = None if d_val is None or d_val <= 0 else d_val

                    # Entrenamiento
                    model = RandomForestRegressor(n_estimators=n_est, max_depth=depth_val, min_samples_split=2,
                                                     min_samples_leaf=2, random_state=RANDOM_SEED)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    test_mae = mean_absolute_error(y_test, preds)
                    
                    # Asignar el mejor resultado
                    if test_mae < best_mae:
                        best_mae = test_mae
                        best_model = model
                        best_n_est = n_est
                        best_depth = depth_val

                    # Registrar parámetros y métricas en la corrida hija
                    mlflow.log_param("data_hash_version", data_hash)
                    mlflow.log_param("model_type", model_family)
                    mlflow.log_param("n_estimators", n_est)
                    mlflow.log_param("max_depth", str(depth_val))
                    mlflow.log_param("seed", RANDOM_SEED)
                    mlflow.log_metric("test_mae", test_mae)

                    # SE AGREGAN LOS ARTEFACTOS (INCISO 5)
                    
                    # A. NOTA: El modelo ya no se registra para todas las corridas, ahora se guarda solamente para el mejor resultado
                    
                    # B. Registro de la versión de datos con log_input
                    train_dataset = mlflow.data.from_pandas(
                        df=pd.concat([X_train, y_train], axis=1),
                        source="viajes_clean.parquet",
                        name="mibici_train_p1",
                        targets="viajes_salientes"
                    )
                    mlflow.log_input(train_dataset, context="training")

                    # C. Gráfica de desempeño organizada en subcarpeta plots/
                    plots_dir = FIGURES_DIR / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    fig_path = plots_dir / f"p1_demanda_pred_est_{n_est}_depth_{d_val}.png"
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(y_test.values[:100], label="Real", alpha=0.7)
                    ax.plot(preds[:100], label="Predicción", alpha=0.7)
                    ax.set_title(f"P1: Real vs Pred (est={n_est}, depth={d_val})")
                    ax.legend()
                    plt.savefig(fig_path)
                    plt.close()
                    
                    mlflow.log_artifact(str(fig_path), artifact_path="plots")

                    # D. Manifiesto de datos en subcarpeta data/
                    manifest_path = PROCESSED_DATA_DIR / "data_manifest.json"
                    if manifest_path.exists():
                        mlflow.log_artifact(str(manifest_path), artifact_path="data")

                    # E. Archivo de configuración en subcarpeta config/
                    config_dir = PROCESSED_DATA_DIR.parent / "configs"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    config_file_path = config_dir / f"config_p1_est_{n_est}_depth_{d_val}.json"
                    
                    config_dict = {
                        "question": "p1",
                        "model_type": model_family,
                        "n_estimators": n_est,
                        "max_depth": str(depth_val),
                        "seed": RANDOM_SEED,
                        "train_start": TRAIN_START_DATE,
                        "train_end": TRAIN_END_DATE
                    }
                    with open(config_file_path, "w", encoding="utf-8") as f:
                        json.dump(config_dict, f, indent=4)
                        
                    mlflow.log_artifact(str(config_file_path), artifact_path="config")
                    # =========================================================

                    print(f"Hija [n_est={n_est}, depth={depth_val}] completada | test_mae: {test_mae:.4f}")

    print("Búsqueda anidada de P1 finalizada con éxito")
    
    # 6. Se registra en MLflow el mejor modelo encontrado
    print(
        f"Mejor configuración: "
        f"n_estimators={best_n_est}, "
        f"max_depth={best_depth}, "
        f"MAE={best_mae:.4f}"
    )
    
    signature = infer_signature(X_test, best_model.predict(X_test))
    input_example = X_test.head(5)
    mlflow.sklearn.log_model(sk_model=best_model, name="best_model", signature=signature, input_example=input_example, skops_trusted_types=["sklearn.tree._tree.Tree"])


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test.values[:100], label="Real", alpha=0.7)
    ax.plot(preds[:100], label="Predicción", alpha=0.7)
    ax.set_title("P1: Comparativa Demanda Real vs Predicción")
    ax.legend()
    fig_path = FIGURES_DIR / f"p1_demanda_pred_{run_version}.png"
    plt.savefig(fig_path)
    plt.close()

    registrar_run_mlflow(
        run_name=f"P1_{run_version}_{model_family.upper()}",
        pregunta="P1",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "model_type": model_family,
            "n_estimators": best_n_est,
            "max_depth": str(best_depth),
            "seed": RANDOM_SEED,
        },
        metrics={"test_mae": best_mae},
        artifacts=[fig_path],
    )
    print(f"P1 completado | test_mae: {best_mae:.4f}")


def ejecutar_p2(X_train, y_train, X_test, y_test, commit_hash, data_hash, n_estimators=50, run_version="v1"):

    model_family="RandomForestClassifier"
    # 1. CORRIDA PADRE (Parent Run) para la búsqueda
    with mlflow.start_run(run_name=f"P2_Search_{model_family.upper()}", nested=False):
        #Etiquetas obligatorias de equipo y filtrado solicitadas
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("question", "p2")
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("git_commit", commit_hash)
        
        #Variables para rastrear el mejor modelo
        best_recall = -1

        best_model = None
        best_n_est = None
        best_depth = None

    
        #Definir el grid de hiperparámetros a probas (al menos tres variantes adicionales al baseline)
        estimators_grid = [100, 200, n_estimators]
        depths_grid = [None, 10, 20]
    
        for n_est in estimators_grid:
            for d_val in depths_grid:
                #2. CORRIDAS HIJAS por cada combinación
                with mlflow.start_run(run_name=f"p2_rf_est_{n_est}_depth_{d_val}", nested=True):
                    mlflow.set_tag("equipo", "equipo08")
                    mlflow.set_tag("question", "p2")
                    mlflow.set_tag("model_family", model_family)
                    mlflow.set_tag("stage_of_work", "search")
                    mlflow.set_tag("git_commit", commit_hash)
    
                    depth_val = None if d_val is None or d_val <= 0 else d_val
    
                    # Entrenamiento
                    model = RandomForestClassifier(
                        n_estimators=n_est, 
                        max_depth=depth_val, 
                        min_samples_split=2, 
                        min_samples_leaf=2, 
                        random_state=RANDOM_SEED
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    test_recall = recall_score(y_test, preds, zero_division=0)

                    # Asignar el mejor resultado
                    if test_recall > best_recall:
                        best_recall = test_recall
                        best_model = model
                        best_n_est = n_est
                        best_depth = depth_val

    
                    # Registrar parámetros y métricas en la corrida hija
                    mlflow.log_param("data_hash_version", data_hash)
                    mlflow.log_param("model_type", model_family)
                    mlflow.log_param("n_estimators", n_est)
                    mlflow.log_param("max_depth", str(depth_val))
                    mlflow.log_param("seed", RANDOM_SEED)
                    mlflow.log_metric("test_recall", test_recall)

                    # SE AGREGAN LOS ARTEFACTOS DEL INCISO 5
                    
                    # A. NOTA: El modelo ya no se registra para todas las corridas, ahora se guarda solamente para el mejor resultado

                    # B. Registro de la versión de datos con log_input
                    train_dataset = mlflow.data.from_pandas(
                        df=pd.concat([X_train, y_train], axis=1),
                        source="viajes_clean.parquet",
                        name="mibici_train_p2",
                        targets="target_riesgo_t1"
                    )
                    mlflow.log_input(train_dataset, context="training")

                    # C. Gráfica de desempeño organizada en subcarpeta plots/
                    plots_dir = FIGURES_DIR / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    fig_path = plots_dir / f"p2_riesgo_t1_pred_est_{n_est}_depth_{d_val}.png"
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(y_test.values[:100], label="Real", alpha=0.7)
                    ax.plot(preds[:100], label="Predicción", alpha=0.7)
                    ax.set_title(f"P2: Real vs Pred (est={n_est}, depth={d_val})")
                    ax.legend()
                    plt.savefig(fig_path)
                    plt.close()
                    
                    mlflow.log_artifact(str(fig_path), artifact_path="plots")

                    # D. Manifiesto de datos en subcarpeta data/
                    manifest_path = PROCESSED_DATA_DIR / "data_manifest.json"
                    if manifest_path.exists():
                        mlflow.log_artifact(str(manifest_path), artifact_path="data")

                    # E. Archivo de configuración en subcarpeta config/
                    config_dir = PROCESSED_DATA_DIR.parent / "configs"
                    config_dir.mkdir(parents=True, exist_ok=True)
                    config_file_path = config_dir / f"config_p2_est_{n_est}_depth_{d_val}.json"
                    
                    config_dict = {
                        "question": "p2",
                        "model_type": model_family,
                        "n_estimators": n_est,
                        "max_depth": str(depth_val),
                        "seed": RANDOM_SEED,
                        "train_start": TRAIN_START_DATE,
                        "train_end": TRAIN_END_DATE
                    }
                    with open(config_file_path, "w", encoding="utf-8") as f:
                        json.dump(config_dict, f, indent=4)
                        
                    mlflow.log_artifact(str(config_file_path), artifact_path="config")
                    # =========================================================
    
                    print(f"Hija [n_est={n_est}, depth={depth_val}] completada | test_recall: {test_recall:.4f}")
    
    print("Búsqueda anidada de P2 finalizada con éxito")
    
    # 6. Se registra en MLflow el mejor modelo encontrado
    print(
        f"Mejor configuración: "
        f"n_estimators={best_n_est}, "
        f"max_depth={best_depth}, "
        f"Recall={best_recall:.4f}"
    )
    
    signature = infer_signature(X_test, best_model.predict(X_test))
    input_example = X_test.head(5)
    mlflow.sklearn.log_model(sk_model=best_model, name="best_model", signature=signature, input_example=input_example, skops_trusted_types=["sklearn.tree._tree.Tree"])


    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(preds).value_counts().plot(kind="bar", ax=ax, color=["skyblue", "orange"])
    ax.set_title("P2: Distribución de Predicciones de Riesgo")
    fig_path = FIGURES_DIR / f"p2_riesgo_dist_{run_version}.png"
    plt.savefig(fig_path)
    plt.close()

    registrar_run_mlflow(
        run_name=f"P2_{run_version}_{model_family.upper()}",
        pregunta="P2",
        commit_hash=commit_hash,
        params={
            "data_hash_version": data_hash,
            "model_type": model_family,
            "n_estimators": best_n_est,
            "seed": RANDOM_SEED,
        },
        metrics={"test_recall": best_recall},
        artifacts=[fig_path],
    )
    print(f"P2 completado | test_recall: {best_recall:.4f}")

def ejecutar_p3(X, commit_hash, data_hash, n_clusters=2, run_version="v1"):
    
    model_family="KMeans"
    # 1. CORRIDA PADRE (Parent Run) para agrupar la búsqueda de clustering
    with mlflow.start_run(run_name=f"P3_Search_{model_family}", nested=False):
        # Etiquetas obligatorias y de filtrado requeridas
        mlflow.set_tag("equipo", "equipo08")
        mlflow.set_tag("question", "p3")
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("stage_of_work", "search")
        mlflow.set_tag("git_commit", commit_hash)
        
        #Variables para rastrear el mejor modelo
        best_score = -1
        best_model = None
        best_k = None

        # Rejilla de hiperparámetros a explorar (variantes adicionales al baseline de 2 clústeres)
        clusters_grid = [2, 3, 4, 5]

        for k in clusters_grid:
            # 2. CORRIDAS HIJAS por cada valor de K
            with mlflow.start_run(run_name=f"p3_kmeans_clusters_{k}", nested=True):
                mlflow.set_tag("equipo", "equipo08")
                mlflow.set_tag("question", "p3")
                mlflow.set_tag("model_family", model_family)
                mlflow.set_tag("stage_of_work", "search")
                mlflow.set_tag("git_commit", commit_hash)

                # Entrenamiento de KMeans
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
                clusters = kmeans.fit_predict(X)
                sil_score = silhouette_score(X, clusters)
                
                # Asignar el mejor resultado
                if sil_score > best_score:
                    best_score = sil_score
                    best_model = kmeans
                    best_k = k

                # Generar gráfica de perfiles dinámicamente según K
                fig, ax = plt.subplots(figsize=(8, 4))
                for c in range(k):
                    ax.plot(X[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
                ax.set_title(f"P3: Perfil Horario Promedio por Cluster (K={k})")
                ax.legend()
                fig_path = FIGURES_DIR / f"p3_cluster_profiles_{run_version}_k{k}.png"
                plt.savefig(fig_path)
                plt.close()

                # Registrar parámetros y métricas en la corrida hija
                mlflow.log_param("data_hash_version", data_hash)
                mlflow.log_param("n_clusters", k)
                mlflow.log_param("model_type", model_family)
                mlflow.log_param("seed", RANDOM_SEED)
                mlflow.log_metric("silhouette_score", sil_score)
                

                # SE AGREGAN LOS ARTEFACTOS DEL INCISO 5
                # A. Modelo con firma (signature) y ejemplo de entrada (Se mantiene para KMeans, ya que no representa reto de procesamiento)
                signature = infer_signature(X, clusters)
                input_example = X.head(5)
                mlflow.sklearn.log_model(sk_model=kmeans, name="model", signature=signature, input_example=input_example, skops_trusted_types=["sklearn.tree._tree.Tree"])

                # B. Registro de la versión de datos con log_input
                train_dataset = mlflow.data.from_pandas(
                    df=X,
                    source="viajes_clean.parquet",
                    name="mibici_clustering_p3"
                )
                mlflow.log_input(train_dataset, context="training")

                # C. Gráfica de perfiles organizada en subcarpeta plots/
                plots_dir = FIGURES_DIR / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                fig_path = plots_dir / f"p3_cluster_profiles_{run_version}_k{k}.png"
                
                fig, ax = plt.subplots(figsize=(8, 4))
                for c in range(k):
                    ax.plot(X[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
                ax.set_title(f"P3: Perfil Horario Promedio por Cluster (K={k})")
                ax.legend()
                plt.savefig(fig_path)
                plt.close()
                
                mlflow.log_artifact(str(fig_path), artifact_path="plots")

                # D. Manifiesto de datos en subcarpeta data/
                manifest_path = PROCESSED_DATA_DIR / "data_manifest.json"
                if manifest_path.exists():
                    mlflow.log_artifact(str(manifest_path), artifact_path="data")

                # E. Archivo de configuración en subcarpeta config/
                config_dir = PROCESSED_DATA_DIR.parent / "configs"
                config_dir.mkdir(parents=True, exist_ok=True)
                config_file_path = config_dir / f"config_p3_k{k}.json"
                
                config_dict = {
                    "question": "p3",
                    "model_type": model_family,
                    "n_clusters": k,
                    "seed": RANDOM_SEED
                }
                with open(config_file_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=4)
                    
                mlflow.log_artifact(str(config_file_path), artifact_path="config")
                # =========================================================

                print(f"Hija K-Means [n_clusters={k}] completada | silhouette_score: {sil_score:.4f}")

    print("Búsqueda anidada de P3 finalizada con éxito.")
    print(
    f"Mejor K encontrado: {best_k} "
    f"(silhouette={best_score:.4f})"
)

    # Recalcular con los mejores hiperparámetros
    
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    clusters = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, clusters)

    fig, ax = plt.subplots(figsize=(8, 4))
    for c in range(n_clusters):
        ax.plot(X[clusters == c].mean(axis=0).values, label=f"Cluster {c}")
    ax.set_title("P3: Perfil Horario Promedio por Cluster")
    ax.legend()
    fig_path = FIGURES_DIR / f"p3_cluster_profiles_{run_version}.png"
    plt.savefig(fig_path)
    plt.close()

    registrar_run_mlflow(
        run_name=f"P3_{run_version}_{model_family}",
        pregunta="P3",
        commit_hash=commit_hash,
        params={"data_hash_version": data_hash, "n_clusters": n_clusters, "model_type": "KMeans",},
        metrics={"silhouette_score": sil_score},
        artifacts=[fig_path],
    )
    print(f"P3 completado | silhouette_score: {sil_score:.4f}")


def ejecutar_experimento_base(run_version: str = "v1"):
    """Ejecuta los modelos base de P1, P2 y P3 registrando hiperparámetros, métricas y artefactos en MLflow.

    Args:
        run_version (str): Identificador de versión secuencial para la corrida (ej. 'v1', 'v2').
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROCESSED_DATA_DIR / "viajes_clean.parquet"
    if not parquet_path.exists():
        print(
            f"Error: No existe el archivo {parquet_path}. "
            "Ejecuta primero 'python -m mibici.dataset'"
        )
        return
    df_viajes = pd.read_parquet(parquet_path)

    commit_hash = obtener_commit_hash()
    data_hash = obtener_version_datos()

    # ============================================================
    # PREGUNTA 1 (P1): REGRESIÓN DE DEMANDA FUTURA
    # ============================================================

    print("\n========== P1 ==========")
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

    ejecutar_p1_baseline(X_train=X_train_p1, y_train=y_train_p1, X_test=X_test_p1, y_test=y_test_p1, commit_hash=commit_hash,
                         data_hash=data_hash, run_version=run_version,)
    ejecutar_p1(X_train=X_train_p1, y_train=y_train_p1, X_test=X_test_p1, y_test=y_test_p1, commit_hash=commit_hash,
                data_hash=data_hash, n_estimators=300, max_depth=None, run_version=run_version,)

    # ============================================================
    # PREGUNTA 2 (P2): CLASIFICACIÓN DE RIESGO
    # ============================================================

    print("\n========== P2 ==========")
    p2_df = preparar_features_p2_riesgo(df_viajes)

    # Partición temporal estricta (SIN DATA LEAKAGE)
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

    ejecutar_p2_baseline(X_train=X_train_p2, y_train=y_train_p2, X_test=X_test_p2, y_test=y_test_p2,
                         commit_hash=commit_hash, data_hash=data_hash, run_version=run_version,)
    ejecutar_p2(X_train=X_train_p2, y_train=y_train_p2, X_test=X_test_p2, y_test=y_test_p2, commit_hash=commit_hash,
                data_hash=data_hash, n_estimators=50, run_version=run_version,)

    # ============================================================
    # PREGUNTA 3 (P3): CLUSTERING DE PERFILES HORARIOS
    # ============================================================

    print("\n========== P3 ==========")
    p3_df = preparar_features_p3_clustering(df_viajes)
    X_p3 = p3_df.drop(columns=["origen_id"])

    ejecutar_p3(X=X_p3, commit_hash=commit_hash, data_hash=data_hash, n_clusters=2, run_version=run_version,)
    print("\nEjecución completa finalizada.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento MLOps MiBici")
    parser.add_argument("--question", type=str, default="all")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--n_clusters", type=int, default=2)
    parser.add_argument("--run_version", type=str, default="v1")

    args = parser.parse_args()

    if args.question.lower() == "all":
        ejecutar_experimento_base(run_version="v1")
    else:
        ejecutar_por_pregunta(question=args.question, n_estimators=args.n_estimators,
                              max_depth=args.max_depth, n_clusters=args.n_clusters, run_version=args.run_version,)