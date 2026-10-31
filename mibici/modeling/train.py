"""Entrenamiento de P1, P2 y P3 con tracking en MLflow (se lanza con `mlflow run`).

Estructura de runs en el experimento mibici-equipo08:

    run padre  (el que crea `mlflow run`; role=parent)
     ├─ modo baseline : 1 hija   stage_of_work=baseline
     ├─ modo single   : 1 hija   stage_of_work=candidate
     └─ modo search   : N hijas  stage_of_work=search     (una por combinación)
                        +3 hijas stage_of_work=candidate  (las mejores, con artefactos)

Las hijas de búsqueda se evalúan en VALIDACIÓN (abril). Las candidatas se
reentrenan con todo el periodo de entrenamiento y se evalúan en TEST (mayo),
que es la misma partición del baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")  # sin ventanas: necesario al correr dentro de `mlflow run`
import matplotlib.pyplot as plt
import mlflow
import mlflow.data
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import mibici.config as cfg
from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
    VAL_START_DATE,
)
from mibici.features import (
    preparar_features_p1_demanda,
    preparar_features_p2_riesgo,
    preparar_features_p3_clustering,
)

try:  # gitpython es opcional: sin él el commit queda como "sin_commit_git"
    import git
except ImportError:
    git = None

PARQUET_PATH = PROCESSED_DATA_DIR / "viajes_clean.parquet"
MANIFEST_PATH = PROCESSED_DATA_DIR / "data_manifest.json"
N_CANDIDATAS = 3

FEATURES = {
    "p1": ["origen_id", "hora", "dia_semana", "es_fin_semana"],
    "p2": ["estacion_id", "hora", "entradas", "salidas", "balance_neto", "balance_acumulado"],
}
TARGETS = {"p1": "viajes_salientes", "p2": "target_riesgo_t1"}
FAMILIAS = {"p1": ["rf", "hgb"], "p2": ["rf", "hgb", "logreg"], "p3": ["kmeans"]}

# Métrica con la que se elige la mejor configuración: (clave en MLflow, dirección)
# Se usa la de VALIDACIÓN para no elegir con el test.
METRICA_SELECCION = {
    "p1": ("val_mae", "min"),
    "p2": ("val_recall", "max"),
    "p3": ("silhouette_score", "max"),
}


# --------------------------------------------------------------------------- #
# Utilidades de trazabilidad
# --------------------------------------------------------------------------- #
def obtener_commit_hash() -> str:
    """Hash SHA-1 del commit actual, o 'sin_commit_git' si no hay repositorio."""
    if git is None:
        return "sin_commit_git"
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "sin_commit_git"


def obtener_version_datos() -> str:
    """Primeros 12 caracteres del SHA256 guardado en el manifiesto de datos."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("hash_sha256", "desconocido")[:12]
        except (OSError, ValueError, AttributeError, TypeError) as e:  # ilegible o formato inesperado
            print(f"AVISO: no se pudo leer {MANIFEST_PATH.name}: {e}")
    return "desconocido"


def configurar_mlflow() -> None:
    """Fija el servidor de tracking y, fuera de `mlflow run`, el experimento.

    Dentro de `mlflow run` el experimento lo define la opción --experiment-name
    (MLflow exporta MLFLOW_RUN_ID), por eso no se llama set_experiment ahí.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
@dataclass
class Datos:
    """Particiones temporales (P1/P2) o matriz de perfiles (P3)."""

    question: str
    target: str | None = None
    X_sub: pd.DataFrame | None = None  # ene-mar: entrenamiento de la búsqueda
    y_sub: pd.Series | None = None
    X_val: pd.DataFrame | None = None  # abril: validación
    y_val: pd.Series | None = None
    X_train: pd.DataFrame | None = None  # ene-abr: entrenamiento final
    y_train: pd.Series | None = None
    X_test: pd.DataFrame | None = None  # mayo: prueba
    y_test: pd.Series | None = None
    X_all: pd.DataFrame | None = None  # P3
    estaciones: pd.Series | None = None  # P3


def cargar_datos(question: str) -> Datos:
    """Construye las features de la pregunta y las particiona por fecha."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"No existe {PARQUET_PATH}. Ejecuta primero: mlflow run . -e ingest"
        )
    df = pd.read_parquet(PARQUET_PATH)

    if question == "p3":
        p3 = preparar_features_p3_clustering(df)
        return Datos(
            question,
            X_all=p3.drop(columns=["origen_id"]).astype("float64"),
            estaciones=p3["origen_id"],
        )

    if question == "p1":
        base = preparar_features_p1_demanda(df)
    else:
        base = preparar_features_p2_riesgo(df)
        base[TARGETS["p2"]] = base[TARGETS["p2"]].astype(int)

    fecha = base["fecha"]

    def dia(s: str):
        return pd.to_datetime(s).date()

    m_train = (fecha >= dia(TRAIN_START_DATE)) & (fecha <= dia(TRAIN_END_DATE))
    m_sub = (fecha >= dia(TRAIN_START_DATE)) & (fecha < dia(VAL_START_DATE))
    m_val = (fecha >= dia(VAL_START_DATE)) & (fecha <= dia(TRAIN_END_DATE))
    m_test = (fecha >= dia(TEST_START_DATE)) & (fecha <= dia(TEST_END_DATE))

    X = base[FEATURES[question]].astype("float64")  # float: evita avisos de firma en MLflow
    y = base[TARGETS[question]]
    d = Datos(
        question,
        target=TARGETS[question],
        X_sub=X[m_sub], y_sub=y[m_sub],
        X_val=X[m_val], y_val=y[m_val],
        X_train=X[m_train], y_train=y[m_train],
        X_test=X[m_test], y_test=y[m_test],
    )
    for nombre in ("X_sub", "X_val", "X_train", "X_test"):
        if len(getattr(d, nombre)) == 0:
            raise ValueError(f"La partición {nombre} quedó vacía: revisa las fechas en config.py")
    return d


# --------------------------------------------------------------------------- #
# Modelos e hiperparámetros
# --------------------------------------------------------------------------- #
def _profundidad(valor) -> int | None:
    """-1 / 0 / None significan 'sin límite' (scikit-learn no acepta -1)."""
    return None if valor is None or valor <= 0 else int(valor)


def construir_modelo(question: str, familia: str, hp: dict):
    """Crea el estimador de scikit-learn para (pregunta, familia, hiperparámetros)."""
    n = int(hp.get("n_estimators", 100))
    prof = _profundidad(hp.get("max_depth"))

    if question == "p1":
        if familia == "dummy":
            return DummyRegressor(strategy="mean")
        if familia == "rf":
            return RandomForestRegressor(
                n_estimators=n, max_depth=prof, min_samples_split=2, min_samples_leaf=2,
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        if familia == "hgb":
            return HistGradientBoostingRegressor(max_iter=n, max_depth=prof, random_state=RANDOM_SEED)

    if question == "p2":
        if familia == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if familia == "rf":
            return RandomForestClassifier(
                n_estimators=n, max_depth=prof, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        if familia == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=n, max_depth=prof, class_weight="balanced", random_state=RANDOM_SEED
            )
        if familia == "logreg":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
            )

    if question == "p3" and familia == "kmeans":
        return KMeans(n_clusters=int(hp["n_clusters"]), random_state=RANDOM_SEED, n_init=10)

    raise ValueError(f"Combinación no soportada: {question}/{familia}")


def rejilla(question: str, familia: str, n_estimators: int) -> list[dict]:
    """Combinaciones de hiperparámetros a probar (editar aquí para ampliarla)."""
    if question == "p3":
        return [{"n_clusters": k} for k in range(2, 7)]
    if familia == "logreg":
        return [{}]
    estimadores = sorted({100, int(n_estimators)})
    profundidades = {"rf": [None, 10, 20], "hgb": [None, 6, 10]}[familia]
    return [{"n_estimators": e, "max_depth": p} for e in estimadores for p in profundidades]


def nombre_corrida(question: str, familia: str, hp: dict) -> str:
    """Nombre legible y único por configuración (reemplaza al antiguo 'v1')."""
    partes = [f"{k.replace('n_estimators', 'n').replace('max_depth', 'd').replace('n_clusters', 'k')}{v}"
              for k, v in hp.items()]
    return "_".join([question, familia, *partes])


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metrica_validacion(question: str, modelo, datos: Datos) -> float:
    """Métrica principal en abril, con el modelo entrenado en ene-mar."""
    pred = modelo.predict(datos.X_val)
    if question == "p1":
        return float(mean_absolute_error(datos.y_val, pred))
    return float(recall_score(datos.y_val, pred, zero_division=0))


def metricas_test(question: str, modelo, datos: Datos) -> tuple[dict, np.ndarray]:
    """Métricas en mayo y las predicciones (se reutilizan para gráficas y firma)."""
    pred = modelo.predict(datos.X_test)
    y = datos.y_test
    if question == "p1":
        return {
            "test_mae": float(mean_absolute_error(y, pred)),
            "test_rmse": float(mean_squared_error(y, pred) ** 0.5),
        }, pred
    proba = modelo.predict_proba(datos.X_test)
    score = proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(y))
    return {
        "test_recall": float(recall_score(y, pred, zero_division=0)),
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_f1": float(f1_score(y, pred, zero_division=0)),
        "test_pr_auc": float(average_precision_score(y, score)),
    }, pred


def metricas_clustering(X: pd.DataFrame, etiquetas: np.ndarray, modelo) -> dict:
    return {
        "silhouette_score": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "inertia": float(modelo.inertia_),
    }


# --------------------------------------------------------------------------- #
# Etiquetas, parámetros, gráficas y artefactos
# --------------------------------------------------------------------------- #
def etiquetar(question: str, familia: str, stage: str, ctx: dict, rol: str) -> None:
    mlflow.set_tags({
        "equipo": "equipo08",
        "question": question,
        "model_family": familia,
        "stage_of_work": stage,
        "run_role": rol,
        "git_commit": ctx["commit"],
    })


def parametros(question: str, familia: str, hp: dict, modelo, ctx: dict) -> dict:
    ultimo = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
    p = {
        "data_hash_version": ctx["data_hash"],
        "model_family": familia,
        "model_type": type(ultimo).__name__,
        "seed": RANDOM_SEED,
        "train_start": TRAIN_START_DATE,
        "train_end": TRAIN_END_DATE,
        "val_start": VAL_START_DATE,
        "test_start": TEST_START_DATE,
        "test_end": TEST_END_DATE,
    }
    p.update({k: str(v) for k, v in hp.items()})
    return p


def graficar(question: str, modelo, datos: Datos, pred: np.ndarray, titulo: str, destino: Path):
    """Gráficas acordes a la tarea: P1 real/predicho y residuales, P2 matriz y PR, P3 perfiles."""
    if question == "p1":
        y = datos.y_test.to_numpy()
        idx = np.random.default_rng(RANDOM_SEED).choice(len(y), size=min(3000, len(y)), replace=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y[idx], pred[idx], s=6, alpha=0.4)
        tope = float(max(y.max(), pred.max()))
        ax.plot([0, tope], [0, tope], "r--", label="ideal")
        ax.set(xlabel="Real", ylabel="Predicho", title=f"P1 real vs predicho\n{titulo}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(destino / "real_vs_predicho.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(y - pred, bins=50)
        ax.set(xlabel="Residual (real - predicho)", ylabel="Frecuencia", title=f"P1 residuales\n{titulo}")
        fig.tight_layout()
        fig.savefig(destino / "residuales.png", dpi=110)
        plt.close(fig)

    elif question == "p2":
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(datos.y_test, pred, ax=ax, colorbar=False)
        ax.set_title(f"P2 matriz de confusión\n{titulo}")
        fig.tight_layout()
        fig.savefig(destino / "matriz_confusion.png", dpi=110)
        plt.close(fig)

        proba = modelo.predict_proba(datos.X_test)
        score = proba[:, 1] if proba.shape[1] == 2 else np.zeros(len(datos.y_test))
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(datos.y_test, score, ax=ax)
        ax.set_title(f"P2 curva precisión-recall\n{titulo}")
        fig.tight_layout()
        fig.savefig(destino / "curva_precision_recall.png", dpi=110)
        plt.close(fig)

    else:  # p3: pred = etiquetas de cluster
        X = datos.X_all
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in sorted(set(pred)):
            ax.plot(X[pred == c].mean(axis=0).to_numpy(), label=f"Cluster {c} (n={(pred == c).sum()})")
        ax.set(xlabel="Hora del día", ylabel="% de viajes", title=f"P3 perfil horario por grupo\n{titulo}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(destino / "perfiles_clusters.png", dpi=110)
        plt.close(fig)


def registrar_dataset(datos: Datos) -> None:
    """Versión de los datos con mlflow.log_input (solo estaciones y horas, sin datos personales)."""
    try:
        if datos.question == "p3":
            ds = mlflow.data.from_pandas(datos.X_all, source=str(PARQUET_PATH), name="mibici_perfiles_p3")
        else:
            df = pd.concat([datos.X_train, datos.y_train], axis=1)
            ds = mlflow.data.from_pandas(
                df, source=str(PARQUET_PATH), name=f"mibici_train_{datos.question}", targets=datos.target
            )
        mlflow.log_input(ds, context="training")
    except Exception as e:  # no tumbar una corrida larga por el dataset
        print(f"AVISO: no se pudo registrar el dataset con log_input: {e}")


def registrar_modelo(modelo, datos: Datos, pred: np.ndarray) -> None:
    """Guarda el modelo con firma y ejemplo de entrada."""
    X = datos.X_all if datos.question == "p3" else datos.X_test
    firma = infer_signature(X.head(100), pred[:100])
    ejemplo = X.head(5)
    try:
        info = mlflow.sklearn.log_model(
            sk_model=modelo, name="model", signature=firma, input_example=ejemplo,
            serialization_format="cloudpickle",  # skops (default de MLflow 3.x) rechaza los árboles
        )
    except TypeError:  # MLflow < 3 usa artifact_path
        info = mlflow.sklearn.log_model(
            sk_model=modelo, artifact_path="model", signature=firma, input_example=ejemplo,
            serialization_format="cloudpickle",
        )
    uri = getattr(info, "model_uri", None)
    if uri:
        mlflow.set_tag("logged_model_uri", uri)  # respaldo para register.py


def guardar_artefactos(question, modelo, datos, pred, familia, hp, ctx, titulo) -> None:
    """Sube plots/, data/ y config/ desde una carpeta temporal propia de cada corrida."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        plots, datad, cfgd = tmp / "plots", tmp / "data", tmp / "config"
        for carpeta in (plots, datad, cfgd):
            carpeta.mkdir()

        graficar(question, modelo, datos, pred, titulo, plots)

        if MANIFEST_PATH.exists():
            shutil.copy(MANIFEST_PATH, datad / "data_manifest.json")
        if question == "p3":
            pd.DataFrame({"origen_id": datos.estaciones, "cluster": pred}).to_csv(
                datad / "estaciones_cluster.csv", index=False
            )

        (cfgd / "run_config.json").write_text(
            json.dumps({"question": question, "model_family": familia, "hyperparameters": hp,
                        "seed": RANDOM_SEED, "train": [TRAIN_START_DATE, TRAIN_END_DATE],
                        "validation_start": VAL_START_DATE, "test": [TEST_START_DATE, TEST_END_DATE],
                        "data_hash_version": ctx["data_hash"], "git_commit": ctx["commit"]},
                       indent=2, default=str),
            encoding="utf-8",
        )
        shutil.copy(Path(cfg.__file__), cfgd / "config.py")

        mlflow.log_artifacts(str(plots), artifact_path="plots")
        mlflow.log_artifacts(str(datad), artifact_path="data")
        mlflow.log_artifacts(str(cfgd), artifact_path="config")


# --------------------------------------------------------------------------- #
# Corridas
# --------------------------------------------------------------------------- #
def correr_hija_busqueda(question, familia, hp, datos, ctx) -> dict:
    """Una combinación de la búsqueda: ligera (sin modelo), evaluada en validación."""
    with mlflow.start_run(run_name=nombre_corrida(question, familia, hp), nested=True) as run:
        etiquetar(question, familia, "search", ctx, "child")
        modelo = construir_modelo(question, familia, hp)
        mlflow.log_params(parametros(question, familia, hp, modelo, ctx))

        if question == "p3":
            etiquetas = modelo.fit_predict(datos.X_all)
            metricas = metricas_clustering(datos.X_all, etiquetas, modelo)
        else:
            modelo.fit(datos.X_sub, datos.y_sub)
            metricas = {METRICA_SELECCION[question][0]: metrica_validacion(question, modelo, datos)}
        mlflow.log_metrics(metricas)

        clave = METRICA_SELECCION[question][0]
        print(f"  [{familia}] {hp} -> {clave} = {metricas[clave]:.4f}")
        return {"familia": familia, "hp": hp, "valor": metricas[clave], "run_id": run.info.run_id}


def correr_final(question, familia, hp, datos, ctx, stage, val_valor=None) -> None:
    """Corrida con modelo, firma, datos, gráficas y configuración (baseline o candidata)."""
    with mlflow.start_run(run_name=f"{stage}_{nombre_corrida(question, familia, hp)}", nested=True):
        etiquetar(question, familia, stage, ctx, "child")
        modelo = construir_modelo(question, familia, hp)
        mlflow.log_params(parametros(question, familia, hp, modelo, ctx))
        registrar_dataset(datos)

        if question == "p3":
            pred = modelo.fit_predict(datos.X_all)
            metricas = metricas_clustering(datos.X_all, pred, modelo)
        else:
            if val_valor is None:  # corrida suelta: calcula la validación aquí
                previo = construir_modelo(question, familia, hp).fit(datos.X_sub, datos.y_sub)
                val_valor = metrica_validacion(question, previo, datos)
            modelo.fit(datos.X_train, datos.y_train)
            metricas, pred = metricas_test(question, modelo, datos)
            metricas[METRICA_SELECCION[question][0]] = val_valor
        mlflow.log_metrics(metricas)

        titulo = f"{familia} {hp}" if hp else familia
        guardar_artefactos(question, modelo, datos, pred, familia, hp, ctx, titulo)
        registrar_modelo(modelo, datos, pred)
        resumen = ", ".join(f"{k}={v:.4f}" for k, v in metricas.items())
        print(f"  [{stage}] {familia} {hp} -> {resumen}")


def elegir_candidatas(question: str, resultados: list[dict]) -> list[dict]:
    """Mejor de cada familia y, si faltan, las siguientes mejores hasta N_CANDIDATAS."""
    descendente = METRICA_SELECCION[question][1] == "max"
    ordenados = sorted(resultados, key=lambda r: r["valor"], reverse=descendente)
    elegidos, familias_vistas = [], set()
    for r in ordenados:
        if r["familia"] not in familias_vistas:
            elegidos.append(r)
            familias_vistas.add(r["familia"])
    for r in ordenados:
        if len(elegidos) >= N_CANDIDATAS:
            break
        if r not in elegidos:
            elegidos.append(r)
    elegidos = elegidos[:N_CANDIDATAS]
    return sorted(elegidos, key=lambda r: r["valor"], reverse=descendente)


def ejecutar_busqueda(question, model, n_estimators, datos, ctx) -> None:
    familias = FAMILIAS[question] if model in ("all", "kmeans") or question == "p3" else [model]
    resultados = []
    for familia in familias:
        for hp in rejilla(question, familia, n_estimators):
            resultados.append(correr_hija_busqueda(question, familia, hp, datos, ctx))

    clave = METRICA_SELECCION[question][0]
    candidatas = elegir_candidatas(question, resultados)
    print(f"\nCandidatas (por {clave}):")
    for r in candidatas:
        correr_final(question, r["familia"], r["hp"], datos, ctx, "candidate", val_valor=r["valor"])
    mlflow.log_metric(f"best_{clave}", candidatas[0]["valor"])
    mlflow.log_metric("n_combinaciones", len(resultados))


def ejecutar(question, mode, model, n_estimators, max_depth, n_clusters) -> None:
    question, mode, model = question.lower(), mode.lower(), model.lower()
    if question not in FAMILIAS:
        raise ValueError("question debe ser p1, p2 o p3")
    if mode not in ("baseline", "single", "search"):
        raise ValueError("mode debe ser baseline, single o search")
    if question != "p3" and model not in FAMILIAS[question] + ["all"]:
        raise ValueError(f"Para {question} model debe ser uno de {FAMILIAS[question] + ['all']}")

    configurar_mlflow()
    datos = cargar_datos(question)
    ctx = {"commit": obtener_commit_hash(), "data_hash": obtener_version_datos()}

    # Dentro de `mlflow run` este start_run retoma el run que ya creó el proyecto.
    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"{question}_{mode}")
        mlflow.set_tags({"equipo": "equipo08", "question": question, "run_role": "parent",
                         "git_commit": ctx["commit"]})
        mlflow.log_param("data_hash_version", ctx["data_hash"])
        if mode == "search":
            mlflow.set_tags({"stage_of_work": "search", "model_family": model})

        print(f"\n--- {question.upper()} | modo={mode} | datos={ctx['data_hash']} | commit={ctx['commit'][:7]} ---")
        if mode == "baseline":
            if question == "p3":
                correr_final(question, "kmeans", {"n_clusters": n_clusters}, datos, ctx, "baseline")
            else:
                correr_final(question, "dummy", {}, datos, ctx, "baseline")
        elif mode == "single":
            if model == "all":
                raise ValueError("En modo single indica un modelo concreto, por ejemplo -P model=rf")
            familia = "kmeans" if question == "p3" else model
            if question == "p3":
                hp = {"n_clusters": n_clusters}
            elif familia == "logreg":
                hp = {}
            else:
                hp = {"n_estimators": n_estimators, "max_depth": _profundidad(max_depth)}
            correr_final(question, familia, hp, datos, ctx, "candidate")
        else:
            ejecutar_busqueda(question, model, n_estimators, datos, ctx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento MLOps MiBici (P1, P2, P3)")
    parser.add_argument("--question", default="p1", choices=["p1", "p2", "p3"])
    parser.add_argument("--mode", default="search", choices=["baseline", "single", "search"])
    parser.add_argument("--model", default="all")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--n_clusters", type=int, default=2)
    a = parser.parse_args()
    ejecutar(a.question, a.mode, a.model, a.n_estimators, a.max_depth, a.n_clusters)
