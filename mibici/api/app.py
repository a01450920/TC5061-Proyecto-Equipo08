import os
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc

from mibici.api.schemas import PredictRequestP1, PredictResponseP1, PredictRequestP2, PredictResponseP2
from mibici.config import MLFLOW_TRACKING_URI

app = FastAPI(
    title="API de Inferencia MiBici - Equipo 08",
    description="Servicio de predicción para los modelos P1 (demanda) y P2 (riesgo de desbalance) utilizando modelos Champion de MLflow.",
    version="1.0.0"
)

#Reemplazar
#tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
#mlflow.set_tracking_uri(tracking_uri)

#Por esto:
db_path = os.path.abspath("mlflow.db")
default_sqlite_uri = f"sqlite:///{db_path}"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", default_sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)

P1_MODEL_ALIAS_URI = "models:/mibici-equipo08-p1-demanda@champion"
P2_MODEL_ALIAS_URI = "models:/mibici-equipo08-p2-riesgo@champion"
model_p1 = None
model_p2 = None

@app.on_event("startup")
def load_model():
    """Carga los modelos registrados en MLflow al iniciar el servicio."""
    global model_p1, model_p2

    try:
        model_p1 = mlflow.pyfunc.load_model(P1_MODEL_ALIAS_URI)
        print(f"Modelo P1 cargado exitosamente desde: {P1_MODEL_ALIAS_URI}")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo P1 desde {P1_MODEL_ALIAS_URI}: {e}")

    try:
        model_p2 = mlflow.pyfunc.load_model(P2_MODEL_ALIAS_URI)
        print(f"Modelo P2 cargado exitosamente desde: {P2_MODEL_ALIAS_URI}")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo P2 desde {P2_MODEL_ALIAS_URI}: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "MiBici Demand Prediction API"}


@app.post("/predict/p1", response_model=PredictResponseP1)
def predict_p1(request: PredictRequestP1):
    if model_p1 is None:
        raise HTTPException(status_code=503, detail="El modelo P1 no está disponible o cargado.")
    
    try:
        # Convertir explícitamente a float64 para cumplir con la firma esperada por MLflow
        input_data = pd.DataFrame([{
            "origen_id": float(request.origen_id),
            "hora": float(request.hora),
            "dia_semana": float(request.dia_semana),
            "es_fin_semana": float(request.es_fin_semana)
        }], dtype="float64")
        
        prediction = model_p1.predict(input_data)
        pred_value = float(prediction[0])
        
        return PredictResponseP1(origen_id=request.origen_id, prediccion_viajes=pred_value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción del modelo P1: {str(e)}")


@app.post("/predict/p2", response_model=PredictResponseP2)
def predict_p2(request: PredictRequestP2):

    if model_p2 is None:
        raise HTTPException(status_code=503, detail="El modelo P2 no está disponible o cargado.")

    try:
        # Convertir explícitamente a float64 para cumplir con la firma esperada por MLflow
        input_data = pd.DataFrame([{
            "estacion_id": float(request.estacion_id),
            "hora": float(request.hora),
            "entradas": float(request.entradas),
            "salidas": float(request.salidas),
            "balance_neto": float(request.balance_neto),
            "balance_acumulado": float(request.balance_acumulado)
        }], dtype="float64")

        prediction = model_p2.predict(input_data)
        pred_value = int(prediction[0])

        return PredictResponseP2(estacion_id=request.estacion_id, riesgo_desbalance=pred_value)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción del modelo P2: {str(e)}")