import os
import pandas as pd
from fastapi import FastAPI, HTTPException
import mlflow.pyfunc

from mibici.api.schemas import PredictRequestP1, PredictResponseP1
from mibici.config import MLFLOW_TRACKING_URI

app = FastAPI(
    title="API de Inferencia MiBici - Equipo 08",
    description="Servicio de predicción de demanda de viajes en tiempo real utilizando el modelo Champion de MLflow.",
    version="1.0.0"
)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
mlflow.set_tracking_uri(tracking_uri)

MODEL_ALIAS_URI = "models:/mibici-equipo08-p1-demanda@champion"
model = None

@app.on_event("startup")
def load_model():
    """Carga el modelo registrado en MLflow al iniciar el servicio."""
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_ALIAS_URI)
        print(f"Modelo cargado exitosamente desde: {MODEL_ALIAS_URI}")
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo desde {MODEL_ALIAS_URI}: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "MiBici Demand Prediction API"}

@app.post("/predict/p1", response_model=PredictResponseP1)
def predict_p1(request: PredictRequestP1):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible o cargado.")
    
    try:
        # Convertir explícitamente a float64 para cumplir con la firma esperada por MLflow
        input_data = pd.DataFrame([{
            "origen_id": float(request.origen_id),
            "hora": float(request.hora),
            "dia_semana": float(request.dia_semana),
            "es_fin_semana": float(request.es_fin_semana)
        }], dtype="float64")
        
        prediction = model.predict(input_data)
        pred_value = float(prediction[0])
        
        return PredictResponseP1(
            origen_id=request.origen_id,
            prediccion_viajes=pred_value
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción del modelo: {str(e)}")