from pydantic import BaseModel, Field

class PredictRequestP1(BaseModel):
    origen_id: int = Field(..., description="ID de la estación de origen", example=10)
    hora: int = Field(..., ge=0, le=23, description="Hora del día (0-23)", example=8)
    dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana (0=Lunes, 6=Domingo)", example=1)
    es_fin_semana: int = Field(..., ge=0, le=1, description="Indica si es fin de semana (1) o no (0)", example=0)

class PredictResponseP1(BaseModel):
    origen_id: int
    prediccion_viajes: float
    modelo_version: str = "champion"
    
class PredictRequestP2(BaseModel):
    estacion_id: int = Field(..., description="ID de la estación", example=10)
    hora: int = Field(..., ge=0, le=23, description="Hora del día (0-23)", example=8)
    entradas: float = Field(..., description="Cantidad de bicicletas recibidas en la estación durante la hora", example=15.0)
    salidas: float = Field(..., description="Cantidad de bicicletas retiradas desde la estación durante la hora", example=12.0)
    balance_neto: float = Field(..., description="Diferencia entre entradas y salidas de la estación", example=3.0)
    balance_acumulado: float = Field(..., description="Balance acumulado de bicicletas en la estación para el día actual", example=25.0)

class PredictResponseP2(BaseModel):
    estacion_id: int
    riesgo_desbalance: int = Field(..., description="Predicción de riesgo para la siguiente hora (0=sin riesgo, 1=riesgo)", example=1)
    modelo_version: str = "champion"