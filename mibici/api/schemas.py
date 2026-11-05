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