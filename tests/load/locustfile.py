# Escenario de prueba de carga con Locust para la API de MiBici (pesos 3:2 entre P1 y P2).
import random
from locust import HttpUser, task, between


class MiBiciPredictionUser(HttpUser):
    wait_time = between(0.5, 2.0)
    #Tarea de predicción de demanda (peso 3); evalúa respuesta HTTP 200 en POST /predict/p1.
    @task(3)
    def predict_demand_p1(self):
        payload = {
            "origen_id": random.randint(1, 300),
            "hora": random.randint(0, 23),
            "dia_semana": random.randint(0, 6),
            "es_fin_semana": random.choice([0, 1]),
        }
        with self.client.post(
            "/predict/p1",
            json=payload,
            catch_response=True,
            name="POST /predict/p1",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"P1 fallo ({resp.status_code}): {resp.text}")
    # Tarea de riesgo de desbalance (peso 2); calcula balance_neto en POST /predict/p2.
    @task(2)
    def predict_imbalance_p2(self):
        entradas = round(random.uniform(0.0, 25.0), 1)
        salidas = round(random.uniform(0.0, 25.0), 1)
        balance_neto = round(entradas - salidas, 1)
        payload = {
            "estacion_id": random.randint(1, 300),
            "hora": random.randint(0, 23),
            "entradas": entradas,
            "salidas": salidas,
            "balance_neto": balance_neto,
            "balance_acumulado": round(random.uniform(-30.0, 30.0), 1),
        }
        with self.client.post(
            "/predict/p2",
            json=payload,
            catch_response=True,
            name="POST /predict/p2",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"P2 fallo ({resp.status_code}): {resp.text}")