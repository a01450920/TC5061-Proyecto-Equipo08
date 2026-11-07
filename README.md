<!--
GUÍA DE USO (borrar este bloque al pegar en el README.md)

1. Reemplaza la sección "## 7. Pruebas unitarias con PyTest" completa por la sección 7 de abajo.
2. En la sección "## 8. Chequeos de seguridad":
   - Añade al final el subapartado "Decisión sobre las vulnerabilidades" (abajo).
   - Corrige la frase "53 vulnerabilidades conocidas detectadas en 4 dependencias" por:
     "53 filas en el reporte, que corresponden a 39 identificadores únicos en 4 dependencias".
   - Añade la nota "Historial de Git" (abajo).
3. Las secciones 9 (DeepChecks) se agrega cuando la prueba de DeepChecks quede corregida.
4. Correcciones menores al resto del README: ver la lista al final de este archivo.
-->

## 7. Pruebas automatizadas y chequeos de calidad

<<<<<<< Updated upstream

## Flujo de trabajo Git

El equipo seguirá GitHub Flow.

### Ramas

- main: rama protegida
- feature/ingestion
- feature/features
- feature/mlflow
- feature/docs

### Convención de commits

Se utilizará Conventional Commits:

- feat: nueva funcionalidad
- fix: corrección de errores
- docs: documentación
- refactor: reorganización de código
- test: pruebas
- chore: configuración o mantenimiento

### Pull Requests

Todo cambio debe realizarse en una rama de trabajo y fusionarse mediante Pull Request hacia `main`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
=======
### Entorno de pruebas

Las herramientas de prueba y seguridad se instalan aparte del entorno de entrenamiento, para que
`mlflow run` no instale paquetes que no necesita.

```bash
python -m venv .venv
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Linux / macOS / WSL:   source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` incluye `requirements.txt` más PyTest, DeepChecks, Bandit y pip-audit
(y `setuptools` fijado, porque DeepChecks 0.19.1 requiere `pkg_resources`).

### Ejecución con un solo comando

```bash
python -m pytest tests -v
```

Se usa `python -m pytest` para que la raíz del proyecto quede en la ruta de importación y
las pruebas puedan importar el paquete `mibici`.

### Qué cubre cada archivo

| Archivo | Qué prueba |
|---|---|
| `tests/test_data.py` | Feature engineering de P1, P2 y P3 (caso normal y caso vacío) y reproducibilidad del entrenamiento (misma semilla, mismas predicciones) |
| `tests/test_ingesta.py` | Limpieza de viajes (cancelados, muy cortos o largos, edad imposible), límites de 1 y 240 minutos, lectura en latin-1 y hash SHA256 |
| `tests/test_modelado.py` | Conversión de `max_depth=-1`, rejilla de hiperparámetros, elección de candidatas, particiones temporales sin traslape, versión de datos y nombres del Model Registry |
| `tests/test_data_validation.py` | Validación de integridad y drift del dataset con DeepChecks |

### Evidencia

Reporte de ejecución: `reports/testing/pytest_report.txt`.
Capturas: `reports/figures/pytest_execution.png` y `reports/figures/pytest_ingesta_modelado.png`.

---

<!-- Añadir al final de la sección 8 -->

### Decisión sobre las vulnerabilidades de dependencias

`pip-audit` reportó 53 filas (39 identificadores únicos, algunos repetidos en el listado).
El reporte indica la versión que corrige cada una. El detalle de cada identificador se consulta con
`pip-audit -r requirements.txt --desc`.

| Dependencia | Versión actual | Identificadores únicos | Corregida en | Decisión propuesta |
|---|---|---|---|---|
| GitPython | 3.1.45 | 34 | 3.2.1 | Actualizar. Solo se usa para leer el hash del commit; validar con `pytest` |
| MLflow | 3.16.1 | 3 | 3.17.2 | No actualizar durante esta entrega: el servidor de tracking es local (127.0.0.1) y actualizar puede requerir migrar `mlflow.db`. Riesgo aceptado y documentado; actualizar en la siguiente etapa |
| PyArrow | 21.0.0 | 1 | 23.0.1 | Actualizar y validar la lectura de `viajes_clean.parquet` |
| Requests | 2.32.5 | 1 | 2.33.0 | Actualizar. Solo descarga los CSV de MiBici |

*(Ajustar la columna "Decisión" a lo que el equipo acuerde.)*

### Hallazgo de Bandit

El único hallazgo (B110, severidad baja: `except Exception: pass` en `obtener_version_datos`)
se corrigió: ahora solo captura errores de lectura o formato del manifiesto y avisa por consola.
Tiene pruebas en `tests/test_modelado.py`.

### Historial de Git

El archivo `mlflow.db` estuvo versionado en commits anteriores y después se retiró y se agregó al
`.gitignore`. Contiene metadatos de corridas (sin credenciales). No se reescribió el historial
para no afectar las ramas del equipo; el árbol actual del repositorio no lo contiene.

---

## 9. Validación de datos con DeepChecks

`tests/test_data_validation.py` valida la tabla de demanda de P1 (estación x fecha x hora), que no
contiene datos personales. Usa `data/processed/viajes_clean.parquet` y, si no existe, viajes
sintéticos con el mismo esquema. Entrenamiento (enero-abril) y prueba (mayo) se separan con las
fechas de `mibici/config.py`, igual que en el modelado.

| Tipo | Qué revisa | Resultado |
|---|---|---|
| Reglas de calidad (pandas) | Valores faltantes, filas duplicadas en la llave (estación, fecha, hora), tipos numéricos, rangos (hora 0-23, día 0-6, conteo >= 1) y separación temporal entre entrenamiento y prueba | **La prueba falla** si no se cumplen |
| DeepChecks `data_integrity` | Reporte de integridad del entrenamiento | Se genera el HTML y se avisa si hay condiciones no superadas |
| DeepChecks `train_test_validation` | Drift entre entrenamiento y prueba | Se genera el HTML y se avisa si hay condiciones no superadas |

Los hallazgos de DeepChecks **se reportan sin hacer fallar la prueba**: sus condiciones por defecto
(por ejemplo, 0 % de duplicados) no aplican a esta tabla, porque al quitar la fecha las combinaciones
de variables se repiten entre días por construcción, y entre enero-abril y mayo es esperable un drift
estacional. Las reglas que sí son errores de calidad se verifican de forma explícita.

Los reportes se guardan en `reports/testing/generated/` (ignorada por Git). Para actualizar la
evidencia versionada (`reports/testing/deepchecks_data_integrity.html` y
`reports/testing/deepchecks_train_test_drift.html`):

```powershell
$env:DEEPCHECKS_GUARDAR_EVIDENCIA = "1"
python -m pytest tests/test_data_validation.py -q
Remove-Item Env:DEEPCHECKS_GUARDAR_EVIDENCIA
```

**Limitación conocida:** DeepChecks 0.19.1 usa `pkg_resources` (de ahí `setuptools==69.5.1`) y
requiere parches de compatibilidad con NumPy 2 y scikit-learn, que están en el propio archivo de
pruebas. Con scikit-learn 1.8 o superior podría fallar y habría que actualizar DeepChecks.

---

<!--
CORRECCIONES MENORES AL RESTO DEL README

- "Instalación": usa `source mlops-env/bin/activate` (solo Linux). Reemplazar por los pasos de la sección 7.
- "Ejecución de experimentos": `python -m mibici.modeling.train` ya no ejecuta P1, P2 y P3 juntas.
  Reemplazar por: "Ver la sección 'Reproducción con MLflow Project'".
- Árbol del proyecto: quitar `mlruns/` y la línea de `mlflow.db`; en `tests/` listar los 4 archivos
  y añadir `requirements-dev.txt`.
- .gitignore: tiene entradas repetidas (`mlruns/`, `mlartifacts/`, `mlflow.db`, `.env`), y la regla
  `.env.*` también ignora `.env.example`. Dejar una sola vez cada regla y añadir `!.env.example`.
-->
>>>>>>> Stashed changes
