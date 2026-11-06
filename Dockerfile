#FROM python:3.13-slim
FROM python:3.14.4-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY mibici ./mibici
COPY pyproject.toml .
COPY MLproject .
COPY .env.example .
COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns

EXPOSE 8000

CMD ["uvicorn", "mibici.api.app:app", "--host", "0.0.0.0", "--port", "8000"]