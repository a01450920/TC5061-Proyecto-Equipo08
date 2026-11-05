FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY mibici ./mibici
COPY pyproject.toml .
COPY MLproject .
COPY .env.example .

EXPOSE 8000

CMD ["uvicorn", "mibici.api.app:app", "--host", "0.0.0.0", "--port", "8000"]