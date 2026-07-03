FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Instalar libvulkan1 y limpiar la caché de apt en la misma capa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar e instalar requerimientos de Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Copiar el código de la aplicación
COPY app /app/app
COPY profiles /app/profiles

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${SERVER_PORT:-8000}"]