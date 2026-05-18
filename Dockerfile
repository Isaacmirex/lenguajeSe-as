# Dockerfile pensado para Hugging Face Spaces (SDK: Docker)
# Sirve la API FastAPI en el puerto 7860 (default de HF Spaces).

FROM python:3.11-slim

# Dependencias del sistema para OpenCV / Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      && rm -rf /var/lib/apt/lists/*

# HF Spaces requiere ejecutar como usuario no-root
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

# Instalar deps Python (cache layer)
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copiar el codigo y los pesos del modelo
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
