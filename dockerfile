# --- Base ---
FROM python:3.12-slim

# --- Variables de entorno ---
ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# --- Directorio de trabajo ---
WORKDIR /app

# --- Copiar requirements ---
COPY requirements.txt .

# --- Crear venv e instalar dependencias ---
RUN python -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip setuptools wheel \
    && $VENV_PATH/bin/pip install -r requirements.txt

# --- Copiar la app ---
COPY . .

# --- Ajustar PATH para usar venv ---
ENV PATH="$VENV_PATH/bin:$PATH"

# --- Exponer puerto Flask ---
EXPOSE 8002

# --- Comando por defecto ---
CMD ["python", "app.py"]
