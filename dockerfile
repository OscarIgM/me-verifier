
FROM python:3.12-slim

# --- 2. Variables de entorno ---
ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv

# --- 3. Crear directorio de la app ---
WORKDIR /app

COPY requirements.txt .

RUN python -m venv $VENV_PATH \
    && $VENV_PATH/bin/pip install --upgrade pip setuptools wheel \
    && $VENV_PATH/bin/pip install -r requirements.txt

COPY . .

ENV PATH="$VENV_PATH/bin:$PATH"

EXPOSE 8002

CMD ["uvicorn", "mcp_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
