FROM python:3.11-slim

# No generar .pyc y loguear sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias del sistema para OpenCV y similares
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Primero requirements para aprovechar la cache de Docker
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ahora copiamos el resto del código (incluye main.py, services/, best.pt, etc.)
COPY . .

# Puerto interno donde escuchará uvicorn
ENV PORT=8000

# Comando de arranque (ajusta main:app si tu archivo o variable se llama distinto)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
