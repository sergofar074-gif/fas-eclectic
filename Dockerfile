# FAS Eclectic Audio Enhancer — production image
# Работает на Hugging Face Spaces (порт 7860), Render, Railway, Fly.io, VPS.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    FAS_JOBS_DIR=/tmp/jobs \
    FAS_SEPARATOR=auto

# libsndfile1 — аудиокодеки WAV; libgomp1 — рантайм onnxruntime (Silero VAD)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /tmp/jobs && rm -rf jobs/* __pycache__ pipeline/__pycache__

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=6s --start-period=25s --retries=3 \
  CMD python3 -c "import urllib.request, os; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','7860') + '/api/health', timeout=5)"

CMD ["python3", "app.py"]
