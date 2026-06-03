# ── Dynamic DB Studio — production image ─────────────────────────────────────
FROM python:3.13-slim

# curl is used by the container HEALTHCHECK; build-essential covers any deps
# that ship without a wheel for this platform.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so they cache across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY . .

# Per-user SQLite DBs + ChromaDB live here; mounted as a volume in compose.
RUN mkdir -p data
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app_db.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
