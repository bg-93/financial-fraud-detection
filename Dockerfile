FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRAUD_ARTIFACT_DIR=/app/artifacts \
    PORT=8000

WORKDIR /app

# XGBoost requires the OpenMP runtime. Installing dependencies before source
# files keeps this expensive layer cached when application code changes.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install --no-cache-dir --requirement requirements-api.txt

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts \
    && chown app:app /app/artifacts

# The model artifacts are included when they exist in the build context. The
# same image can also be built before training and supplied artifacts later via
# a read-only volume mounted at /app/artifacts.
COPY --chown=app:app . ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=2)"]

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port \"${PORT}\""]
