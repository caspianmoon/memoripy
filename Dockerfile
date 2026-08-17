FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 memoripy \
    && mkdir -p /data /config \
    && chown -R memoripy:memoripy /data /config

USER memoripy
EXPOSE 8080

CMD ["memoripy", "gateway", "/data", "/config/registry.json", "--host", "0.0.0.0", "--port", "8080"]
