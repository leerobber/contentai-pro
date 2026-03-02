FROM python:3.12-slim
# Pin to a specific digest in production for reproducibility, e.g.:
# FROM python:3.12-slim@sha256:<digest>

WORKDIR /app

# Install production dependencies only (copy .env.example for reference, not .env)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root user for security
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Use uvicorn directly; set DEBUG=false in production to disable reload
CMD ["sh", "-c", "uvicorn contentai_pro.main:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --log-level ${LOG_LEVEL:-info}"]
