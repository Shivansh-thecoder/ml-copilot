# Development image for the FastAPI backend.
# NOTE: dev-focused (uses --reload); a separate production-optimized
# multi-stage build can be added in a later phase.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
