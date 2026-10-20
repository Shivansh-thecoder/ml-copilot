# Development image for the FastAPI backend.
# NOTE: dev-focused (uses --reload); a separate production-optimized
# multi-stage build can be added in a later phase.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp/requirements.txt /tmp/mcp-requirements.txt
RUN python -m venv /opt/mcp-venv \
    && /opt/mcp-venv/bin/pip install --no-cache-dir -r /tmp/mcp-requirements.txt
ENV MCP_SERVER_PYTHON=/opt/mcp-venv/bin/python

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "python -m alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
