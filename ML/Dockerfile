# --- Builder Stage ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optimize python/uv execution
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# Tell uv to use the system python (3.12) and not download its own
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies into .venv, using cache mounts to speed up builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# --- Final Production Stage ---
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy the application code and ML models
COPY . /app

EXPOSE 8000

# Run uvicorn directly from the virtual environment path
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
