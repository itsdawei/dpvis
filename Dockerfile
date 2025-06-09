FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set working directory
WORKDIR /app

# Copy poetry files first for better Docker caching
COPY pyproject.toml poetry.lock* README.md /app/

# Now copy your actual code
COPY . /app

# Configure Poetry to install into global env
RUN poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi

# Accept demo name via build arg and set environment variable
ARG DEMO
ENV DEMO=${DEMO}

# Run Dash via gunicorn
CMD ["sh", "-c", "gunicorn demos.${DEMO}:server -b 0.0.0.0:8080"]
