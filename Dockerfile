# =============================================================================
#  docker/Dockerfile.sandbox  —  Test Execution Sandbox
# =============================================================================
#
#  WHAT IS THIS?
#  ─────────────
#  This is the Docker image used to run the LLM-generated tests safely.
#  Every time the Verification Node needs to run tests, it spins up a fresh
#  container from this image, runs pytest, and destroys the container.
#
#  HOW TO BUILD THIS IMAGE (you must do this once):
#  ─────────────────────────────────────────────────
#  From the pr-engine/ directory, run:
#      docker build -f docker/Dockerfile.sandbox -t pr-engine-sandbox:latest .
#
#  WHAT'S INSTALLED:
#  ─────────────────
#  - Python 3.11 slim base
#  - pytest + common testing libraries
#  - Common Python packages that most repos use
#
#  SECURITY:
#  ─────────
#  - --network=none: No internet access during tests
#  - --user=nobody: Runs as unprivileged user
#  - The repo is mounted READ-ONLY (-v /path:/app:ro)
#  - Memory and CPU are limited by the docker run command
#
# =============================================================================

FROM python:3.11-slim

# Install system dependencies that Python packages often need
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install common Python testing and utility packages
# These cover most repos without needing custom per-repo setup
RUN pip install --no-cache-dir \
    pytest==8.3.3 \
    pytest-timeout==2.3.1 \
    pytest-mock==3.14.0 \
    pytest-cov==5.0.0 \
    requests==2.32.3 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    fastapi==0.115.4 \
    sqlalchemy==2.0.36 \
    boto3==1.35.0 \
    redis==5.1.1 \
    celery==5.4.0

# Default command — this gets overridden by docker run arguments
CMD ["pytest", "--help"]