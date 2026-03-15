# =============================================================================
#  api.py  —  FastAPI Server
# =============================================================================
#
#  WHAT IS THIS FILE?
#  ──────────────────
#  This is the entry point for the PR Engine. It exposes two endpoints:
#
#    POST /run     — Takes a GitHub issue URL, runs the full agentic pipeline,
#                    and returns the PR URL (or error details).
#
#    GET /health   — Returns service health + configuration info.
#
#  HOW TO RUN:
#  ──────────────────
#    uvicorn api:app --reload --port 8000
#
#  HOW TO USE:
#  ──────────────────
#    curl -X POST http://localhost:8000/run \
#         -H "Content-Type: application/json" \
#         -d '{"issue_url": "https://github.com/owner/repo/issues/42"}'
#
# =============================================================================

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import settings
from logger import get_logger, setup_logging

# Initialize structured logging on module load
setup_logging(settings.log_level)

log = get_logger(__name__)


# ── Pydantic Request / Response Models ───────────────────────────────────────

class RunRequest(BaseModel):
    """Request body for POST /run."""
    issue_url: str = Field(
        description="Full GitHub issue URL, e.g. https://github.com/owner/repo/issues/42"
    )

    @field_validator("issue_url")
    @classmethod
    def validate_issue_url(cls, v: str) -> str:
        """Ensure the URL looks like a valid GitHub issue."""
        import re
        pattern = r"https?://github\.com/[^/]+/[^/]+/issues/\d+"
        if not re.match(pattern, v):
            raise ValueError(
                f"Invalid GitHub issue URL: {v}\n"
                f"Expected format: https://github.com/owner/repo/issues/123"
            )
        return v


class RunResponse(BaseModel):
    """Response body for POST /run."""
    status: str = Field(description="'success' or 'failed'")
    pr_url: Optional[str] = Field(default=None, description="Pull Request URL (if successful)")
    issue_url: str = Field(description="The input issue URL")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    duration_seconds: float = Field(description="Total wall-clock time in seconds")


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = "ok"
    timestamp: str
    model: str
    max_retries: int
    log_level: str
    version: str = "1.0.0"


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="PR Engine",
    description=(
        "Autonomous multi-agent system that takes a GitHub Issue URL and "
        "generates a Pull Request with a fix — fully automated."
    ),
    version="1.0.0",
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return service health and configuration summary."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=settings.model_name,
        max_retries=settings.max_retry_count,
        log_level=settings.log_level,
    )


@app.post("/run", response_model=RunResponse)
async def run_pipeline(request: RunRequest):
    """
    Run the full PR Engine pipeline for a given GitHub issue.

    1. Triage Agent identifies relevant files
    2. Research Agent analyzes dependencies
    3. Coder Agent generates a fix + tests
    4. Verification Node runs tests in Docker sandbox
    5. If tests pass → PR Agent opens a Pull Request
    6. If tests fail → Coder Agent retries (up to max_retries)
    """
    start_time = time.time()
    issue_url = request.issue_url

    log.info("pipeline started", issue_url=issue_url)

    try:
        # Import the compiled graph (lazy to avoid circular imports)
        from graph import app as graph_app

        # Run the full pipeline
        result = graph_app.invoke({"issue_url": issue_url})

        duration = round(time.time() - start_time, 1)
        pr_url = result.get("pr_url")
        retry_count = result.get("retry_count", 0)
        test_passed = result.get("test_passed", False)

        if pr_url:
            log.info(
                "pipeline succeeded",
                pr_url=pr_url,
                duration=duration,
                retries=retry_count,
            )
            return RunResponse(
                status="success",
                pr_url=pr_url,
                issue_url=issue_url,
                retry_count=retry_count,
                duration_seconds=duration,
            )
        else:
            # Pipeline completed but no PR was created (max retries exceeded)
            error_msg = (
                f"Max retries exceeded ({retry_count}). "
                f"Tests did not pass after {retry_count} attempt(s)."
            )
            log.warning(
                "pipeline failed — max retries",
                duration=duration,
                retries=retry_count,
            )
            return RunResponse(
                status="failed",
                pr_url=None,
                issue_url=issue_url,
                error=error_msg,
                retry_count=retry_count,
                duration_seconds=duration,
            )

    except Exception as e:
        duration = round(time.time() - start_time, 1)
        error_msg = f"{type(e).__name__}: {e}"
        log.error("pipeline crashed", error=error_msg, duration=duration)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "error": error_msg,
                "issue_url": issue_url,
                "duration_seconds": duration,
            },
        )


# ── Startup Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    log.info(
        "PR Engine started",
        model=settings.model_name,
        max_retries=settings.max_retry_count,
        port=settings.port,
    )
