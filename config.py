# =============================================================================
#  config.py  —  Centralized Configuration
# =============================================================================
#
#  WHAT IS THIS FILE?
#  ──────────────────
#  This loads all environment variables from your .env file and exposes them
#  as a typed Python object. Every other file imports from here — no one
#  reads os.environ directly. This makes it easy to change settings in one
#  place and catch missing variables early (on startup, not mid-run).
#
#  HOW TO USE:
#  ──────────────────
#  from config import settings
#  print(settings.github_token)
#  print(settings.max_retry_count)
#
# =============================================================================

import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings  # pip install pydantic-settings


class Settings(BaseSettings):
    """
    All configuration values for the PR Engine.
    Values are loaded from your .env file automatically.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(
        description="Your Anthropic API key. Get it at console.anthropic.com"
    )

    # ── GitHub ───────────────────────────────────────────────────────────────
    github_token: str = Field(
        description="GitHub Personal Access Token with repo + PR permissions"
    )

    # ── Docker Sandbox ───────────────────────────────────────────────────────
    sandbox_image_name: str = Field(
        default="pr-engine-sandbox:latest",
        description="Docker image used for running tests. Built in Phase 4."
    )

    # ── Behaviour ────────────────────────────────────────────────────────────
    max_retry_count: int = Field(
        default=3,
        description="Max times Coder Agent retries before the graph gives up"
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    port: int = Field(
        default=8000,
        description="Port for the FastAPI server"
    )

    # ── LLM Model ────────────────────────────────────────────────────────────
    model_name: str = Field(
        default="claude-opus-4-5",
        description=(
            "Anthropic model to use for all agents.\n"
            "Options: claude-opus-4-5 (best), claude-sonnet-4-5 (faster/cheaper)"
        )
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        # Allow field names starting with "model_" (Pydantic reserves this
        # namespace by default for BaseModel internals).
        "protected_namespaces": (),
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.

    @lru_cache means this function only reads the .env file ONCE,
    no matter how many times it's called across the app.

    Usage:
        from config import settings
        print(settings.github_token)
    """
    return Settings()


# ── Module-level singleton ─────────────────────────────────────────────────
# Import this directly: `from config import settings`
settings = get_settings()