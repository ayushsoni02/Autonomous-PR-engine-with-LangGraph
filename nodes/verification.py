# =============================================================================
#  nodes/verification.py  —  Docker Sandbox Test Runner (Non-LLM Node)
# =============================================================================
#
#  WHAT DOES THIS NODE DO?
#  ───────────────────────
#  This is a NON-LLM node — no AI calls, just deterministic logic:
#
#    1. Parse the Coder Agent's file_changes from state.patch (JSON string)
#    2. Clone the target repo into a temporary directory
#    3. Overwrite files with the Coder Agent's fixes
#    4. Write the test file
#    5. Run pytest inside a sandboxed Docker container
#    6. Capture exit code + stdout/stderr
#    7. Return test results (pass/fail) and error logs
#
#  SECURITY:
#    - Docker container runs with --network=none (no internet)
#    - Memory capped at 512MB, CPU capped at 1.0
#    - Runs as unprivileged user (nobody)
#    - Workspace mounted read-only
#    - 60-second timeout on the entire container
#
#  INPUTS (from state):
#    - repo_name       str
#    - issue_number    int
#    - patch           str   (JSON: {filepath: new_content})
#    - test_code       str
#    - retry_count     int
#
#  OUTPUTS (written to state):
#    - test_output     str
#    - test_passed     bool
#    - error_logs      str   (only if failed)
#    - retry_count     int   (incremented if failed)
#    - messages        list[dict]
#
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import settings
from logger import get_logger
from state import AgentState

log = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DOCKER_TIMEOUT_SECONDS = 120      # Max time for the entire Docker run
PYTEST_TIMEOUT_SECONDS = 30       # Per-test timeout inside pytest
CONTAINER_MEMORY_LIMIT = "512m"
CONTAINER_CPU_LIMIT = "1.0"


# ── Helper Functions ─────────────────────────────────────────────────────────

def _clone_repo(repo_name: str, target_dir: str) -> None:
    """
    Shallow-clone a GitHub repo into target_dir.

    Uses --depth=1 for speed — we only need the latest code, not history.
    The GITHUB_TOKEN is embedded in the URL for private repo access.
    """
    token = settings.github_token
    clone_url = f"https://{token}@github.com/{repo_name}.git"

    log.info("cloning repo", repo=repo_name, target=target_dir)

    result = subprocess.run(
        ["git", "clone", "--depth=1", clone_url, target_dir],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}):\n{result.stderr}"
        )

    log.info("repo cloned", repo=repo_name)


def _apply_file_changes(workspace: str, file_changes: dict[str, str]) -> None:
    """
    Overwrite files in the workspace with the Coder Agent's fixes.

    Creates parent directories if they don't exist.
    """
    for file_path, content in file_changes.items():
        full_path = os.path.join(workspace, file_path)

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        log.info("file applied", path=file_path, size=len(content))


def _write_test_file(
    workspace: str, issue_number: int, test_code: str
) -> str:
    """
    Write the pytest test file into the workspace.

    Returns the path relative to the workspace (for the pytest command).
    """
    test_file_path = f"tests/test_fix_issue_{issue_number}.py"
    full_path = os.path.join(workspace, test_file_path)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(test_code)

    log.info("test file written", path=test_file_path, size=len(test_code))
    return test_file_path


def _run_docker_tests(workspace: str, test_file_path: str) -> tuple[int, str]:
    """
    Run pytest inside a Docker container.

    Returns:
        (exit_code, combined_stdout_stderr)
    """
    image = settings.sandbox_image_name

    docker_cmd = [
        "docker", "run",
        "--rm",                                     # Remove container after run
        "--network=none",                           # No internet access
        f"--memory={CONTAINER_MEMORY_LIMIT}",       # Memory cap
        f"--cpus={CONTAINER_CPU_LIMIT}",            # CPU cap
        "--user=nobody",                            # Unprivileged user
        "-v", f"{workspace}:/app",                  # Mount workspace
        "-w", "/app",                               # Set working directory
        image,                                      # Sandbox image
        "pytest",                                   # Command
        test_file_path,                             # Test file
        "-v",                                       # Verbose output
        "--tb=short",                               # Short traceback format
        f"--timeout={PYTEST_TIMEOUT_SECONDS}",      # Per-test timeout
        "--no-header",                              # Clean output
    ]

    log.info(
        "running docker tests",
        image=image,
        test_file=test_file_path,
        cmd=" ".join(docker_cmd),
    )

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT_SECONDS,
        )
        output = result.stdout + "\n" + result.stderr
        return result.returncode, output.strip()

    except subprocess.TimeoutExpired:
        return 1, f"TIMEOUT: Docker container exceeded {DOCKER_TIMEOUT_SECONDS}s limit"

    except FileNotFoundError:
        return 1, (
            "ERROR: Docker is not installed or not in PATH.\n"
            "Install Docker and ensure 'docker' command is available."
        )


def _extract_error_logs(test_output: str) -> str:
    """
    Extract the most relevant error information from pytest output.

    Filters out noise and keeps: FAILED lines, assertion errors,
    tracebacks, and summary lines.
    """
    relevant_lines = []
    capture = False

    for line in test_output.split("\n"):
        line_lower = line.lower().strip()

        # Always capture these
        if any(keyword in line_lower for keyword in [
            "failed", "error", "assert", "traceback",
            "raise", "exception", "importerror", "modulenotfounderror",
            "syntaxerror", "typeerror", "nameerror", "attributeerror",
            "short test summary",
        ]):
            capture = True

        if capture:
            relevant_lines.append(line)

        # Stop capturing after blank lines following error blocks
        if capture and line.strip() == "" and len(relevant_lines) > 3:
            capture = False

    # If we didn't find anything specific, return the last 50 lines
    if not relevant_lines:
        lines = test_output.split("\n")
        relevant_lines = lines[-50:]

    return "\n".join(relevant_lines).strip()


# ── Node Function ────────────────────────────────────────────────────────────

def verification_node(state: AgentState) -> dict:
    """
    LangGraph node: Verification (non-LLM).

    Clones the repo, applies the Coder Agent's fixes, runs pytest in
    a Docker sandbox, and returns the results.
    """
    repo_name = state["repo_name"]
    issue_number = state.get("issue_number", 0)
    patch_json = state["patch"]
    test_code = state["test_code"]
    retry_count = state.get("retry_count", 0)

    log.info(
        "verification node started",
        repo=repo_name,
        issue=issue_number,
        retry_count=retry_count,
    )

    # ── Step 1: Parse file changes from JSON ─────────────────────────────
    try:
        file_changes: dict[str, str] = json.loads(patch_json)
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse patch JSON: {e}\nRaw patch: {patch_json[:500]}"
        log.error("patch parse failed", error=str(e))
        return {
            "test_output": error_msg,
            "test_passed": False,
            "error_logs": error_msg,
            "retry_count": retry_count + 1,
            "messages": [
                {
                    "agent": "verification",
                    "action": "Failed to parse patch JSON",
                    "error": str(e),
                }
            ],
        }

    log.info("patch parsed", files_to_change=list(file_changes.keys()))

    # ── Step 2: Create temp workspace and clone repo ─────────────────────
    workspace = tempfile.mkdtemp(prefix=f"pr-engine-issue-{issue_number}-")

    try:
        _clone_repo(repo_name, workspace)

        # ── Step 3: Apply file changes ───────────────────────────────────
        _apply_file_changes(workspace, file_changes)

        # ── Step 4: Write test file ──────────────────────────────────────
        test_file_path = _write_test_file(workspace, issue_number, test_code)

        # ── Step 5: Run tests in Docker ──────────────────────────────────
        exit_code, test_output = _run_docker_tests(workspace, test_file_path)

        test_passed = exit_code == 0

        log.info(
            "docker tests complete",
            exit_code=exit_code,
            passed=test_passed,
            output_length=len(test_output),
        )

        # ── Step 6: Build result ─────────────────────────────────────────
        if test_passed:
            return {
                "test_output": test_output,
                "test_passed": True,
                "error_logs": "",
                "retry_count": retry_count,
                "messages": [
                    {
                        "agent": "verification",
                        "action": "Tests PASSED",
                        "exit_code": exit_code,
                        "retry_count": retry_count,
                    }
                ],
            }
        else:
            error_logs = _extract_error_logs(test_output)
            return {
                "test_output": test_output,
                "test_passed": False,
                "error_logs": error_logs,
                "retry_count": retry_count + 1,
                "messages": [
                    {
                        "agent": "verification",
                        "action": f"Tests FAILED (retry {retry_count + 1})",
                        "exit_code": exit_code,
                        "retry_count": retry_count + 1,
                        "error_summary": error_logs[:300],
                    }
                ],
            }

    except Exception as e:
        error_msg = f"Verification node error: {type(e).__name__}: {e}"
        log.error("verification failed", error=error_msg)
        return {
            "test_output": error_msg,
            "test_passed": False,
            "error_logs": error_msg,
            "retry_count": retry_count + 1,
            "messages": [
                {
                    "agent": "verification",
                    "action": "Verification node crashed",
                    "error": error_msg,
                }
            ],
        }

    finally:
        # ── Step 7: Cleanup temp directory ───────────────────────────────
        try:
            shutil.rmtree(workspace, ignore_errors=True)
            log.info("workspace cleaned up", path=workspace)
        except Exception:
            log.warning("failed to cleanup workspace", path=workspace)
