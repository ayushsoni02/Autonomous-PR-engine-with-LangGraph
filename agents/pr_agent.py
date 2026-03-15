# =============================================================================
#  agents/pr_agent.py  —  PR Agent (Agent 04)
# =============================================================================
#
#  WHAT DOES THIS AGENT DO?
#  ────────────────────────
#  The PR Agent is the FINAL node in the pipeline (runs only when tests pass).
#  It:
#    1. Creates a feature branch (fix/issue-{N}-{slugified-title})
#    2. Commits all changed files + the test file to the branch
#    3. Uses the LLM to write a PR description
#    4. Opens a Pull Request on GitHub
#
#  INPUTS (from state):
#    - repo_name        str
#    - issue_number     int
#    - issue_title      str
#    - issue_body       str
#    - plan             str
#    - patch            str   (JSON: {filepath: new_content})
#    - test_code        str
#
#  OUTPUTS (written to state):
#    - branch_name      str
#    - pr_url           str
#    - messages         list[dict]
#
# =============================================================================

from __future__ import annotations

import json
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from state import AgentState
from tools.github_tools import (
    create_branch,
    commit_file_changes,
    open_pull_request,
)

log = get_logger(__name__)


# ── Structured Output Model ─────────────────────────────────────────────────

class PRDescription(BaseModel):
    """Structured output for the PR description."""

    title: str = Field(
        description=(
            "Concise PR title that summarizes the fix. "
            "Format: 'Fix #{issue_number}: {brief description}'"
        )
    )
    body: str = Field(
        description=(
            "Detailed PR description in Markdown. Include:\n"
            "- What the issue was\n"
            "- Root cause analysis\n"
            "- What was changed and why\n"
            "- How it was tested\n"
            "Reference the issue number with #N."
        )
    )


# ── System Prompt ────────────────────────────────────────────────────────────

PR_SYSTEM_PROMPT = """\
You are a senior software engineer writing a Pull Request description.

Write a clear, professional PR description that helps reviewers understand:
1. **What** was changed
2. **Why** it was changed (link to the issue)
3. **How** the fix works (brief technical explanation)
4. **Testing** — what tests were added

FORMAT RULES:
- Use Markdown formatting
- Start with a summary line
- Use ## headers for sections: Summary, Changes, Testing
- Reference the issue with "Fixes #N" so GitHub auto-closes it
- Keep it concise but informative
- List changed files with brief descriptions
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _slugify(text: str, max_length: int = 40) -> str:
    """
    Convert text to a URL-safe slug for branch names.

    "Handle null password in login" → "handle-null-password-in-login"
    """
    # Lowercase and replace non-alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    # Remove leading/trailing hyphens
    slug = slug.strip("-")
    # Truncate to max_length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug


# ── Node Function ────────────────────────────────────────────────────────────

def pr_agent(state: AgentState) -> dict:
    """
    LangGraph node: PR Agent.

    Creates a branch, commits the fix, generates a PR description via LLM,
    and opens a Pull Request on GitHub.
    """
    repo_name = state["repo_name"]
    issue_number = state.get("issue_number", 0)
    issue_title = state["issue_title"]
    issue_body = state["issue_body"]
    plan = state.get("plan", "")
    patch_json = state["patch"]
    test_code = state["test_code"]

    log.info("pr agent started", repo=repo_name, issue=issue_number)

    # ── Step 1: Parse file changes ───────────────────────────────────────
    file_changes: dict[str, str] = json.loads(patch_json)

    # Add the test file to the commit
    test_file_path = f"tests/test_fix_issue_{issue_number}.py"
    file_changes[test_file_path] = test_code

    log.info(
        "files to commit",
        count=len(file_changes),
        files=list(file_changes.keys()),
    )

    # ── Step 2: Create branch ────────────────────────────────────────────
    slug = _slugify(issue_title)
    branch_name = f"fix/issue-{issue_number}-{slug}"

    create_branch.invoke({
        "repo_full_name": repo_name,
        "branch_name": branch_name,
    })
    log.info("branch created", branch=branch_name)

    # ── Step 3: Commit all file changes ──────────────────────────────────
    commit_message = f"fix: resolve issue #{issue_number} — {issue_title}"

    commit_file_changes.invoke({
        "repo_full_name": repo_name,
        "branch_name": branch_name,
        "file_changes": file_changes,
        "commit_message": commit_message,
    })
    log.info("files committed", branch=branch_name)

    # ── Step 4: Generate PR description via LLM ─────────────────────────
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(PRDescription)

    changed_files_list = "\n".join(f"- `{f}`" for f in file_changes.keys())
    human_message = (
        f"## Issue #{issue_number}: {issue_title}\n\n"
        f"{issue_body}\n\n"
        f"---\n\n"
        f"## Fix Plan\n{plan}\n\n"
        f"---\n\n"
        f"## Changed Files\n{changed_files_list}\n\n"
        f"Write a PR description for this fix."
    )

    pr_desc: PRDescription = structured_llm.invoke([
        SystemMessage(content=PR_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])

    log.info("PR description generated", title=pr_desc.title)

    # ── Step 5: Open Pull Request ────────────────────────────────────────
    pr_url = open_pull_request.invoke({
        "repo_full_name": repo_name,
        "branch_name": branch_name,
        "title": pr_desc.title,
        "body": pr_desc.body,
    })

    log.info("pull request opened", pr_url=pr_url)

    # ── Return updated state fields ──────────────────────────────────────
    return {
        "branch_name": branch_name,
        "pr_url": pr_url,
        "messages": [
            {
                "agent": "pr_agent",
                "action": "Opened Pull Request",
                "branch": branch_name,
                "pr_url": pr_url,
                "pr_title": pr_desc.title,
                "files_committed": list(file_changes.keys()),
            }
        ],
    }
