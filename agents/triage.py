# =============================================================================
#  agents/triage.py  —  Triage Agent (Agent 01)
# =============================================================================
#
#  WHAT DOES THIS AGENT DO?
#  ────────────────────────
#  The Triage Agent is the FIRST node in the pipeline. It:
#
#    1. Fetches the GitHub issue (title, body, labels)
#    2. Fetches the full file tree of the repository
#    3. Sends both to the LLM, which selects 3–10 files most relevant
#       to fixing the issue
#
#  INPUTS (from state):
#    - issue_url         (set by the API layer)
#
#  OUTPUTS (written to state):
#    - issue_title       str
#    - issue_body        str
#    - issue_number      int
#    - repo_name         str
#    - file_tree         list[str]
#    - relevant_files    list[str]    ← the key output
#    - messages          list[dict]   ← audit log entry
#
# =============================================================================

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from state import AgentState
from tools.github_tools import get_issue_details, get_file_tree

log = get_logger(__name__)


# ── Structured Output Model ─────────────────────────────────────────────────
# The LLM must return JSON matching this schema. LangChain's
# `with_structured_output` enforces this via Anthropic's tool-use mode.

class TriageOutput(BaseModel):
    """Structured output from the Triage Agent's LLM call."""

    reasoning: str = Field(
        description=(
            "Brief explanation of WHY these files are relevant to the issue. "
            "Mention specific clues from the issue title/body that led you "
            "to each file."
        )
    )
    relevant_files: list[str] = Field(
        description=(
            "List of 3–10 file paths from the repository that are most "
            "relevant to fixing the issue. Include files that contain the "
            "bug, files that test the buggy code, and files that would need "
            "to be modified for the fix."
        ),
        min_length=1,
        max_length=15,
    )


# ── System Prompt ────────────────────────────────────────────────────────────

TRIAGE_SYSTEM_PROMPT = """\
You are a senior software engineer performing bug triage on a GitHub repository.

YOUR TASK:
Given a GitHub issue and the complete file tree of the repository, identify
the 3–10 files that are MOST RELEVANT to understanding and fixing this issue.

SELECTION CRITERIA:
1. Files that likely CONTAIN the bug (based on issue description keywords,
   stack traces, function/class names mentioned)
2. Files that TEST the buggy functionality (test files, spec files)
3. Files that IMPORT or DEPEND ON the buggy code (e.g., if a utility
   function is broken, include files that call it)
4. Configuration files if the issue is config-related

EXCLUSION CRITERIA:
- Do NOT include unrelated files (e.g., docs, CI configs, README) unless
  the issue specifically mentions them
- Do NOT include lock files, generated files, or binary files
- Prefer fewer, more relevant files over many loosely related ones

IMPORTANT:
- Return EXACT file paths as they appear in the file tree
- If the repository is small (< 20 files), you may include most of them
- Always explain your reasoning
"""


# ── Node Function ────────────────────────────────────────────────────────────

def triage_agent(state: AgentState) -> dict:
    """
    LangGraph node: Triage Agent.

    Reads the issue, scans the file tree, and uses the LLM to select
    the most relevant files for further analysis.
    """
    issue_url = state["issue_url"]
    log.info("triage agent started", issue_url=issue_url)

    # ── Step 1: Fetch issue details via GitHub tool ──────────────────────
    issue_details = get_issue_details.invoke({"issue_url": issue_url})

    issue_title = issue_details["title"]
    issue_body = issue_details["body"]
    issue_number = issue_details["issue_number"]
    repo_name = issue_details["repo_name"]

    log.info(
        "issue details fetched",
        title=issue_title,
        issue_number=issue_number,
        repo=repo_name,
    )

    # ── Step 2: Fetch the full file tree ─────────────────────────────────
    file_tree = get_file_tree.invoke({"repo_full_name": repo_name})
    log.info("file tree fetched", total_files=len(file_tree))

    # ── Step 3: Ask LLM to select relevant files ────────────────────────
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(TriageOutput)

    # Build the human message with issue + file tree
    file_tree_str = "\n".join(file_tree)
    human_message = (
        f"## GitHub Issue #{issue_number}: {issue_title}\n\n"
        f"{issue_body}\n\n"
        f"---\n\n"
        f"## Repository File Tree ({len(file_tree)} files)\n\n"
        f"```\n{file_tree_str}\n```\n\n"
        f"Select the 3–10 most relevant files for fixing this issue."
    )

    result: TriageOutput = structured_llm.invoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])

    relevant_files = result.relevant_files
    log.info(
        "triage complete",
        selected_files=len(relevant_files),
        reasoning=result.reasoning[:200],
    )

    # ── Return updated state fields ──────────────────────────────────────
    return {
        "issue_title": issue_title,
        "issue_body": issue_body,
        "issue_number": issue_number,
        "repo_name": repo_name,
        "file_tree": file_tree,
        "relevant_files": relevant_files,
        "messages": [
            {
                "agent": "triage",
                "action": f"Selected {len(relevant_files)} relevant files",
                "reasoning": result.reasoning,
                "files": relevant_files,
            }
        ],
    }
