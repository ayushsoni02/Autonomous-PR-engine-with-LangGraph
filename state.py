# =============================================================================
#  state.py  —  Shared Agent State (LangGraph TypedDict)
# =============================================================================
#
#  WHAT IS THIS FILE?
#  ──────────────────
#  This defines the single state object that flows through every node in the
#  LangGraph pipeline. Each agent reads what it needs and returns only the
#  fields it updated. LangGraph merges them back automatically.
#
#  WHY TypedDict (not Pydantic)?
#  ─────────────────────────────
#  LangGraph's StateGraph expects a TypedDict (or a class with __annotations__)
#  as the state schema. Using TypedDict keeps things simple, lightweight,
#  and fully compatible with LangGraph's internal state merging.
#
#  HOW TO USE:
#  ──────────────────
#  from state import AgentState
#
#  # In a LangGraph node function:
#  def my_node(state: AgentState) -> dict:
#      return {"plan": "do X, then Y"}   # return ONLY updated fields
#
# =============================================================================

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between every node in the PR Engine pipeline.

    Fields are grouped by which agent produces them.
    `total=False` means every field is optional — agents only return
    the fields they update, and LangGraph merges them into the full state.
    """

    # ── Input (set once by the API layer) ────────────────────────────────────
    issue_url: str                  # Full GitHub issue URL from the user
    issue_number: int               # Parsed from the URL (e.g. 42)
    repo_name: str                  # "owner/repo" format

    # ── Triage Agent outputs ─────────────────────────────────────────────────
    issue_title: str                # Issue title fetched from GitHub
    issue_body: str                 # Issue description / body text
    file_tree: list[str]            # Every file path in the repo
    relevant_files: list[str]       # 3–10 files selected by LLM as relevant

    # ── Research Agent outputs ───────────────────────────────────────────────
    file_contents: dict[str, str]   # {filepath: file_content} for relevant files
    dependency_map: str             # JSON string: LLM's analysis of dependencies

    # ── Coder Agent outputs ──────────────────────────────────────────────────
    plan: str                       # Step-by-step fix strategy
    patch: str                      # JSON string: {filepath: new_full_content}
    test_code: str                  # Complete pytest file content

    # ── Verification Node outputs ────────────────────────────────────────────
    test_output: str                # Raw stdout+stderr from Docker pytest run
    test_passed: bool               # True if pytest exited with code 0
    error_logs: str                 # Filtered failure output (for retry context)
    retry_count: int                # Incremented on each failed verification

    # ── PR Agent outputs ─────────────────────────────────────────────────────
    branch_name: str                # Git branch created for this fix
    pr_url: str                     # Final Pull Request URL

    # ── Audit trail ──────────────────────────────────────────────────────────
    # `operator.add` tells LangGraph to APPEND new messages instead of
    # overwriting. Every agent appends a dict like:
    #   {"agent": "triage", "action": "selected 5 relevant files", ...}
    messages: Annotated[list[dict], operator.add]
