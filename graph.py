# =============================================================================
#  graph.py  —  LangGraph StateGraph Wiring
# =============================================================================
#
#  WHAT IS THIS FILE?
#  ──────────────────
#  This is the heart of the agentic pipeline. It defines:
#    - Which nodes (agents/functions) exist
#    - What order they run in
#    - The conditional routing logic (retry loop on failed verification)
#
#  GRAPH STRUCTURE:
#
#    START → triage → research → coder → verify ─┬─ PASS → pr_agent → END
#                                                 │
#                                                 ├─ FAIL (retries left) → coder (loop)
#                                                 │
#                                                 └─ FAIL (max retries) → END
#
#  HOW TO USE:
#  ──────────────────
#  from graph import app
#  result = app.invoke({"issue_url": "https://github.com/owner/repo/issues/42"})
#
# =============================================================================

from __future__ import annotations

from langgraph.graph import END, StateGraph

from config import settings
from logger import get_logger
from state import AgentState

# ── Import all node functions ────────────────────────────────────────────────
from agents.triage import triage_agent
from agents.research import research_agent
from agents.coder import coder_agent
from nodes.verification import verification_node
from agents.pr_agent import pr_agent

log = get_logger(__name__)


# ── Conditional Router ───────────────────────────────────────────────────────

def route_after_verification(state: AgentState) -> str:
    """
    Decide what happens after the Verification Node runs.

    Returns the NAME of the next node to execute:
      - "pr_agent"  → tests passed, open a PR
      - "coder"     → tests failed, retry (loop back to Coder Agent)
      - END         → tests failed and max retries reached
    """
    test_passed = state.get("test_passed", False)
    retry_count = state.get("retry_count", 0)
    max_retries = settings.max_retry_count

    if test_passed:
        log.info("routing to pr_agent", reason="tests passed")
        return "pr_agent"

    if retry_count >= max_retries:
        log.warning(
            "routing to END",
            reason="max retries exceeded",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return END

    log.info(
        "routing back to coder",
        reason="tests failed, retrying",
        retry_count=retry_count,
        max_retries=max_retries,
    )
    return "coder"


# ── Build the Graph ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph that can be invoked with:
        result = app.invoke({"issue_url": "https://github.com/..."})
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────
    workflow.add_node("triage", triage_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("coder", coder_agent)
    workflow.add_node("verify", verification_node)
    workflow.add_node("pr_agent", pr_agent)

    # ── Set the entry point ──────────────────────────────────────────────
    workflow.set_entry_point("triage")

    # ── Linear edges ─────────────────────────────────────────────────────
    workflow.add_edge("triage", "research")      # triage → research (always)
    workflow.add_edge("research", "coder")       # research → coder (always)
    workflow.add_edge("coder", "verify")         # coder → verify (always)
    workflow.add_edge("pr_agent", END)           # pr_agent → END (always)

    # ── Conditional edge after verification ──────────────────────────────
    # This is the retry loop: if tests fail, go back to coder; if pass, go to pr_agent
    workflow.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "pr_agent": "pr_agent",    # tests passed
            "coder": "coder",          # tests failed, retry
            END: END,                  # max retries exceeded
        },
    )

    log.info("graph built", nodes=5, edges=5)
    return workflow


# ── Compiled App (module-level singleton) ────────────────────────────────────
# Import this directly: `from graph import app`
app = build_graph().compile()
