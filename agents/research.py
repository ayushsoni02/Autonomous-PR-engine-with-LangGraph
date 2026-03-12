# =============================================================================
#  agents/research.py  —  Research Agent (Agent 02)
# =============================================================================
#
#  WHAT DOES THIS AGENT DO?
#  ────────────────────────
#  The Research Agent is the SECOND node in the pipeline. It:
#
#    1. Reads the content of every file selected by the Triage Agent
#    2. Sends all file contents + the issue to the LLM
#    3. The LLM produces a dependency map: which functions/classes call
#       what, where the bug likely lives, and how the code connects
#
#  INPUTS (from state):
#    - repo_name         str
#    - issue_title       str
#    - issue_body        str
#    - relevant_files    list[str]
#
#  OUTPUTS (written to state):
#    - file_contents     dict[str, str]  — {path: content}
#    - dependency_map    str             — JSON analysis from LLM
#    - messages          list[dict]      — audit log entry
#
# =============================================================================

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from state import AgentState
from tools.github_tools import read_file_content

log = get_logger(__name__)


# ── Structured Output Model ─────────────────────────────────────────────────

class FileDependency(BaseModel):
    """Describes one file's role and relationships."""

    file_path: str = Field(description="Path to this file in the repository")
    purpose: str = Field(
        description="One-sentence summary of what this file does"
    )
    key_components: list[str] = Field(
        description=(
            "Important classes, functions, or variables defined in this file"
        )
    )
    depends_on: list[str] = Field(
        description=(
            "File paths this file imports from or depends on "
            "(only include files within the relevant set)"
        )
    )
    depended_by: list[str] = Field(
        description=(
            "File paths that import from or depend on this file "
            "(only include files within the relevant set)"
        )
    )


class ResearchOutput(BaseModel):
    """Structured output from the Research Agent's LLM call."""

    bug_location: str = Field(
        description=(
            "The specific file(s) and function(s)/line(s) where "
            "the bug most likely lives, with reasoning."
        )
    )
    root_cause_analysis: str = Field(
        description=(
            "Explanation of what is causing the issue, based on code analysis. "
            "Reference specific code patterns, variable names, or logic errors."
        )
    )
    dependency_analysis: list[FileDependency] = Field(
        description=(
            "Dependency analysis for each relevant file — what it does, "
            "what it exports, and how it relates to other files."
        )
    )
    suggested_approach: str = Field(
        description=(
            "High-level suggestion for how to fix the bug, mentioning "
            "which files need changes and what kind of changes."
        )
    )


# ── System Prompt ────────────────────────────────────────────────────────────

RESEARCH_SYSTEM_PROMPT = """\
You are a senior software engineer performing deep code analysis.

YOUR TASK:
Given a GitHub issue and the source code of relevant files, produce a
comprehensive analysis including:

1. **Bug Location**: Pinpoint exactly WHERE the bug is (file, function, line).
2. **Root Cause**: Explain WHY the bug happens (logic error, missing check, etc.).
3. **Dependency Map**: For each file, describe its purpose, key exports, and
   relationships with other files.
4. **Suggested Approach**: High-level fix strategy.

ANALYSIS GUIDELINES:
- Read ALL the code carefully before concluding anything
- Trace the execution flow from entry point to where the bug manifests
- Look for edge cases: null/None checks, off-by-one, type mismatches
- Consider imports and cross-file dependencies
- Reference specific function names, variable names, and line patterns
- If the issue mentions an error message or stack trace, trace it through the code

OUTPUT:
Provide structured analysis covering bug location, root cause, file
dependencies, and a suggested approach. Be specific and reference actual
code from the files.
"""


# ── Node Function ────────────────────────────────────────────────────────────

def research_agent(state: AgentState) -> dict:
    """
    LangGraph node: Research Agent.

    Reads the content of all relevant files and produces a dependency
    map + root cause analysis via LLM.
    """
    repo_name = state["repo_name"]
    relevant_files = state["relevant_files"]
    issue_title = state["issue_title"]
    issue_body = state["issue_body"]
    issue_number = state.get("issue_number", 0)

    log.info(
        "research agent started",
        repo=repo_name,
        file_count=len(relevant_files),
    )

    # ── Step 1: Read content of each relevant file ───────────────────────
    file_contents: dict[str, str] = {}

    for file_path in relevant_files:
        try:
            content = read_file_content.invoke({
                "repo_full_name": repo_name,
                "file_path": file_path,
            })
            file_contents[file_path] = content
            log.info("file read", path=file_path, size=len(content))
        except Exception as e:
            # If a file can't be read (deleted, binary, etc.), skip it
            log.warning("failed to read file", path=file_path, error=str(e))
            file_contents[file_path] = f"<ERROR: Could not read file — {e}>"

    log.info("all files read", read_count=len(file_contents))

    # ── Step 2: Build the prompt with all file contents ──────────────────
    files_section = ""
    for path, content in file_contents.items():
        files_section += (
            f"\n### File: `{path}`\n"
            f"```\n{content}\n```\n\n"
        )

    human_message = (
        f"## GitHub Issue #{issue_number}: {issue_title}\n\n"
        f"{issue_body}\n\n"
        f"---\n\n"
        f"## Relevant Source Files\n"
        f"{files_section}\n"
        f"---\n\n"
        f"Analyze these files, identify the bug location, map dependencies, "
        f"and suggest a fix approach."
    )

    # ── Step 3: Ask LLM for structured analysis ─────────────────────────
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=8192,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ResearchOutput)

    result: ResearchOutput = structured_llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=human_message),
    ])

    # Serialize the dependency map to a JSON string for state
    dependency_map = result.model_dump_json(indent=2)

    log.info(
        "research complete",
        bug_location=result.bug_location[:200],
        files_analyzed=len(result.dependency_analysis),
    )

    # ── Return updated state fields ──────────────────────────────────────
    return {
        "file_contents": file_contents,
        "dependency_map": dependency_map,
        "messages": [
            {
                "agent": "research",
                "action": f"Analyzed {len(file_contents)} files",
                "bug_location": result.bug_location,
                "root_cause": result.root_cause_analysis,
                "suggested_approach": result.suggested_approach,
            }
        ],
    }
