# =============================================================================
#  agents/coder.py  —  Coder Agent (Agent 03)
# =============================================================================
#
#  WHAT DOES THIS AGENT DO?
#  ────────────────────────
#  The Coder Agent is the THIRD node in the pipeline. It has two modes:
#
#  FIRST ATTEMPT:
#    - Receives issue details, all file contents, and the dependency map
#    - Generates a step-by-step fix plan
#    - Produces modified file contents that fix the issue
#    - Writes a pytest file that verifies the fix
#
#  RETRY (after failed verification):
#    - In addition to the above, receives error_logs from the failed run
#    - Reflects on what went wrong
#    - Produces a corrected fix and updated test
#
#  INPUTS (from state):
#    - issue_title        str
#    - issue_body         str
#    - issue_number       int
#    - file_contents      dict[str, str]
#    - dependency_map     str
#    - error_logs         str           (only on retry)
#    - retry_count        int           (only on retry)
#
#  OUTPUTS (written to state):
#    - plan               str
#    - patch              str           (JSON: {filepath: new_content})
#    - test_code          str
#    - messages           list[dict]
#
# =============================================================================

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config import settings
from logger import get_logger
from state import AgentState

log = get_logger(__name__)


# ── Structured Output Model ─────────────────────────────────────────────────

class CoderOutput(BaseModel):
    """Structured output from the Coder Agent's LLM call."""

    plan: str = Field(
        description=(
            "Step-by-step strategy for fixing the issue. Each step should "
            "reference specific files, functions, and what changes are needed."
        )
    )
    file_changes: dict[str, str] = Field(
        description=(
            "Dictionary mapping file paths to their COMPLETE new content. "
            "Each value must be the ENTIRE file content after the fix, "
            "not just a diff or snippet. Include ALL files that need changes."
        )
    )
    test_file_path: str = Field(
        description=(
            "Path where the test file should be written, e.g. "
            "'tests/test_fix_issue_42.py'"
        )
    )
    test_code: str = Field(
        description=(
            "Complete pytest file content that verifies the fix works. "
            "Must include imports, test functions, and assertions. "
            "Tests should cover both the fix AND edge cases."
        )
    )


# ── System Prompt ────────────────────────────────────────────────────────────

CODER_SYSTEM_PROMPT = """\
You are an expert software engineer who writes precise, production-quality code fixes.

YOUR TASK:
Given a GitHub issue and the relevant source code, produce:
1. A step-by-step fix plan
2. Modified file contents that fix the issue
3. A pytest test file that verifies the fix

CODE QUALITY RULES:
- Return the COMPLETE content of every modified file, not just the changed lines
- Preserve all existing functionality — don't break anything else
- Follow the existing code style (indentation, naming conventions, imports)
- Add comments only where the fix is non-obvious
- Handle edge cases (None values, empty strings, missing keys)

TEST QUALITY RULES:
- Write pytest-style tests (def test_xxx():)
- Import the actual modules being tested
- Test both the happy path (fix works) AND edge cases
- Use descriptive test names that explain what's being verified
- Keep tests self-contained — no external dependencies or network calls
- Include at least 2 test functions

FILE CHANGES:
- Only include files that actually need modifications
- Each file value must be the ENTIRE file content (not a diff)
- Maintain correct Python imports and module structure
- Do NOT add new dependencies unless absolutely necessary

IMPORTANT:
- If the issue mentions a specific error, make sure your fix prevents that exact error
- If there's a stack trace, trace through the code to find the root cause
- Test file path should follow the pattern: tests/test_fix_issue_{N}.py
"""

RETRY_ADDENDUM = """\

⚠️  YOUR PREVIOUS ATTEMPT FAILED.

The test output from your last attempt is shown below. Study the errors
carefully, understand what went wrong, and produce a CORRECTED fix.

Common failure reasons:
- Import errors: wrong module path or missing dependency
- Assertion errors: fix didn't fully address the issue
- Syntax errors: malformed code in the generated changes
- Type errors: wrong argument types or missing parameters
- Test isolation: tests depending on external state

Error logs from previous attempt:
```
{error_logs}
```

This is retry #{retry_count}. Learn from the errors and fix them.
Do NOT repeat the same mistakes.
"""


# ── Node Function ────────────────────────────────────────────────────────────

def coder_agent(state: AgentState) -> dict:
    """
    LangGraph node: Coder Agent.

    Generates code fixes and tests. On retry, reflects on previous
    errors and produces corrected output.
    """
    issue_title = state["issue_title"]
    issue_body = state["issue_body"]
    issue_number = state.get("issue_number", 0)
    file_contents = state["file_contents"]
    dependency_map = state["dependency_map"]
    error_logs = state.get("error_logs", "")
    retry_count = state.get("retry_count", 0)

    is_retry = retry_count > 0 and error_logs
    mode = "retry" if is_retry else "first_attempt"

    log.info(
        "coder agent started",
        mode=mode,
        retry_count=retry_count,
        issue_number=issue_number,
    )

    # ── Step 1: Build the prompt ─────────────────────────────────────────
    # File contents section
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
        f"## Codebase Analysis (from Research Agent)\n\n"
        f"```json\n{dependency_map}\n```\n\n"
        f"---\n\n"
        f"## Source Files\n"
        f"{files_section}\n"
        f"---\n\n"
        f"Generate the fix, modified files, and test code."
    )

    # Build system prompt — add retry context if this is a retry
    system_prompt = CODER_SYSTEM_PROMPT
    if is_retry:
        system_prompt += RETRY_ADDENDUM.format(
            error_logs=error_logs,
            retry_count=retry_count,
        )
        log.info("retry context added to prompt", retry_count=retry_count)

    # ── Step 2: Call LLM with structured output ──────────────────────────
    llm = ChatAnthropic(
        model=settings.model_name,
        api_key=settings.anthropic_api_key,
        max_tokens=16384,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(CoderOutput)

    result: CoderOutput = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message),
    ])

    log.info(
        "coder agent complete",
        mode=mode,
        files_changed=list(result.file_changes.keys()),
        test_path=result.test_file_path,
        plan_length=len(result.plan),
    )

    # ── Step 3: Serialize patch as JSON string for state ─────────────────
    patch = json.dumps(result.file_changes, indent=2)

    # ── Return updated state fields ──────────────────────────────────────
    return {
        "plan": result.plan,
        "patch": patch,
        "test_code": result.test_code,
        "messages": [
            {
                "agent": "coder",
                "action": f"Generated fix ({mode})",
                "mode": mode,
                "retry_count": retry_count,
                "files_changed": list(result.file_changes.keys()),
                "test_file": result.test_file_path,
                "plan_summary": result.plan[:300],
            }
        ],
    }
