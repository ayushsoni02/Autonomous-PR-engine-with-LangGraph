# =============================================================================
#  tools/github_tools.py  —  GitHub API Functions (LangChain @tool)
# =============================================================================
#
#  WHAT IS THIS FILE?
#  ──────────────────
#  Six functions that wrap PyGithub operations. Each is decorated with
#  LangChain's @tool so that LLM agents can call them natively.
#
#  These tools are the ONLY way agents interact with GitHub — no agent
#  makes raw API calls. This keeps all GitHub logic in one place.
#
#  USAGE:
#  ──────
#  # Agents bind these as tools:
#  from tools.github_tools import get_issue_details, get_file_tree, ...
#
#  # Direct invocation (for testing):
#  result = get_issue_details.invoke({"issue_url": "https://github.com/..."})
#
# =============================================================================

from __future__ import annotations

import re
from typing import Any

from github import Github, GithubException
from langchain_core.tools import tool

from config import settings
from logger import get_logger

log = get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_github_client() -> Github:
    """Create a fresh PyGithub client using the configured token."""
    return Github(settings.github_token)


def _parse_issue_url(issue_url: str) -> tuple[str, int]:
    """
    Parse a GitHub issue URL into (repo_full_name, issue_number).

    Accepts URLs like:
        https://github.com/owner/repo/issues/42
        http://github.com/owner/repo/issues/42

    Returns:
        ("owner/repo", 42)

    Raises:
        ValueError: if the URL doesn't match the expected pattern.
    """
    pattern = r"https?://github\.com/([^/]+/[^/]+)/issues/(\d+)"
    match = re.match(pattern, issue_url)
    if not match:
        raise ValueError(
            f"Invalid GitHub issue URL: {issue_url}\n"
            f"Expected format: https://github.com/owner/repo/issues/123"
        )
    repo_full_name = match.group(1)
    issue_number = int(match.group(2))
    return repo_full_name, issue_number


# =============================================================================
#  TOOL 1: get_issue_details
# =============================================================================

@tool
def get_issue_details(issue_url: str) -> dict[str, Any]:
    """
    Fetch the title, body, labels, and repo name for a GitHub issue.

    Args:
        issue_url: Full GitHub issue URL, e.g.
                   "https://github.com/owner/repo/issues/42"

    Returns:
        A dict with keys: title, body, labels, repo_name, issue_number
    """
    repo_full_name, issue_number = _parse_issue_url(issue_url)
    log.info("fetching issue details", repo=repo_full_name, issue=issue_number)

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)
    issue = repo.get_issue(number=issue_number)

    result = {
        "title": issue.title,
        "body": issue.body or "",
        "labels": [label.name for label in issue.labels],
        "repo_name": repo_full_name,
        "issue_number": issue_number,
    }

    log.info("issue fetched", title=result["title"], labels=result["labels"])
    return result


# =============================================================================
#  TOOL 2: get_file_tree
# =============================================================================

@tool
def get_file_tree(repo_full_name: str) -> list[str]:
    """
    Get the full list of file paths in a GitHub repository.

    Uses the Git tree API with recursive=True for efficiency (single API call).

    Args:
        repo_full_name: Repository in "owner/repo" format.

    Returns:
        A list of file path strings (directories excluded).
    """
    log.info("fetching file tree", repo=repo_full_name)

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)

    # Get the tree for the default branch, recursively
    default_branch = repo.default_branch
    tree = repo.get_git_tree(sha=default_branch, recursive=True)

    # Filter to blobs only (files, not directories)
    file_paths = [
        item.path
        for item in tree.tree
        if item.type == "blob"
    ]

    log.info("file tree fetched", file_count=len(file_paths))
    return file_paths


# =============================================================================
#  TOOL 3: read_file_content
# =============================================================================

@tool
def read_file_content(repo_full_name: str, file_path: str) -> str:
    """
    Read the content of a single file from a GitHub repository.

    Args:
        repo_full_name: Repository in "owner/repo" format.
        file_path: Path to the file within the repo, e.g. "src/main.py"

    Returns:
        The decoded text content of the file.
    """
    log.info("reading file", repo=repo_full_name, path=file_path)

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)

    file_content = repo.get_contents(file_path)

    # get_contents can return a list for directories — we only handle files
    if isinstance(file_content, list):
        raise ValueError(
            f"Path '{file_path}' is a directory, not a file. "
            f"Use get_file_tree to list files."
        )

    decoded = file_content.decoded_content.decode("utf-8")
    log.info("file read", path=file_path, size=len(decoded))
    return decoded


# =============================================================================
#  TOOL 4: create_branch
# =============================================================================

@tool
def create_branch(repo_full_name: str, branch_name: str) -> str:
    """
    Create a new branch in the repository from the default branch HEAD.

    Args:
        repo_full_name: Repository in "owner/repo" format.
        branch_name: Name for the new branch, e.g. "fix/issue-42-handle-null"

    Returns:
        The full ref string, e.g. "refs/heads/fix/issue-42-handle-null"
    """
    log.info("creating branch", repo=repo_full_name, branch=branch_name)

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)

    # Get the SHA of the default branch HEAD
    default_branch = repo.default_branch
    source_branch = repo.get_branch(default_branch)
    sha = source_branch.commit.sha

    # Create the new branch ref
    ref = f"refs/heads/{branch_name}"
    repo.create_git_ref(ref=ref, sha=sha)

    log.info("branch created", ref=ref, base_sha=sha[:8])
    return ref


# =============================================================================
#  TOOL 5: commit_file_changes
# =============================================================================

@tool
def commit_file_changes(
    repo_full_name: str,
    branch_name: str,
    file_changes: dict[str, str],
    commit_message: str,
) -> str:
    """
    Commit one or more file changes to a branch.

    For each file in file_changes, this either creates or updates the file
    on the specified branch. Changes are committed one file at a time
    (GitHub Contents API limitation).

    Args:
        repo_full_name: Repository in "owner/repo" format.
        branch_name: Target branch name (without refs/heads/).
        file_changes: Dict of {file_path: new_file_content}.
        commit_message: The commit message.

    Returns:
        A summary string of what was committed.
    """
    log.info(
        "committing file changes",
        repo=repo_full_name,
        branch=branch_name,
        file_count=len(file_changes),
    )

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)

    committed_files = []

    for file_path, new_content in file_changes.items():
        try:
            # Try to get existing file (to update it)
            existing = repo.get_contents(file_path, ref=branch_name)
            if isinstance(existing, list):
                log.warning("skipping directory path", path=file_path)
                continue

            repo.update_file(
                path=file_path,
                message=commit_message,
                content=new_content,
                sha=existing.sha,
                branch=branch_name,
            )
            log.info("file updated", path=file_path)

        except GithubException as e:
            if e.status == 404:
                # File doesn't exist yet — create it
                repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=new_content,
                    branch=branch_name,
                )
                log.info("file created", path=file_path)
            else:
                raise

        committed_files.append(file_path)

    summary = f"Committed {len(committed_files)} file(s) to {branch_name}: {committed_files}"
    log.info("commit complete", summary=summary)
    return summary


# =============================================================================
#  TOOL 6: open_pull_request
# =============================================================================

@tool
def open_pull_request(
    repo_full_name: str,
    branch_name: str,
    title: str,
    body: str,
) -> str:
    """
    Open a Pull Request from a feature branch to the default branch.

    Args:
        repo_full_name: Repository in "owner/repo" format.
        branch_name: The source branch (without refs/heads/).
        title: PR title.
        body: PR description (Markdown supported).

    Returns:
        The URL of the newly created Pull Request.
    """
    log.info("opening pull request", repo=repo_full_name, branch=branch_name)

    g = _get_github_client()
    repo = g.get_repo(repo_full_name)

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch_name,
        base=repo.default_branch,
    )

    log.info("pull request opened", pr_url=pr.html_url, pr_number=pr.number)
    return pr.html_url
