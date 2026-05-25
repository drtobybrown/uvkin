"""Git revision helpers for run.log provenance."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitRevision:
    path: str
    sha: str
    dirty: bool
    branch: str | None


def _git_rev(repo: Path) -> GitRevision | None:
    if not (repo / ".git").exists():
        return None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            != ""
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return GitRevision(path=str(repo.resolve()), sha=sha, dirty=dirty, branch=branch)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def uvkin_git_revision() -> GitRevision | None:
    repo = Path(__file__).resolve().parent.parent
    return _git_rev(repo)


def uvfit_git_revision() -> GitRevision | None:
    env = os.environ.get("UVFIT_PATH")
    if env:
        return _git_rev(Path(env))
    try:
        import uvfit

        pkg = Path(uvfit.__file__).resolve().parent
        repo = pkg.parent.parent
        return _git_rev(repo)
    except ImportError:
        return None


def format_git_log_block() -> str:
    lines = ["GIT REVISIONS:"]
    for label, rev_fn in (("uvkin", uvkin_git_revision), ("uvfit", uvfit_git_revision)):
        rev = rev_fn()
        if rev is None:
            lines.append(f"  {label}: (not a git checkout or git unavailable)")
        else:
            dirty = " dirty" if rev.dirty else ""
            br = rev.branch or "?"
            lines.append(f"  {label}: {rev.sha[:12]}{dirty}  branch={br}  path={rev.path}")
    return "\n".join(lines)
