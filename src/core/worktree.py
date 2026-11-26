from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

_SYMLINK_MODE_PREFIXES = ("new file mode 120", "new mode 120", "old mode 120")


def _clean_patch_path(raw: str) -> str:
    cleaned = raw.strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    if cleaned.startswith(("a/", "b/")):
        cleaned = cleaned[2:]
    return cleaned


def extract_raw_patch_paths(patch_text: str) -> list[str]:
    """Extract all workspace-relative candidate file paths from a unified diff & reject symlinks."""
    targets: list[str] = []

    def _add(raw_path: str) -> None:
        if not raw_path or raw_path == "/dev/null":
            return
        cleaned = _clean_patch_path(raw_path)
        if cleaned and cleaned not in targets:
            targets.append(cleaned)

    for line in patch_text.splitlines():
        if line.startswith(_SYMLINK_MODE_PREFIXES):
            raise PermissionError("Symbolic link modifications in apply_patch are forbidden.")
        if line.startswith("diff --git "):
            try:
                tokens = shlex.split(line)
            except ValueError:
                tokens = line.split()
            for tok in tokens[2:]:
                _add(tok)
            continue
        if line.startswith(("--- ", "+++ ")):
            _add(line[4:].split("\t", 1)[0])
            continue
        for prefix in ("rename from ", "rename to ", "copy from ", "copy to "):
            if line.startswith(prefix):
                _add(line[len(prefix) :])
                break

    if not targets:
        raise ValueError("Patch contains no valid target files.")
    return targets

_IGNORED_PREFIXES = (
    ".scratch/",
    "lakehouse/",
    "incidents/",
    "dbt_project/target/",
    "dbt_project/logs/",
    "dbt_project/dbt_packages/",
)
_IGNORED_EXACT = frozenset({".scratch", "lakehouse", "incidents", "dbt_project/.user.yml"})


def _is_ignored_workspace_path(rel_path: str) -> bool:
    norm = rel_path.replace("\\", "/").lstrip("./")
    if norm in _IGNORED_EXACT or norm.startswith(_IGNORED_PREFIXES):
        return True
    parts = norm.split("/")
    return "__pycache__" in parts or norm.endswith(".pyc")


class RunWorktreeManager:
    """Manage per-Run detached git worktrees isolated from the production workspace."""

    @staticmethod
    def worktree_path(runs_dir: Path, run_id: str) -> Path:
        return runs_dir.resolve() / run_id / "worktree"

    @staticmethod
    def _ensure_git_repo(workspace_dir: Path) -> Path:
        resolved = workspace_dir.resolve()
        if not (resolved / ".git").exists():
            raise ValueError(f"Workspace is not a git repository: {resolved}")
        return resolved

    @classmethod
    def create(cls, workspace_dir: Path, runs_dir: Path, run_id: str) -> Path:
        """Create a clean detached git worktree at `runs/<run_id>/worktree` from `HEAD`."""
        repo_root = cls._ensure_git_repo(workspace_dir)
        wt = cls.worktree_path(runs_dir, run_id)
        if wt.exists():
            cls.cleanup(repo_root, wt)
        wt.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", "--detach", str(wt), "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return wt

    @staticmethod
    def _status_entries(worktree_dir: Path) -> list[tuple[str, str]]:
        proc = subprocess.run(
            ["git", "-C", str(worktree_dir), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            check=True,
            capture_output=True,
        )
        tokens = [tok.decode("utf-8", errors="replace") for tok in proc.stdout.split(b"\0") if tok]
        entries: list[tuple[str, str]] = []
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            status_code = token[:2]
            path_part = token[3:]
            # Renames emit a second NUL token for the original path.
            if status_code[0] in ("R", "C") or status_code[1] in ("R", "C"):
                idx += 1
            norm = path_part.replace("\\", "/").lstrip("./")
            if not _is_ignored_workspace_path(norm):
                entries.append((status_code, norm))
            idx += 1
        return entries

    @classmethod
    def changed_files(cls, worktree_dir: Path) -> list[str]:
        """Return sorted workspace-relative paths modified or added inside `worktree_dir`."""
        return sorted({path for _status, path in cls._status_entries(worktree_dir)})

    @classmethod
    def unified_diff(cls, worktree_dir: Path) -> str:
        """Return the complete unified diff of tracked and new files inside `worktree_dir`."""
        resolved = worktree_dir.resolve()
        entries = cls._status_entries(resolved)
        if not entries:
            return ""

        tracked_paths = [path for status, path in entries if status != "??"]
        untracked_paths = [path for status, path in entries if status == "??"]
        chunks: list[str] = []

        if tracked_paths:
            diff_proc = subprocess.run(
                ["git", "-C", str(resolved), "diff", "HEAD", "--", *tracked_paths],
                check=True,
                capture_output=True,
                text=True,
            )
            if diff_proc.stdout.strip():
                chunks.append(diff_proc.stdout.rstrip("\n"))

        for rel in sorted(dict.fromkeys(untracked_paths)):
            untracked_proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(resolved),
                    "diff",
                    "--no-index",
                    "--",
                    "/dev/null",
                    rel,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if untracked_proc.returncode not in (0, 1):
                raise RuntimeError(
                    f"git diff --no-index failed for {rel}: {untracked_proc.stderr.strip()}"
                )
            if untracked_proc.stdout.strip():
                chunks.append(untracked_proc.stdout.rstrip("\n"))

        return "\n".join(chunks) + ("\n" if chunks else "")

    @staticmethod
    def cleanup(workspace_dir: Path, worktree_dir: Path) -> None:
        """Remove `worktree_dir` cleanly from git worktree metadata and disk."""
        repo_root = workspace_dir.resolve()
        wt = worktree_dir.resolve()
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(wt)],
            check=False,
            capture_output=True,
            text=True,
        )
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
        )
