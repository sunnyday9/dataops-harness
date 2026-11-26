from __future__ import annotations

from pathlib import Path

from src.core.loop import DEFAULT_STATIC_PROMPT

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"
GLOBAL_PLAYBOOK_FILENAME = "GLOBAL_PLAYBOOK.md"


def load_global_playbook(skills_dir: Path | None = None) -> str:
    """Load the static global operating playbook text from `skills/GLOBAL_PLAYBOOK.md`."""
    resolved_dir = (skills_dir or DEFAULT_SKILLS_DIR).resolve()
    playbook_path = resolved_dir / GLOBAL_PLAYBOOK_FILENAME
    if not playbook_path.is_file():
        raise FileNotFoundError(f"Global playbook not found: {playbook_path}")
    return playbook_path.read_text(encoding="utf-8").strip()


def list_lazy_playbooks(skills_dir: Path | None = None) -> list[str]:
    """Return sorted relative `skills/<name>.md` paths for on-demand engine playbooks."""
    resolved_dir = (skills_dir or DEFAULT_SKILLS_DIR).resolve()
    if not resolved_dir.is_dir():
        return []
    entries: list[str] = []
    for path in sorted(resolved_dir.glob("*.md")):
        if path.name == GLOBAL_PLAYBOOK_FILENAME:
            continue
        entries.append(f"skills/{path.name}")
    return entries


def build_static_system_prompt(skills_dir: Path | None = None) -> str:
    """Build a cache-stable system prefix combining base instructions, global playbook, and lazy index.

    Lazy engine playbooks are intentionally indexed by path only so they are loaded via
    `read_file` on demand without bloating or mutating the prefix cache across turns.
    """
    global_text = load_global_playbook(skills_dir)
    lazy_paths = list_lazy_playbooks(skills_dir)
    parts = [DEFAULT_STATIC_PROMPT, "", global_text]
    if lazy_paths:
        index_lines = "\n".join(f"- {path}" for path in lazy_paths)
        parts.extend(
            [
                "",
                "Available engine-specific playbooks (load via `read_file` when relevant):",
                index_lines,
            ]
        )
    return "\n".join(parts)
