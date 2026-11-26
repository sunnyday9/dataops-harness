from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.core.playbooks import DEFAULT_SKILLS_DIR, list_lazy_playbooks

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = REPO_ROOT / ".agents" / "skills"
PLACEHOLDER_TOKENS = ("TODO", "TBD", "FIXME", "<FILL_", "PLACEHOLDER")


def _parse_frontmatter(skill_md: Path) -> dict[str, object]:
    text = skill_md.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{skill_md} must begin with YAML frontmatter."
    parts = text.split("---\n", 2)
    assert len(parts) >= 3, f"{skill_md} frontmatter closing delimiter missing."
    loaded = yaml.safe_load(parts[1])
    assert isinstance(loaded, dict), f"{skill_md} frontmatter must deserialize to dict."
    return loaded


def test_project_skills_frontmatter_zero_placeholders_and_runtime_playbooks_intact() -> None:
    """Validate skill frontmatter, zero placeholders, mapped symbols, and untouched playbooks."""
    assert list_lazy_playbooks(DEFAULT_SKILLS_DIR) == [
        "skills/dbt_troubleshooting.md",
        "skills/spark_troubleshooting.md",
    ]

    for skill_name in ("verify-run-state-machine", "verify-dataops-harness"):
        skill_dir = SKILLS_ROOT / skill_name
        skill_md = skill_dir / "SKILL.md"
        assert skill_md.is_file()
        fm = _parse_frontmatter(skill_md)
        assert fm.get("name") == skill_name
        assert isinstance(fm.get("description"), str) and len(str(fm["description"]).strip()) >= 20

        for md_file in skill_dir.rglob("*.md"):
            body = md_file.read_text(encoding="utf-8")
            for token in PLACEHOLDER_TOKENS:
                assert token not in body, f"Placeholder {token!r} found in {md_file}"

    formal_md = (SKILLS_ROOT / "verify-run-state-machine" / "SKILL.md").read_text(encoding="utf-8")
    for expected_path in (
        "src/core/context.py",
        "src/core/loop.py",
        "src/hooks/stop_gate.py",
        "src/core/promotion.py",
        "configs/policies.yaml",
    ):
        assert expected_path in formal_md
        assert (REPO_ROOT / expected_path).is_file()


def test_formal_state_machine_skill_passes_bfs_negative_control_and_conformance(
    tmp_path: Path,
) -> None:
    """Run verify-run-state-machine checker and verify BFS, negative control, and 6 live checks."""
    formal_out = tmp_path / "formal_check.json"
    script = SKILLS_ROOT / "verify-run-state-machine" / "scripts" / "check_run_state_machine.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--json-output", str(formal_out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    data = json.loads(formal_out.read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert data["bfs_model_check"]["passed"] is True
    assert data["bfs_model_check"]["states_explored"] > 20
    assert data["mutation_negative_control"]["bug_caught"] is True
    live = data["live_code_conformance"]
    assert live["passed"] is True
    assert len(live["checks"]) == 6
    assert all(c["passed"] for c in live["checks"])


def test_verify_dataops_harness_end_to_end_and_proof_survival(tmp_path: Path) -> None:
    """Run verify-dataops-harness end-to-end and prove all 5 proof artifacts survive teardown."""
    proof_dir = tmp_path / "durable_proof"
    script = SKILLS_ROOT / "verify-dataops-harness" / "scripts" / "run_verification.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--proof-dir", str(proof_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    summary = json.loads(proc.stdout)
    assert summary["passed"] is True
    assert summary["isolated_removed"] is True
    assert not Path(summary["isolated_root"]).exists()

    for artifact in (
        "doctor.json",
        "cli_lifecycle.json",
        "formal_check.json",
        "ablation_report.md",
        "verification_summary.md",
    ):
        p = proof_dir / artifact
        assert p.is_file() and p.stat().st_size > 0, f"Missing or empty proof: {artifact}"

    cli_data = json.loads((proof_dir / "cli_lifecycle.json").read_text(encoding="utf-8"))
    assert cli_data["final_status"]["status"] == "RESOLVED"
    assert abs(cli_data["promoted_gmv_total"] - cli_data["baseline_gmv_total"]) <= 0.01
