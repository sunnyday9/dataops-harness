from __future__ import annotations

from pathlib import Path

from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.core.context import IncidentInput, ToolCall, UserContext
from src.core.loop import HarnessLoop
from src.core.playbooks import (
    DEFAULT_SKILLS_DIR,
    build_static_system_prompt,
    list_lazy_playbooks,
    load_global_playbook,
)
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = DEFAULT_SKILLS_DIR


def test_global_playbook_and_lazy_index_prefix_isolation() -> None:
    """Global Playbook loads statically without repo dev rules or eager lazy bodies."""
    global_text = load_global_playbook(SKILLS_DIR)
    lazy_paths = list_lazy_playbooks(SKILLS_DIR)
    prefix = build_static_system_prompt(SKILLS_DIR)

    assert "RCA Report" in global_text
    assert lazy_paths == [
        "skills/dbt_troubleshooting.md",
        "skills/spark_troubleshooting.md",
    ]

    # Prefix includes global playbook + lazy paths index, never lazy bodies themselves.
    assert global_text in prefix
    for rel_path in lazy_paths:
        assert rel_path in prefix
    assert "Driver Out-of-Memory (`384m` Heap" not in prefix
    assert "SQL `NOT IN` NULL Trap" not in prefix

    # Zero collision with repository-level developer instructions from AGENTS.md.
    for repo_only_marker in ("sunnyday9/dataops-harness", "needs-triage", "triage-labels"):
        assert repo_only_marker not in prefix


def test_e2e_lazy_spark_and_dbt_playbook_loading_preserves_prefix_stability(
    tmp_path: Path,
) -> None:
    """FakeReplayModelAdapter loads engine playbooks lazily via read_file with byte-identical prefix."""
    ws = tmp_path / "workspace"
    (ws / "spark_jobs").mkdir(parents=True)
    (ws / "spark_jobs" / "build_silver_orders.py").write_text("# spark job\n", encoding="utf-8")
    (ws / "dbt_project" / "models" / "marts").mkdir(parents=True)
    (ws / "dbt_project" / "models" / "marts" / "fct_daily_gmv.sql").write_text(
        "SELECT 1\n", encoding="utf-8"
    )

    static_prompt = build_static_system_prompt(SKILLS_DIR)
    registry = ToolRegistry(workspace_dir=ws, skills_dir=SKILLS_DIR)

    # 1. Spark incident -> agent reads skills/spark_troubleshooting.md lazily via read_file
    spark_model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(
                        id="s1",
                        name="read_file",
                        arguments={"path": "skills/spark_troubleshooting.md"},
                    )
                ]
            ),
            TurnResponse(content="Verified Spark skew remediation via spark_troubleshooting.md."),
        ]
    )
    spark_loop = HarnessLoop(
        model_adapter=spark_model,
        tool_registry=registry,
        runs_dir=tmp_path / "runs_spark",
        static_system_prompt=static_prompt,
    )
    spark_state = spark_loop.run(
        IncidentInput(
            incident_id="INC-SPARK-PLAYBOOK",
            title="Silver orders GMV doubled",
            summary="Suspected append write mode in build_silver_orders.py",
            affected_assets=["silver_orders", "fct_daily_gmv"],
        ),
        UserContext(role="de"),
        run_id="run-pb-spark",
    )
    assert spark_state.status == "AWAITING_APPROVAL"
    spark_tool_msgs = [m for m in spark_state.messages if m.role == "tool"]
    assert "Driver Out-of-Memory" in spark_tool_msgs[0].content
    assert "DESCRIBE HISTORY silver_orders" in spark_tool_msgs[0].content
    assert (
        spark_model.recorded_requests[0].static_system_prompt
        == spark_model.recorded_requests[1].static_system_prompt
        == static_prompt
    )
    assert spark_model.recorded_requests[0].tools == spark_model.recorded_requests[1].tools

    # 2. dbt incident -> agent reads skills/dbt_troubleshooting.md lazily via read_file
    dbt_model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(
                        id="d1",
                        name="read_file",
                        arguments={"path": "skills/dbt_troubleshooting.md"},
                    )
                ]
            ),
            TurnResponse(content="Verified NOT IN NULL trap via dbt_troubleshooting.md."),
        ]
    )
    dbt_loop = HarnessLoop(
        model_adapter=dbt_model,
        tool_registry=registry,
        runs_dir=tmp_path / "runs_dbt",
        static_system_prompt=static_prompt,
    )
    dbt_state = dbt_loop.run(
        IncidentInput(
            incident_id="INC-DBT-PLAYBOOK",
            title="fct_daily_gmv dropped valid orders",
            summary="Suspected SQL NOT IN NULL trap in dbt model",
            affected_assets=["fct_daily_gmv"],
        ),
        UserContext(role="da"),
        run_id="run-pb-dbt",
    )
    assert dbt_state.status == "DIAGNOSED"
    dbt_tool_msgs = [m for m in dbt_state.messages if m.role == "tool"]
    assert "SQL `NOT IN` NULL Trap" in dbt_tool_msgs[0].content
    assert (
        dbt_model.recorded_requests[0].static_system_prompt
        == dbt_model.recorded_requests[1].static_system_prompt
        == static_prompt
    )
    assert dbt_model.recorded_requests[0].tools == dbt_model.recorded_requests[1].tools
