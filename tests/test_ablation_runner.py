from __future__ import annotations

import json
from pathlib import Path

from benchmarks.ablation_runner import (
    ARM_CONFIGS,
    AblationBenchmarkRunner,
    _ArmToolRegistry,
    _NoCacheModelAdapter,
    main,
    render_markdown_report,
)
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnRequest, TurnResponse
from src.core.context import Message, ToolCall
from src.core.loop import DEFAULT_STATIC_PROMPT
from src.core.playbooks import build_static_system_prompt


def test_ablation_feature_flag_separation(tmp_path: Path) -> None:
    """Verify Group A/B/C cleanly separate lineage tool, playbooks, cache, and safety flags."""
    a = ARM_CONFIGS["group_a"]
    b = ARM_CONFIGS["group_b"]
    c = ARM_CONFIGS["group_c"]

    assert a.enable_lineage_tool is False
    assert a.enable_playbooks is False
    assert a.enable_compactor is False
    assert a.enable_prompt_caching is False
    assert a.enable_stop_gate is False
    assert a.enable_rbac_hook is False
    assert a.enable_sql_guard is False
    assert a.enable_pii_redactor is False
    assert a.enable_anti_thrash is False
    assert a.enable_alert_aggregator is False

    assert b.enable_lineage_tool is True
    assert b.enable_playbooks is True
    assert b.enable_compactor is True
    assert b.enable_prompt_caching is True
    assert b.enable_stop_gate is False
    assert b.enable_rbac_hook is False

    assert all(
        (
            c.enable_lineage_tool,
            c.enable_playbooks,
            c.enable_compactor,
            c.enable_prompt_caching,
            c.enable_rbac_hook,
            c.enable_sql_guard,
            c.enable_pii_redactor,
            c.enable_stop_gate,
            c.enable_anti_thrash,
            c.enable_alert_aggregator,
        )
    )

    ws = tmp_path / "ws"
    ws.mkdir()
    reg_a = _ArmToolRegistry(ws, enable_lineage_tool=False)
    reg_b = _ArmToolRegistry(ws, enable_lineage_tool=True)
    assert "query_lineage" not in {d.name for d in reg_a.definitions()}
    assert "query_lineage" in {d.name for d in reg_b.definitions()}
    blocked_lineage = reg_a.execute(ToolCall(id="x1", name="query_lineage", arguments={}))
    assert blocked_lineage.blocked is True and blocked_lineage.is_error is True

    # Prompt caching wrapper converts cached tokens into uncached input for Group A.
    inner = FakeReplayModelAdapter([TurnResponse(content="t1"), TurnResponse(content="t2")])
    no_cache = _NoCacheModelAdapter(inner)
    req = TurnRequest(static_system_prompt="SAME", messages=[Message(role="user", content="hi")])
    no_cache.complete(req)
    resp2 = no_cache.complete(req)
    assert resp2.usage.cached_input_tokens == 0
    assert len(build_static_system_prompt()) > len(DEFAULT_STATIC_PROMPT)


def test_ablation_runner_zero_token_dry_run_all_fourteen_scenarios(tmp_path: Path) -> None:
    """Execute zero-token fake_replay ablation across all 14 scenarios and verify report metrics."""
    md_out = tmp_path / "ablation_report.md"
    json_out = tmp_path / "ablation_report.json"

    rc = main(
        [
            "--provider",
            "fake_replay",
            "--max-turns",
            "20",
            "--output",
            str(md_out),
            "--json-output",
            str(json_out),
        ]
    )
    assert rc == 0
    assert md_out.is_file()
    assert json_out.is_file()

    md_text = md_out.read_text(encoding="utf-8")
    for required_heading in (
        "Verified Fix Rate",
        "Safety Interception Rate",
        "Avg Turns",
        "Token Cost",
        "Prompt Cache Hit Rate",
        "Failure Attribution",
    ):
        assert required_heading in md_text

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload["scenario_ids"]) == 14
    arms = {arm["arm_id"]: arm for arm in payload["arms"]}
    assert set(arms) == {"group_a", "group_b", "group_c"}

    arm_a, arm_b, arm_c = arms["group_a"], arms["group_b"], arms["group_c"]
    assert arm_a["cost_usd"] == 0.0
    assert arm_b["cost_usd"] == 0.0
    assert arm_c["cost_usd"] == 0.0

    assert arm_a["verified_fix_rate"] < arm_b["verified_fix_rate"] < arm_c["verified_fix_rate"]
    assert arm_c["verified_fix_rate"] == 100.0
    assert arm_a["safety_interception_rate"] == 0.0
    assert arm_b["safety_interception_rate"] == 0.0
    assert arm_c["safety_interception_rate"] == 100.0
    assert arm_c["pii_leak_incidents"] == 0
    assert arm_a["prompt_cache_hit_rate"] == 0.0
    assert arm_b["prompt_cache_hit_rate"] > 0.0
    assert arm_c["prompt_cache_hit_rate"] > 0.0
    assert arm_c["scenarios_passed"] == 14
    assert arm_a["failure_attribution_counts"]
    assert not arm_c["failure_attribution_counts"]


def test_ablation_report_renderer_standalone(tmp_path: Path) -> None:
    """Quick single-scenario runner + renderer sanity check."""
    runner = AblationBenchmarkRunner(provider="fake_replay", max_turns=20)
    report = runner.run(scenario_ids=["ops_alert_storm_convergence"])
    rendered = render_markdown_report(report)
    assert "ops_alert_storm_convergence" in rendered
    assert "alert_storm_amplification" in rendered
