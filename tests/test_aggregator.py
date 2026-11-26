from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import FakeReplayModelAdapter, TurnResponse
from src.core.aggregator import (
    AggregatedIncident,
    AlertStormAggregator,
    RawAlert,
)
from src.core.context import IncidentInput, RunState, ToolCall, UserContext
from src.core.loop import HarnessLoop
from src.core.policy import load_policies
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"
BASE_TS = datetime(2026, 11, 26, 3, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def lineage_adapter() -> LineageAdapter:
    return LineageAdapter.from_workspace(WORKSPACE_DIR)


@pytest.fixture(scope="module")
def aggregator(lineage_adapter: LineageAdapter) -> AlertStormAggregator:
    return AlertStormAggregator(lineage=lineage_adapter)


def _alert(
    alert_id: str,
    asset: str,
    offset_seconds: int = 0,
    *,
    severity: str = "high",
    alert_type: str = "dbt_test_failure",
    summary: str = "",
    tz: bool = True,
) -> RawAlert:
    ts = BASE_TS + timedelta(seconds=offset_seconds)
    if not tz:
        ts = ts.replace(tzinfo=None)
    return RawAlert(
        alert_id=alert_id,
        asset=asset,
        timestamp=ts,
        severity=severity,
        alert_type=alert_type,
        summary=summary or f"{alert_type} on {asset}",
    )


def test_default_window_comes_from_policies_and_validates_bounds(
    lineage_adapter: LineageAdapter,
) -> None:
    """Default aggregator window matches configs/policies.yaml (900s) and rejects negative."""
    expected_window = load_policies().aggregator.window_seconds
    agg = AlertStormAggregator(lineage=lineage_adapter)
    assert expected_window == 900
    assert agg.window_seconds == 900
    assert agg.lineage is lineage_adapter
    assert agg.ingest([]) == []
    assert agg.dispatch_runs([], lambda inc, ctx: RunState(run_id="x", incident=inc, user_context=ctx)) == []

    with pytest.raises(ValueError, match="non-negative"):
        AlertStormAggregator(lineage=lineage_adapter, window_seconds=-1)


def test_silent_upstream_failure_converges_sibling_marts_via_lca(
    aggregator: AlertStormAggregator,
) -> None:
    """Silent upstream failure converges sibling downstream marts and surfaces LCA + shared ancestors."""
    alerts = [
        _alert("a-gmv", "fct_daily_gmv", 30, summary="assert_gmv_reconciliation failed"),
        _alert("a-ltv", "dim_user_ltv", 90, summary="not_null user_id failed"),
    ]

    incidents = aggregator.ingest(alerts)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.alert_ids == ["a-gmv", "a-ltv"]
    assert inc.affected_assets == ["fct_daily_gmv", "dim_user_ltv"]
    assert inc.root_cause_candidates == ["int_orders_usd"]
    assert "int_orders_usd" in inc.shared_ancestors
    assert "stg_orders" in inc.shared_ancestors
    assert "silver_orders" in inc.shared_ancestors
    assert inc.window_start == BASE_TS + timedelta(seconds=30)
    assert inc.window_end == BASE_TS + timedelta(seconds=90)

    inc_input = inc.to_incident_input()
    assert inc_input.incident_id == inc.incident_id
    assert inc_input.affected_assets == ["fct_daily_gmv", "dim_user_ltv"]
    assert inc_input.shared_ancestors[0] == "int_orders_usd"
    assert "silver_orders" in inc_input.shared_ancestors
    assert len(inc_input.alerts) == 2
    # Must remain JSON-serializable for HarnessLoop._format_initial_user_message.
    json.dumps(inc_input.model_dump())

    # Staging sibling alerts also surface their silent Spark upstream LCA (`silver_users`).
    stg_incidents = aggregator.ingest(
        [
            _alert("a-stg-ord", "stg_orders", 0),
            _alert("a-stg-usr", "stg_users", 120),
        ]
    )
    assert len(stg_incidents) == 1
    assert stg_incidents[0].root_cause_candidates == ["silver_users"]


def test_ancestor_descendant_storm_ranks_upstream_root_regardless_of_arrival_order(
    aggregator: AlertStormAggregator,
) -> None:
    """Upstream + downstream alerts within window converge and rank upstream ancestor first."""
    # Downstream mart fires first chronologically; upstream Spark job fires 200s later.
    alerts = [
        _alert("a-mart", "fct_daily_gmv", 0, severity="high"),
        _alert("a-stg", "stg_orders", 100, severity="medium"),
        _alert("a-spark", "silver_orders", 200, severity="critical"),
    ]

    incidents = aggregator.aggregate(list(reversed(alerts)))
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.alert_ids == ["a-mart", "a-stg", "a-spark"]
    assert inc.affected_assets == ["fct_daily_gmv", "stg_orders", "silver_orders"]
    assert inc.root_cause_candidates == ["silver_orders"]
    assert inc.severity == "critical"

    # Deep 4-hop ancestor-descendant pair (beyond default query_lineage depth=3) also converges.
    deep_incidents = aggregator.ingest(
        [
            _alert("a-raw", "raw_orders", 0),
            _alert("a-gmv", "fct_daily_gmv", 60),
        ]
    )
    assert len(deep_incidents) == 1
    assert deep_incidents[0].root_cause_candidates == ["raw_orders"]


def test_same_asset_burst_deduplicates_and_escalates_severity(
    aggregator: AlertStormAggregator,
) -> None:
    """Multiple alerts on the same asset (including dotted FQDN) merge into one single-asset Incident."""
    alerts = [
        _alert("a1", "fct_daily_gmv", 0, severity="low", summary="latency spike"),
        _alert("a2", "mart.main.fct_daily_gmv", 45, severity="critical", summary="gmv mismatch"),
        _alert("a3", "fct_daily_gmv", 120, severity="medium", summary="freshness sla"),
    ]

    incidents = aggregator.ingest(alerts)
    assert len(incidents) == 1
    inc = incidents[0]
    assert inc.alert_ids == ["a1", "a2", "a3"]
    assert inc.affected_assets == ["fct_daily_gmv"]
    assert inc.root_cause_candidates == ["fct_daily_gmv"]
    assert inc.shared_ancestors == []
    assert inc.severity == "critical"
    assert inc.to_incident_input().shared_ancestors == []


def test_time_window_boundary_exact_900s_vs_901s(
    aggregator: AlertStormAggregator,
) -> None:
    """Alerts <=900s apart merge; alerts >900s apart stay separate even when lineage-related."""
    merged = aggregator.ingest(
        [
            _alert("a-0", "fct_daily_gmv", 0),
            _alert("a-900", "dim_user_ltv", 900),
        ]
    )
    assert len(merged) == 1
    assert merged[0].alert_ids == ["a-0", "a-900"]

    separated = aggregator.ingest(
        [
            _alert("b-0", "fct_daily_gmv", 0),
            _alert("b-901", "dim_user_ltv", 901),
        ]
    )
    assert len(separated) == 2
    assert separated[0].alert_ids == ["b-0"]
    assert separated[0].root_cause_candidates == ["fct_daily_gmv"]
    assert separated[1].alert_ids == ["b-901"]
    assert separated[1].root_cause_candidates == ["dim_user_ltv"]


def test_unrelated_and_unknown_assets_stay_separate_and_safe(
    aggregator: AlertStormAggregator,
) -> None:
    """Unrelated lineage branches and unknown assets do not falsely merge or raise KeyError."""
    alerts = [
        _alert("a-orders", "silver_orders", 0, tz=False),
        _alert("a-fx", "silver_exchange_rates", 30, tz=True),
        _alert("a-ext-1", "external_vendor_feed", 40),
        _alert("a-ext-2", "external_vendor_feed", 80),
    ]

    incidents = aggregator.ingest(alerts)
    assert len(incidents) == 3

    assert incidents[0].affected_assets == ["silver_orders"]
    assert incidents[0].root_cause_candidates == ["silver_orders"]

    assert incidents[1].affected_assets == ["silver_exchange_rates"]
    assert incidents[1].root_cause_candidates == ["silver_exchange_rates"]

    assert incidents[2].affected_assets == ["external_vendor_feed"]
    assert incidents[2].alert_ids == ["a-ext-1", "a-ext-2"]
    assert incidents[2].root_cause_candidates == ["external_vendor_feed"]


def test_multi_root_cascade_surfaces_independent_upstream_roots(
    aggregator: AlertStormAggregator,
) -> None:
    """When two independent upstream failures cascade into a shared mart, both roots are surfaced."""
    alerts = [
        _alert("a-ord", "silver_orders", 0),
        _alert("a-fx", "silver_exchange_rates", 10),
        _alert("a-mart", "fct_daily_gmv", 20),
    ]

    incidents = aggregator.ingest(alerts)
    assert len(incidents) == 1
    assert incidents[0].root_cause_candidates == ["silver_orders", "silver_exchange_rates"]


def test_dispatch_runs_deduplicates_and_executes_end_to_end_via_harness_loop(
    aggregator: AlertStormAggregator,
    lineage_adapter: LineageAdapter,
    tmp_path: Path,
) -> None:
    """Alert storm spawns exactly one HarnessLoop Run per Incident using FakeReplayModelAdapter."""
    # 4 alerts from one silent-upstream/cascade storm + 1 later unrelated alert (>900s later)
    storm_alerts = [
        _alert("storm-1", "fct_daily_gmv", 0, summary="GMV reconciliation mismatch"),
        _alert("storm-2", "dim_user_ltv", 45, summary="LTV total mismatch"),
        _alert("storm-3", "stg_orders", 90, summary="Duplicate order_id in staging"),
        _alert("storm-4", "fct_daily_gmv", 150, summary="Repeated reconciliation failure"),
        _alert("later-fx", "silver_exchange_rates", 1200, summary="FX feed delayed"),
    ]

    incidents = aggregator.ingest(storm_alerts)
    assert len(incidents) == 2
    assert incidents[0].alert_ids == ["storm-1", "storm-2", "storm-3", "storm-4"]
    assert incidents[0].root_cause_candidates == ["stg_orders"]
    assert incidents[1].alert_ids == ["later-fx"]

    # 1. Verify single-storm Alert Storm dispatches strictly 1 Run through live HarnessLoop + ToolRegistry.
    single_storm_alerts = storm_alerts[:2]
    model = FakeReplayModelAdapter(
        [
            TurnResponse(
                content="Checking upstream lineage for converged sibling mart alerts.",
                tool_calls=[
                    ToolCall(
                        id="call_lin_1",
                        name="query_lineage",
                        arguments={
                            "table_name": "fct_daily_gmv",
                            "direction": "upstream",
                            "depth": 4,
                        },
                    )
                ],
            ),
            TurnResponse(
                content=(
                    "RCA Report: Silent upstream issue rooted in int_orders_usd / silver_orders "
                    "caused both fct_daily_gmv and dim_user_ltv alerts."
                )
            ),
        ]
    )
    registry = ToolRegistry(workspace_dir=WORKSPACE_DIR, lineage_adapter=lineage_adapter)
    loop = HarnessLoop(
        model_adapter=model,
        tool_registry=registry,
        runs_dir=tmp_path / "runs",
    )

    states = aggregator.dispatch_runs(
        single_storm_alerts,
        runner=loop,
        user_context=UserContext(user_id="oncall-de", role="de"),
        max_turns=6,
    )
    assert len(states) == 1
    run_state = states[0]
    assert run_state.status == "AWAITING_APPROVAL"
    assert run_state.turn == 2
    assert run_state.incident.affected_assets == ["fct_daily_gmv", "dim_user_ltv"]
    assert run_state.incident.shared_ancestors[0] == "int_orders_usd"
    assert len(run_state.incident.alerts) == 2
    assert "silver_orders" in run_state.messages[0].content
    tool_msgs = [m for m in run_state.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "int_orders_usd" in tool_msgs[0].content
    assert len(model.recorded_requests) == 2

    # 2. Verify callback runner and duplicate AggregatedIncident dedup across 5 raw alerts -> 2 Runs.
    dispatched: list[IncidentInput] = []

    def fake_runner(incident: IncidentInput, ctx: UserContext) -> RunState:
        dispatched.append(incident)
        return RunState(
            run_id=f"run-{incident.incident_id}",
            incident=incident,
            user_context=ctx,
            status="DIAGNOSED",
        )

    all_states = aggregator.dispatch_runs(
        storm_alerts,
        runner=fake_runner,
        user_context=UserContext(user_id="analyst-1", role="da"),
    )
    assert len(all_states) == 2
    assert len(dispatched) == 2
    assert all(s.user_context.resolution_mode == "diagnosis_only" for s in all_states)

    # Duplicate AggregatedIncident entries in a batch must not spawn duplicate runs.
    dup_incidents: list[AggregatedIncident] = [incidents[0], incidents[0], incidents[1]]
    dispatched.clear()
    dedup_states = aggregator.dispatch_runs(dup_incidents, runner=fake_runner)
    assert len(dedup_states) == 2
    assert [s.incident.incident_id for s in dedup_states] == [
        incidents[0].incident_id,
        incidents[1].incident_id,
    ]
