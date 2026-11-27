from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.adapters.model_adapter import (
    AnthropicMessagesAdapter,
    FakeReplayModelAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    ProviderConfig,
    TurnResponse,
)
from src.core.context import (
    IncidentInput,
    Message,
    RunState,
    ToolCall,
    ToolResult,
    UsageMetrics,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.core.policy import AntiThrashPolicy, LimitsPolicy, load_policies
from src.hooks.anti_thrash import ANTI_THRASH_METADATA_KEY, AntiThrashHook
from src.hooks.base import HookPipeline
from src.hooks.compactor import FOLDED_MARKER, ContextCompactorHook
from src.tools.registry import ToolRegistry


def _make_state(
    *,
    run_id: str = "run-test-1",
    turn: int = 1,
    max_turns: int = 20,
    cost_usd: float = 0.0,
    rejection_count: int = 0,
    messages: list[Message] | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        incident=IncidentInput(
            incident_id="INC-100",
            title="GMV mismatch",
            summary="fct_daily_gmv doubled after backfill",
        ),
        user_context=UserContext(role="de"),
        turn=turn,
        max_turns=max_turns,
        rejection_count=rejection_count,
        usage=UsageMetrics(cost_usd=cost_usd),
        messages=messages or [Message(role="user", content="Investigate INC-100")],
    )


def test_compactor_spills_large_output_by_lines_and_chars(tmp_path: Path) -> None:
    """Outputs exceeding 120 lines or 4000 chars spill untruncated to .scratch/."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    compactor = ContextCompactorHook(workspace_root=ws)
    state = _make_state(run_id="run-spill-1", turn=1)

    # 1. Normal output stays untouched.
    small = ToolResult(tool_call_id="c0", name="read_file", output="line1\nline2\n")
    small_decision = compactor.on_post_tool_use(
        state, ToolCall(id="c0", name="read_file", arguments={"path": "a.py"}), small
    )
    assert small_decision.action == "allow"
    assert small_decision.modified_output is None

    # 2. >120 lines triggers spill with full file content preserved.
    many_lines = "\n".join(f"row_{idx:03d}: value={idx * 10}" for idx in range(1, 151))
    res_lines = ToolResult(tool_call_id="call_lines", name="grep_search", output=many_lines)
    dec_lines = compactor.on_post_tool_use(
        state, ToolCall(id="call_lines", name="grep_search", arguments={"pattern": "row_"}), res_lines
    )
    assert dec_lines.action == "modify"
    assert dec_lines.modified_output is not None

    expected_spill = ws / ".scratch" / "run-spill-1" / "call_lines.txt"
    assert expected_spill.is_file()
    assert expected_spill.read_text(encoding="utf-8") == many_lines
    assert "150 lines" in dec_lines.modified_output
    assert f"{len(many_lines)} chars" in dec_lines.modified_output
    assert ".scratch/run-spill-1/call_lines.txt" in dec_lines.modified_output
    assert "read_file(path=" in dec_lines.modified_output
    assert "row_001" in dec_lines.modified_output
    assert "row_150" in dec_lines.modified_output
    assert len(dec_lines.modified_output.splitlines()) <= compactor.limits.compactor_max_lines
    assert len(dec_lines.modified_output) <= compactor.limits.compactor_max_chars

    # 3. Single >4000-char line also spills and stays bounded.
    huge_line = ("A" * 2500) + ("M" * 1500) + ("Z" * 2500)
    res_chars = ToolResult(tool_call_id="call_chars", name="run_sql_readonly", output=huge_line)
    dec_chars = compactor.on_post_tool_use(
        state,
        ToolCall(id="call_chars", name="run_sql_readonly", arguments={"sql": "SELECT 1"}),
        res_chars,
    )
    assert dec_chars.action == "modify"
    assert dec_chars.modified_output is not None
    chars_spill = ws / ".scratch" / "run-spill-1" / "call_chars.txt"
    assert chars_spill.read_text(encoding="utf-8") == huge_line
    assert "1 lines" in dec_chars.modified_output
    assert f"{len(huge_line)} chars" in dec_chars.modified_output
    assert "AAA" in dec_chars.modified_output
    assert "ZZZ" in dec_chars.modified_output
    assert len(dec_chars.modified_output) <= compactor.limits.compactor_max_chars


def test_compactor_sanitizes_spill_paths_against_traversal(tmp_path: Path) -> None:
    """Adversarial run_id / tool_call.id values cannot escape workspace/.scratch/."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    compactor = ContextCompactorHook(workspace_root=ws)
    state = _make_state(run_id="../../etc/evil", turn=1)
    big_output = "x" * 5000

    decision = compactor.on_post_tool_use(
        state,
        ToolCall(id="../../../passwd", name="read_file", arguments={"path": "a.py"}),
        ToolResult(tool_call_id="../../../passwd", name="read_file", output=big_output),
    )
    assert decision.action == "modify"
    assert decision.modified_output is not None

    scratch_root = (ws / ".scratch").resolve()
    spilled_files = list(scratch_root.rglob("*.txt"))
    assert len(spilled_files) == 1
    assert spilled_files[0].resolve().is_relative_to(scratch_root)
    assert spilled_files[0].read_text(encoding="utf-8") == big_output


def test_compactor_folds_older_tool_messages_preserving_prefix_and_idempotency(
    tmp_path: Path,
) -> None:
    """Older tool outputs fold while static prefix and recent turns stay byte-identical."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    compactor = ContextCompactorHook(workspace_root=ws, recent_turns_to_keep=2)

    messages: list[Message] = [
        Message(role="system", content="STATIC_SYSTEM_PROMPT"),
        Message(role="user", content="INCIDENT_INITIAL_PROMPT"),
    ]
    for idx in range(1, 12):
        tc = ToolCall(id=f"c{idx}", name="read_file", arguments={"path": f"f{idx}.py"})
        messages.append(Message(role="assistant", content=f"step {idx}", tool_calls=[tc]))
        if idx == 2:
            body = (
                "[Output spilled: 200 lines, 9000 chars -> .scratch/run-fold/c2.txt]\n"
                "Hint: read_file\n--- HEAD ---\n" + ("H" * 300)
            )
        else:
            body = f"turn {idx} detailed diagnostic output " + ("D" * 350)
        messages.append(
            Message(role="tool", tool_call_id=f"c{idx}", tool_name="read_file", content=body)
        )

    prefix_snapshot = [m.model_copy(deep=True) for m in messages[:2]]
    recent_snapshot = [m.model_copy(deep=True) for m in messages[-4:]]
    cost_messages = [m.model_copy(deep=True) for m in messages[:10]]  # system, user, 4 turns

    # At turn=10 (<= compactor_fold_after_turns=10 and ratio 0.5 < 0.65), no folding occurs.
    pre_state = _make_state(turn=10, messages=[m.model_copy(deep=True) for m in messages])
    compactor.on_post_tool_use(
        pre_state,
        ToolCall(id="c10", name="read_file", arguments={"path": "f10.py"}),
        ToolResult(tool_call_id="c10", name="read_file", output="ok"),
    )
    assert all(
        not m.content.startswith(FOLDED_MARKER) for m in pre_state.messages if m.role == "tool"
    )

    # At turn=11 (> 10), older tool messages fold while prefix & last 2 turns stay identical.
    state = _make_state(turn=11, messages=[m.model_copy(deep=True) for m in messages])
    compactor.on_post_tool_use(
        state,
        ToolCall(id="c11", name="read_file", arguments={"path": "f11.py"}),
        ToolResult(tool_call_id="c11", name="read_file", output="ok"),
    )
    assert state.messages[:2] == prefix_snapshot
    assert state.messages[-4:] == recent_snapshot

    older_tools = [m for m in state.messages[2:-4] if m.role == "tool"]
    assert len(older_tools) == 9
    assert all(m.content.startswith(FOLDED_MARKER) for m in older_tools)
    assert ".scratch/run-fold/c2.txt" in older_tools[1].content
    assert state.metadata["compaction"]["compacted_messages"] == 9
    assert state.metadata["compaction"]["fold_runs"] == 1
    assert state.metadata["compaction"]["chars_saved"] > 2000

    # Idempotency: re-running compaction on the same state makes zero further edits.
    folded_snapshot = [m.model_copy(deep=True) for m in state.messages]
    compactor.on_post_tool_use(
        state,
        ToolCall(id="c11", name="read_file", arguments={"path": "f11.py"}),
        ToolResult(tool_call_id="c11", name="read_file", output="ok"),
    )
    assert state.messages == folded_snapshot
    assert state.metadata["compaction"]["compacted_messages"] == 9
    assert state.metadata["compaction"]["fold_runs"] == 1

    # Budget ratio trigger (>= 0.65 of max_cost_usd) also folds even on early turns.
    cost_state = _make_state(turn=4, cost_usd=1.30, messages=cost_messages)
    compactor.on_post_tool_use(
        cost_state,
        ToolCall(id="c4", name="read_file", arguments={"path": "f4.py"}),
        ToolResult(tool_call_id="c4", name="read_file", output="ok"),
    )
    cost_tool_msgs = [m for m in cost_state.messages if m.role == "tool"]
    assert cost_tool_msgs[0].content.startswith(FOLDED_MARKER)
    assert cost_tool_msgs[1].content.startswith(FOLDED_MARKER)
    assert not cost_tool_msgs[2].content.startswith(FOLDED_MARKER)
    assert not cost_tool_msgs[3].content.startswith(FOLDED_MARKER)


def test_compactor_hard_ceilings_abort_budget(tmp_path: Path) -> None:
    """Turn (with HiTL rejection extension), cost, and wall-clock ceilings abort cleanly."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    limits = LimitsPolicy(
        max_turns=5,
        max_wall_clock_seconds=100,
        max_cost_usd=1.0,
        rejection_turn_extension=3,
        max_rejections=2,
    )
    tc = ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})
    res = ToolResult(tool_call_id="c1", name="read_file", output="ok")

    # 1. Turn ceiling without rejections (turn 5 >= 5)
    compactor = ContextCompactorHook(workspace_root=ws, limits=limits)
    turn_state = _make_state(turn=5, max_turns=5)
    turn_dec = compactor.on_post_tool_use(turn_state, tc, res)
    assert turn_dec.abort_status == "ABORTED_BUDGET"
    assert "Turn ceiling reached" in turn_dec.abort_reason

    # 2. HiTL rejection extends effective_max_turns (5 + 1*3 = 8); capped at max_rejections=2 (11).
    extended_state = _make_state(turn=6, max_turns=5, rejection_count=1)
    assert compactor.on_post_tool_use(extended_state, tc, res).abort_status is None
    extended_state.turn = 8
    assert compactor.on_post_tool_use(extended_state, tc, res).abort_status == "ABORTED_BUDGET"

    capped_state = _make_state(turn=10, max_turns=5, rejection_count=5)
    assert compactor.on_post_tool_use(capped_state, tc, res).abort_status is None
    capped_state.turn = 11
    assert compactor.on_post_tool_use(capped_state, tc, res).abort_status == "ABORTED_BUDGET"

    # 3. Cost ceiling exceeded
    cost_state = _make_state(turn=1, max_turns=5, cost_usd=1.0)
    cost_dec = compactor.on_post_tool_use(cost_state, tc, res)
    assert cost_dec.abort_status == "ABORTED_BUDGET"
    assert "Cost ceiling" in cost_dec.abort_reason

    # 4. Wall-clock ceiling exceeded via advancing clock
    clock = [datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)]
    timed_compactor = ContextCompactorHook(
        workspace_root=ws, limits=limits, now_fn=lambda: clock[0]
    )
    wall_state = _make_state(turn=1, max_turns=5)
    assert timed_compactor.on_post_tool_use(wall_state, tc, res).abort_status is None
    clock[0] = clock[0] + timedelta(seconds=101)
    wall_dec = timed_compactor.on_post_tool_use(wall_state, tc, res)
    assert wall_dec.abort_status == "ABORTED_BUDGET"
    assert "Wall-clock budget exceeded" in wall_dec.abort_reason


def test_anti_thrash_repeated_identical_calls_two_strikes() -> None:
    """Identical (tool, normalized_args) fires Strike 1 nudge then Strike 2 ABORTED_THRASH."""
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()

    call_a = ToolCall(id="1", name="read_file", arguments={"path": "x.py", "start_line": 1})
    call_a_reordered = ToolCall(
        id="2", name="read_file", arguments={"start_line": 1, "path": "x.py"}
    )
    call_a_third = ToolCall(id="3", name="read_file", arguments={"path": "x.py", "start_line": 1})
    ok = ToolResult(tool_call_id="1", name="read_file", output="1: code")

    d1 = hook.on_post_tool_use(state, call_a, ok)
    assert d1.abort_status is None and not d1.injected_messages

    d2 = hook.on_post_tool_use(state, call_a_reordered, ok)
    assert d2.abort_status is None
    assert len(d2.injected_messages) == 1
    assert "Strike 1/2" in d2.injected_messages[0].content
    assert "Repeated identical tool call" in d2.injected_messages[0].content
    assert state.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 1

    d3 = hook.on_post_tool_use(state, call_a_third, ok)
    assert d3.abort_status == "ABORTED_THRASH"
    assert "strike 2/2" in d3.abort_reason.lower()


def test_anti_thrash_error_stagnation_and_recovery() -> None:
    """Consecutive errors trigger Strike 1; success resets streak; subsequent streak aborts."""
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()

    def run_sql(idx: int, sql: str, *, is_error: bool, output: str) -> object:
        tc = ToolCall(id=f"c{idx}", name="run_sql_readonly", arguments={"sql": sql})
        res = ToolResult(tool_call_id=f"c{idx}", name="run_sql_readonly", output=output, is_error=is_error)
        return hook.on_post_tool_use(state, tc, res)

    d1 = run_sql(1, "SELECT bad1", is_error=True, output="err 1")
    assert d1.abort_status is None and not d1.injected_messages

    # 2nd consecutive error -> Strike 1 nudge
    d2 = run_sql(2, "SELECT bad2", is_error=True, output="err 2")
    assert d2.abort_status is None
    assert len(d2.injected_messages) == 1
    assert "Error stagnation" in d2.injected_messages[0].content

    # Intervening success resets consecutive error streak, so 1 later error does NOT trip Strike 2.
    d3 = run_sql(3, "SELECT 1", is_error=False, output="[[1]]")
    assert d3.abort_status is None and not d3.injected_messages
    d4 = run_sql(4, "SELECT bad3", is_error=True, output="err 3")
    assert d4.abort_status is None and not d4.injected_messages

    # 2nd consecutive error in new streak -> Strike 2 -> ABORTED_THRASH
    d5 = run_sql(5, "SELECT bad4", is_error=True, output="err 4")
    assert d5.abort_status == "ABORTED_THRASH"
    assert "Error stagnation" in d5.abort_reason


def test_anti_thrash_file_oscillation_on_apply_patch() -> None:
    """Ping-ponging distinct patches on the same file trips Strike 1 then ABORTED_THRASH."""
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()

    for idx in range(1, 5):
        patch_text = (
            "--- a/spark_jobs/job.py\n"
            "+++ b/spark_jobs/job.py\n"
            f"@@ -1 +1 @@\n-v{idx - 1}\n+v{idx}\n"
        )
        tc = ToolCall(id=f"p{idx}", name="apply_patch", arguments={"patch": patch_text})
        res = ToolResult(
            tool_call_id=f"p{idx}",
            name="apply_patch",
            output=f"Applied patch to: spark_jobs/job.py (v{idx})",
            metadata={"patched_files": ["spark_jobs/job.py"]},
        )
        dec = hook.on_post_tool_use(state, tc, res)
        if idx < 3:
            assert dec.abort_status is None and not dec.injected_messages
        elif idx == 3:
            assert dec.abort_status is None
            assert len(dec.injected_messages) == 1
            assert "File edit oscillation" in dec.injected_messages[0].content
            assert "spark_jobs/job.py" in dec.injected_messages[0].content
        else:
            assert dec.abort_status == "ABORTED_THRASH"
            assert "File edit oscillation" in dec.abort_reason


def test_anti_thrash_healthy_varied_investigation_no_false_positives() -> None:
    """A normal varied investigation (including post-patch file re-read) never trips strikes."""
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()

    steps = [
        (
            ToolCall(id="1", name="read_file", arguments={"path": "spark_jobs/job.py", "start_line": 1, "end_line": 50}),
            ToolResult(tool_call_id="1", name="read_file", output="lines 1-50"),
        ),
        (
            ToolCall(id="2", name="read_file", arguments={"path": "spark_jobs/job.py", "start_line": 51, "end_line": 100}),
            ToolResult(tool_call_id="2", name="read_file", output="lines 51-100"),
        ),
        (
            ToolCall(id="3", name="grep_search", arguments={"pattern": "join"}),
            ToolResult(tool_call_id="3", name="grep_search", output="spark_jobs/job.py:12: join"),
        ),
        (
            ToolCall(id="4", name="run_sql_readonly", arguments={"sql": "SELECT COUNT(*) FROM fct_daily_gmv"}),
            ToolResult(tool_call_id="4", name="run_sql_readonly", output="[[7]]"),
        ),
        (
            ToolCall(id="5", name="apply_patch", arguments={"patch": "--- a/spark_jobs/job.py\n+++ b/spark_jobs/job.py\n"}),
            ToolResult(tool_call_id="5", name="apply_patch", output="Applied", metadata={"patched_files": ["spark_jobs/job.py"]}),
        ),
        (
            ToolCall(id="6", name="read_file", arguments={"path": "spark_jobs/job.py", "start_line": 1, "end_line": 50}),
            ToolResult(tool_call_id="6", name="read_file", output="updated lines 1-50"),
        ),
    ]

    for tc, res in steps:
        dec = hook.on_post_tool_use(state, tc, res)
        assert dec.abort_status is None
        assert not dec.injected_messages
    assert state.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 0


def test_anti_thrash_allows_reread_after_failed_apply_patch() -> None:
    """Re-reading a target file after a failed (non-blocked) apply_patch resets its read signature."""
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()

    read_tc = ToolCall(id="r1", name="read_file", arguments={"path": "dbt_project/m.sql"})
    read_ok = ToolResult(tool_call_id="r1", name="read_file", output="SELECT 1")
    assert hook.on_post_tool_use(state, read_tc, read_ok).abort_status is None

    failed_patch_tc = ToolCall(
        id="p1",
        name="apply_patch",
        arguments={
            "patch": "--- dbt_project/m.sql\n+++ dbt_project/m.sql\n@@ -1 +1 @@\n-bad\n+good\n"
        },
    )
    failed_patch_res = ToolResult(
        tool_call_id="p1",
        name="apply_patch",
        output="git apply --check failed",
        is_error=True,
        blocked=False,
    )
    patch_dec = hook.on_post_tool_use(state, failed_patch_tc, failed_patch_res)
    assert patch_dec.abort_status is None and not patch_dec.injected_messages

    reread_tc = ToolCall(id="r2", name="read_file", arguments={"path": "dbt_project/m.sql"})
    reread_dec = hook.on_post_tool_use(state, reread_tc, read_ok)
    assert reread_dec.abort_status is None
    assert not reread_dec.injected_messages
    assert state.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 0


def test_e2e_harness_loop_compactor_spill_prefix_cache_and_wire_adapters(
    tmp_path: Path,
) -> None:
    """Full HarnessLoop + Compactor + AntiThrash spills large logs, preserves prefix cache, and stays wire-valid."""
    ws = tmp_path / "workspace"
    (ws / "logs").mkdir(parents=True)
    big_log = "\n".join(
        f"INFO line {idx:03d}: " + ("X" * 40) for idx in range(1, 161)
    )
    (ws / "logs" / "spark.log").write_text(big_log, encoding="utf-8")
    (ws / "notes.txt").write_text(
        "\n".join(f"note line {idx}: " + ("Y" * 100) for idx in range(1, 31)),
        encoding="utf-8",
    )

    registry = ToolRegistry(workspace_dir=ws)
    compactor = ContextCompactorHook(
        workspace_root=ws,
        limits=LimitsPolicy(
            max_turns=20,
            compactor_max_lines=120,
            compactor_max_chars=4000,
            compactor_fold_after_turns=10,
            compactor_fold_budget_ratio=0.65,
        ),
        recent_turns_to_keep=2,
    )
    anti_thrash = AntiThrashHook(policy=AntiThrashPolicy())

    # Turn 1 reads logs/spark.log (161 lines -> spilled); Turn 2 reads the spilled file back;
    # Turns 3..12 perform varied non-repeating reads across notes.txt; Turn 13 concludes.
    responses: list[TurnResponse] = [
        TurnResponse(
            tool_calls=[
                ToolCall(
                    id="call_big",
                    name="read_file",
                    arguments={"path": "logs/spark.log", "start_line": 1, "end_line": 200},
                )
            ]
        ),
        TurnResponse(
            tool_calls=[
                ToolCall(
                    id="call_read_spill",
                    name="read_file",
                    arguments={
                        "path": ".scratch/run-e2e-compact/call_big.txt",
                        "start_line": 1,
                        "end_line": 40,
                    },
                )
            ]
        ),
    ]
    for idx in range(3, 13):
        responses.append(
            TurnResponse(
                tool_calls=[
                    ToolCall(
                        id=f"call_t{idx}",
                        name="read_file",
                        arguments={"path": "notes.txt", "start_line": idx, "end_line": idx + 10},
                    )
                ]
            )
        )
    responses.append(TurnResponse(content="RCA complete with spilled evidence."))

    model = FakeReplayModelAdapter(responses)
    loop = HarnessLoop(
        model_adapter=model,
        tool_registry=registry,
        hooks=HookPipeline(post_tool_hooks=[compactor, anti_thrash]),
        runs_dir=tmp_path / "runs",
    )
    incident = IncidentInput(
        incident_id="INC-200",
        title="Spark skew & context bloat",
        summary="Investigate large log output across 13 turns",
    )
    state = loop.run(incident, UserContext(role="de"), run_id="run-e2e-compact")

    assert state.status == "AWAITING_APPROVAL"
    assert state.turn == 13
    assert state.usage.cache_hit_rate > 0

    # Verify spill file exists untruncated and Turn 2 read_file read real content from it.
    spill_file = ws / ".scratch" / "run-e2e-compact" / "call_big.txt"
    assert spill_file.is_file()
    assert spill_file.read_text(encoding="utf-8") == (
        "# logs/spark.log (lines 1-160 of 160)\n"
        + "\n".join(f"{idx}: INFO line {idx:03d}: " + ("X" * 40) for idx in range(1, 161))
    )
    assert "INFO line 001" in state.messages[4].content or FOLDED_MARKER in state.messages[4].content

    # Verify static prompt, tools, and initial prefix messages stayed byte-identical across all 13 requests.
    first_req = model.recorded_requests[0]
    for req in model.recorded_requests:
        assert req.static_system_prompt == first_req.static_system_prompt
        assert req.tools == first_req.tools
        assert req.messages[0] == first_req.messages[0]
        if len(req.messages) > 1 and len(model.recorded_requests) > 1:
            assert req.messages[1] == model.recorded_requests[1].messages[1]

    # Verify Turn 11 compacted older messages (indices 0..17), and Turn 12's prefix stayed identical in Turn 13.
    req_turn_12 = model.recorded_requests[11]
    req_turn_13 = model.recorded_requests[12]
    assert req_turn_12.messages[2].content.startswith(FOLDED_MARKER)
    assert ".scratch/run-e2e-compact/call_big.txt" in req_turn_12.messages[2].content
    assert req_turn_12.messages[:18] == req_turn_13.messages[:18]
    assert req_turn_13.messages[18].content.startswith(FOLDED_MARKER)
    # Most recent 2 turns (turn 11 and turn 12) stayed completely intact in Turn 13.
    assert not req_turn_13.messages[-1].content.startswith(FOLDED_MARKER)
    assert not req_turn_13.messages[-3].content.startswith(FOLDED_MARKER)

    # Verify final compacted TurnRequest still serializes cleanly for all 3 real LLM wire adapters.
    anth_payload = AnthropicMessagesAdapter(
        ProviderConfig(api_protocol="anthropic_messages", model="claude-sonnet-4-20250514")
    ).build_payload(req_turn_13)
    chat_payload = OpenAIChatAdapter(
        ProviderConfig(api_protocol="openai_chat", model="gpt-4o")
    ).build_payload(req_turn_13)
    resp_payload = OpenAIResponsesAdapter(
        ProviderConfig(api_protocol="openai_responses", model="gpt-4.1")
    ).build_payload(req_turn_13)
    assert len(anth_payload["messages"]) > 0
    assert len(chat_payload["messages"]) > 0
    assert len(resp_payload["input"]) > 0


def test_e2e_harness_loop_budget_abort_and_anti_thrash_checkpoint_resume(
    tmp_path: Path,
) -> None:
    """End-to-end budget abort and 2-strike AntiThrash across CheckpointStore pause/resume."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "a.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=ws, check=True, capture_output=True)
    registry = ToolRegistry(workspace_dir=ws)
    incident = IncidentInput(incident_id="INC-300", title="Budget & Thrash", summary="Test aborts")

    # 1. Cost ceiling abort through HarnessLoop + ContextCompactorHook
    budget_compactor = ContextCompactorHook(
        workspace_root=ws,
        limits=LimitsPolicy(max_cost_usd=0.50, max_turns=20),
    )
    budget_model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[ToolCall(id="b1", name="read_file", arguments={"path": "a.py"})],
                usage=UsageMetrics(input_tokens=100, output_tokens=20, cost_usd=0.25),
            ),
            TurnResponse(
                tool_calls=[ToolCall(id="b2", name="grep_search", arguments={"pattern": "ok"})],
                usage=UsageMetrics(input_tokens=100, output_tokens=20, cost_usd=0.25),
            ),
            TurnResponse(content="Should not reach turn 3"),
        ]
    )
    budget_loop = HarnessLoop(
        model_adapter=budget_model,
        tool_registry=registry,
        hooks=HookPipeline(post_tool_hooks=[budget_compactor]),
        runs_dir=tmp_path / "runs_budget",
    )
    budget_state = budget_loop.run(incident, UserContext(role="de"), run_id="run-budget")
    assert budget_state.status == "ABORTED_BUDGET"
    assert budget_state.turn == 2
    assert len(budget_model.recorded_requests) == 2

    # 2. AntiThrash Strike 1 -> max_turns pause -> resume with fresh AntiThrashHook -> Strike 2 ABORTED_THRASH
    runs_dir = tmp_path / "runs_thrash"
    thrash_model_1 = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[ToolCall(id="t1", name="grep_search", arguments={"pattern": "ok"})]
            ),
            TurnResponse(
                tool_calls=[ToolCall(id="t2", name="grep_search", arguments={"pattern": "ok"})]
            ),
        ]
    )
    loop_1 = HarnessLoop(
        model_adapter=thrash_model_1,
        tool_registry=registry,
        hooks=HookPipeline(post_tool_hooks=[AntiThrashHook(policy=AntiThrashPolicy())]),
        runs_dir=runs_dir,
    )
    paused = loop_1.run(incident, UserContext(role="de"), run_id="run-thrash-resume", max_turns=2)
    assert paused.status == "ABORTED_BUDGET"
    assert paused.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 1
    assert any(
        m.role == "user" and "ANTI-THRASH WARNING" in m.content for m in paused.messages
    )

    # Fresh HarnessLoop + fresh AntiThrashHook instance resumes from CheckpointStore
    thrash_model_2 = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[ToolCall(id="t3", name="grep_search", arguments={"pattern": "ok"})]
            )
        ]
    )
    loop_2 = HarnessLoop(
        model_adapter=thrash_model_2,
        tool_registry=registry,
        hooks=HookPipeline(post_tool_hooks=[AntiThrashHook(policy=AntiThrashPolicy())]),
        runs_dir=runs_dir,
    )
    resumed = loop_2.resume("run-thrash-resume", extra_turns=3)
    assert resumed.status == "ABORTED_THRASH"
    assert resumed.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 2
    assert "strike 2/2" in resumed.metadata["abort_reason"].lower()


def test_default_policy_loading_and_started_at_metadata_formats(tmp_path: Path) -> None:
    """Hooks load default policies from configs/policies.yaml and parse metadata started_at."""
    policies = load_policies()
    default_compactor = ContextCompactorHook(workspace_root=tmp_path)
    default_thrash = AntiThrashHook()
    assert default_compactor.limits == policies.limits
    assert default_thrash.policy == policies.anti_thrash

    policy_compactor = ContextCompactorHook(workspace_root=tmp_path, policies=policies)
    policy_thrash = AntiThrashHook(policies=policies)
    assert policy_compactor.limits == policies.limits
    assert policy_thrash.policy == policies.anti_thrash

    # Verify ISO timestamp string in state.metadata["started_at"] triggers wall-clock ceiling.
    state = _make_state(turn=1)
    state.metadata["started_at"] = (datetime.now(UTC) - timedelta(seconds=350)).isoformat()
    dec = default_compactor.on_post_tool_use(
        state,
        ToolCall(id="w1", name="read_file", arguments={"path": "a.py"}),
        ToolResult(tool_call_id="w1", name="read_file", output="ok"),
    )
    assert dec.abort_status == "ABORTED_BUDGET"


def test_anti_thrash_simultaneous_signals_count_single_strike_and_e2e_patch_oscillation(
    tmp_path: Path,
) -> None:
    """Multiple signals on one call count as 1 strike; real apply_patch churn aborts via HarnessLoop."""
    # 1. Identical call + consecutive error on the same step still gives Strike 1 nudge first.
    hook = AntiThrashHook(policy=AntiThrashPolicy())
    state = _make_state()
    tc = ToolCall(id="e1", name="run_sql_readonly", arguments={"sql": "SELECT bad"})
    err = ToolResult(tool_call_id="e1", name="run_sql_readonly", output="syntax err", is_error=True)
    assert hook.on_post_tool_use(state, tc, err).abort_status is None
    strike_one = hook.on_post_tool_use(state, tc, err)
    assert strike_one.abort_status is None
    assert len(strike_one.injected_messages) == 1
    assert state.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 1

    # 2. Full HarnessLoop + real ToolRegistry apply_patch oscillation on the same file
    ws = tmp_path / "patch_ws"
    (ws / "spark_jobs").mkdir(parents=True)
    (ws / "spark_jobs" / "job.py").write_text("line1\nval = 0\nline3\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=ws, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=ws, check=True, capture_output=True)

    patches = [
        f"--- a/spark_jobs/job.py\n+++ b/spark_jobs/job.py\n@@ -1,3 +1,3 @@\n line1\n-val = {i}\n+val = {i + 1}\n line3\n"
        for i in range(4)
    ]
    model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(id=f"patch_{i}", name="apply_patch", arguments={"patch": patch})
                ]
            )
            for i, patch in enumerate(patches)
        ]
    )
    loop = HarnessLoop(
        model_adapter=model,
        tool_registry=ToolRegistry(workspace_dir=ws),
        hooks=HookPipeline(post_tool_hooks=[AntiThrashHook()]),
        runs_dir=tmp_path / "runs_patch",
    )
    final_state = loop.run(
        IncidentInput(incident_id="INC-400", title="Patch churn", summary="Oscillation test"),
        UserContext(role="de"),
        run_id="run-patch-osc",
    )
    assert final_state.status == "ABORTED_THRASH"
    assert final_state.turn == 4
    assert "File edit oscillation" in final_state.metadata["abort_reason"]
    assert final_state.metadata[ANTI_THRASH_METADATA_KEY]["strikes"] == 2


def test_compactor_spill_preserves_middle_error_lines(tmp_path: Path) -> None:
    """Spilled oversized logs retain middle ERROR/Traceback lines inside the summary stub."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    compactor = ContextCompactorHook(workspace_root=ws)
    state = _make_state(run_id="run-spill-err", turn=1)

    log_lines = [f"2026-01-01 INFO step={i} ok" for i in range(1, 221)]
    log_lines[104] = "Traceback (most recent call last):"
    log_lines[105] = "ERROR: java.lang.OutOfMemoryError: Java heap space"
    log_lines[106] = "Caused by: executor container killed beyond 384m"
    raw_log = "\n".join(log_lines)

    dec = compactor.on_post_tool_use(
        state,
        ToolCall(id="call_oom", name="read_file", arguments={"path": "logs/spark.log"}),
        ToolResult(tool_call_id="call_oom", name="read_file", output=raw_log, is_error=True),
    )
    assert dec.action == "modify"
    assert dec.modified_output is not None
    assert ".scratch/run-spill-err/call_oom.txt" in dec.modified_output
    assert "KEY ERROR LINES" in dec.modified_output
    assert "Traceback (most recent call last):" in dec.modified_output
    assert "java.lang.OutOfMemoryError: Java heap space" in dec.modified_output
    assert "Caused by: executor container killed beyond 384m" in dec.modified_output
    assert len(dec.modified_output) <= compactor.limits.compactor_max_chars
    assert len(dec.modified_output.splitlines()) <= compactor.limits.compactor_max_lines
    assert (ws / ".scratch" / "run-spill-err" / "call_oom.txt").read_text(encoding="utf-8") == raw_log

