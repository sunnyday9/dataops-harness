from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.adapters.base import LakehousePaths
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.lineage_adapter import LineageAdapter
from src.adapters.model_adapter import (
    AnthropicMessagesAdapter,
    FakeReplayModelAdapter,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    PricingConfig,
    ProviderConfig,
    ToolDefinition,
    TurnRequest,
    TurnResponse,
    build_model_adapter,
    load_model_configs,
)
from src.core.context import (
    CheckpointStore,
    IncidentInput,
    Message,
    RunState,
    ToolCall,
    ToolResult,
    UsageMetrics,
    UserContext,
)
from src.core.loop import HarnessLoop
from src.hooks.base import (
    HookPipeline,
    PostToolDecision,
    PostToolUseHook,
    PreToolDecision,
    PreToolUseHook,
    StopDecision,
    StopHook,
)
from src.tools.registry import ToolRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = REPO_ROOT / "workspace"


def test_anthropic_openai_chat_and_responses_wire_normalization() -> None:
    """All three real API adapters normalize requests, tool calls, and cached usage identically."""
    tools = [
        ToolDefinition(
            name="read_file",
            description="Read workspace file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )
    ]
    req = TurnRequest(
        static_system_prompt="STATIC_PREFIX",
        messages=[
            Message(role="user", content="Investigate order mismatch"),
            Message(
                role="assistant",
                content="Checking file",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "spark_jobs/build_silver_orders.py"},
                    )
                ],
            ),
            Message(
                role="tool",
                tool_call_id="call_1",
                tool_name="read_file",
                content="1: # code",
            ),
        ],
        tools=tools,
        dynamic_tail_note="[Run State] turn=2/20",
    )

    # 1. Anthropic Messages
    anth_cfg = ProviderConfig(
        api_protocol="anthropic_messages",
        model="claude-sonnet-4-20250514",
        pricing_per_1m_tokens=PricingConfig(
            input=3.0, cached_input=0.3, cache_creation=3.75, output=15.0
        ),
    )
    anth = AnthropicMessagesAdapter(anth_cfg)
    anth_payload = anth.build_payload(req)
    assert anth_payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert anth_payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert anth_payload["messages"][-1]["role"] == "user"
    assert anth_payload["messages"][-1]["content"][-1]["text"] == "[Run State] turn=2/20"
    anth_resp = anth.parse_response(
        {
            "content": [
                {"type": "text", "text": "Running lineage"},
                {
                    "type": "tool_use",
                    "id": "call_2",
                    "name": "query_lineage",
                    "input": {"table_name": "fct_daily_gmv"},
                },
            ],
            "usage": {
                "input_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 100,
                "output_tokens": 50,
            },
            "stop_reason": "tool_use",
        }
    )
    assert anth_resp.tool_calls[0].name == "query_lineage"
    assert anth_resp.usage.input_tokens == 200
    assert anth_resp.usage.cached_input_tokens == 800
    assert anth_resp.usage.cache_hit_rate == pytest.approx(0.8)
    assert anth_resp.usage.cost_usd > 0

    # 2. OpenAI Chat Completions
    chat_cfg = ProviderConfig(
        api_protocol="openai_chat",
        model="gpt-4o",
        pricing_per_1m_tokens=PricingConfig(input=2.5, cached_input=1.25, output=10.0),
    )
    chat = OpenAIChatAdapter(chat_cfg)
    chat_payload = chat.build_payload(req)
    assert chat_payload["messages"][0] == {"role": "system", "content": "STATIC_PREFIX"}
    assert chat_payload["messages"][-1] == {
        "role": "user",
        "content": "[Run State] turn=2/20",
    }
    chat_resp = chat.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "Running lineage",
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "query_lineage",
                                    "arguments": '{"table_name": "fct_daily_gmv"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens": 50,
            },
        }
    )
    assert chat_resp.tool_calls == anth_resp.tool_calls
    assert chat_resp.usage.input_tokens == 200
    assert chat_resp.usage.cached_input_tokens == 800
    assert chat_resp.usage.cost_usd > 0

    # 3. OpenAI Responses (stateless local state: store=False, no previous_response_id)
    resp_cfg = ProviderConfig(
        api_protocol="openai_responses",
        model="gpt-4.1",
        store=False,
        pricing_per_1m_tokens=PricingConfig(input=2.0, cached_input=0.5, output=8.0),
    )
    responses = OpenAIResponsesAdapter(resp_cfg)
    resp_payload = responses.build_payload(req)
    assert resp_payload["store"] is False
    assert "previous_response_id" not in resp_payload
    assert resp_payload["instructions"] == "STATIC_PREFIX"
    parsed_resp = responses.parse_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Running lineage"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "query_lineage",
                    "arguments": '{"table_name": "fct_daily_gmv"}',
                },
            ],
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 800},
                "output_tokens": 50,
            },
        }
    )
    assert parsed_resp.tool_calls == anth_resp.tool_calls
    assert parsed_resp.usage.input_tokens == 200
    assert parsed_resp.usage.cached_input_tokens == 800
    assert parsed_resp.usage.cost_usd > 0


def test_model_config_loads_all_four_providers() -> None:
    """configs/models.yaml loads cleanly and constructs FakeReplayModelAdapter."""
    default_name, providers = load_model_configs(REPO_ROOT / "configs" / "models.yaml")
    assert default_name == "fake_replay"
    assert set(providers.keys()) == {
        "anthropic_messages",
        "openai_chat",
        "openai_responses",
        "fake_replay",
    }
    adapter = build_model_adapter(
        replay_responses=[TurnResponse(content="ok", tool_calls=[])]
    )
    assert isinstance(adapter, FakeReplayModelAdapter)


def test_six_orthogonal_tools_boundaries_and_jailing(tmp_path: Path) -> None:
    """Strictly 6 orthogonal tools exist and respect workspace/skills jailing."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "spark_jobs").mkdir()
    (sandbox / "spark_jobs" / "job.py").write_text(
        "line1\nERROR: skew detected\nline3\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-b", "main"], cwd=sandbox, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=sandbox, check=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=sandbox, check=True)
    subprocess.run(["git", "add", "."], cwd=sandbox, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sandbox, check=True)

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "PLAYBOOK.md").write_text("# Static Playbook\n", encoding="utf-8")

    live_paths = LakehousePaths.from_workspace(WORKSPACE_DIR)
    registry = ToolRegistry(
        workspace_dir=sandbox,
        skills_dir=skills_dir,
        lineage_adapter=LineageAdapter.from_workspace(WORKSPACE_DIR),
        duckdb_adapter=DuckDBAdapter(live_paths),
    )

    names = [d.name for d in registry.definitions()]
    assert names == [
        "read_file",
        "grep_search",
        "query_lineage",
        "run_sql_readonly",
        "git_inspect",
        "apply_patch",
    ]

    # read_file reads workspace and read-only skills/, blocks escapes
    ok_read = registry.execute(
        ToolCall(id="1", name="read_file", arguments={"path": "skills/PLAYBOOK.md"})
    )
    assert not ok_read.is_error and "Static Playbook" in ok_read.output
    esc_read = registry.execute(
        ToolCall(id="2", name="read_file", arguments={"path": "../outside.txt"})
    )
    assert esc_read.blocked is True

    # grep_search
    grep_res = registry.execute(
        ToolCall(id="3", name="grep_search", arguments={"pattern": "ERROR"})
    )
    assert "spark_jobs/job.py:2:" in grep_res.output

    # query_lineage & run_sql_readonly (DuckDB)
    lin_res = registry.execute(
        ToolCall(
            id="4",
            name="query_lineage",
            arguments={"table_name": "fct_daily_gmv", "direction": "upstream", "depth": 2},
        )
    )
    assert "int_orders_usd" in json.loads(lin_res.output)["nodes"]

    sql_res = registry.execute(
        ToolCall(
            id="5",
            name="run_sql_readonly",
            arguments={"sql": "SELECT COUNT(*) AS cnt FROM fct_daily_gmv", "engine": "duckdb"},
        )
    )
    assert json.loads(sql_res.output)["rows"] == [[7]]

    # git_inspect allows log, blocks commit
    assert not registry.execute(
        ToolCall(id="6", name="git_inspect", arguments={"subcommand": "log", "args": ["-n", "1"]})
    ).is_error
    assert registry.execute(
        ToolCall(id="7", name="git_inspect", arguments={"subcommand": "commit"})
    ).blocked is True

    # apply_patch succeeds inside sandbox, blocks skills/
    patch = (
        "--- a/spark_jobs/job.py\n"
        "+++ b/spark_jobs/job.py\n"
        "@@ -1,3 +1,3 @@\n"
        " line1\n"
        "-ERROR: skew detected\n"
        "+# fixed\n"
        " line3\n"
    )
    patch_res = registry.execute(
        ToolCall(id="8", name="apply_patch", arguments={"patch": patch})
    )
    assert not patch_res.is_error
    assert patch_res.metadata["patched_files"] == ["spark_jobs/job.py"]
    assert "# fixed" in (sandbox / "spark_jobs" / "job.py").read_text(encoding="utf-8")

    bad_patch = (
        "--- a/../skills/PLAYBOOK.md\n"
        "+++ b/../skills/PLAYBOOK.md\n"
        "@@ -1 +1 @@\n"
        "-# Static Playbook\n"
        "+# Hacked\n"
    )
    assert registry.execute(
        ToolCall(id="9", name="apply_patch", arguments={"patch": bad_patch})
    ).blocked is True


def test_single_loop_hooks_intercept_modify_reject_stop_and_abort(tmp_path: Path) -> None:
    """Hooks block/modify tool calls, transform outputs, reject premature stop, and abort."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text("SECRET_ROW\n", encoding="utf-8")
    executed_reads: list[str] = []

    class RecordingPreHook(PreToolUseHook):
        def on_pre_tool_use(self, state: RunState, tool_call: ToolCall) -> PreToolDecision:
            if tool_call.arguments.get("path") == "forbidden.txt":
                return PreToolDecision(action="block", reason="RBAC blocked")
            if tool_call.name == "read_file":
                return PreToolDecision(
                    action="modify",
                    modified_arguments={"path": "notes.txt", "start_line": 1, "end_line": 5},
                )
            return PreToolDecision(action="allow")

    class RedactingPostHook(PostToolUseHook):
        def on_post_tool_use(
            self, state: RunState, tool_call: ToolCall, result: ToolResult
        ) -> PostToolDecision:
            if "SECRET_ROW" in result.output:
                executed_reads.append(result.output)
                return PostToolDecision(
                    action="modify",
                    modified_output=result.output.replace("SECRET_ROW", "<REDACTED>"),
                )
            if tool_call.name == "grep_search":
                return PostToolDecision(
                    action="allow",
                    abort_status="ABORTED_THRASH",
                    abort_reason="Repeated grep loop",
                )
            return PostToolDecision(action="allow")

    class RequireTwoTurnStopHook(StopHook):
        def on_stop(self, state: RunState, candidate_output: str) -> StopDecision:
            if "EVIDENCE" not in candidate_output:
                return StopDecision(
                    allow_stop=False,
                    feedback_message="Missing EVIDENCE section; run a check and retry.",
                )
            return StopDecision(allow_stop=True)

    registry = ToolRegistry(workspace_dir=ws)
    incident = IncidentInput(
        incident_id="INC-1",
        title="GMV doubled",
        summary="fct_daily_gmv failed reconciliation",
    )

    # 1. Fix mode with PreTool block + PreTool modify + PostTool redact + Stop reject -> AWAITING_APPROVAL
    model = FakeReplayModelAdapter(
        [
            TurnResponse(
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "forbidden.txt"}),
                    ToolCall(id="c2", name="read_file", arguments={"path": "other.txt"}),
                ]
            ),
            TurnResponse(content="Premature summary without proof."),
            TurnResponse(content="RCA complete. EVIDENCE: notes verified."),
        ]
    )
    loop = HarnessLoop(
        model_adapter=model,
        tool_registry=registry,
        hooks=HookPipeline(
            pre_tool_hooks=[RecordingPreHook()],
            post_tool_hooks=[RedactingPostHook()],
            stop_hooks=[RequireTwoTurnStopHook()],
        ),
        runs_dir=tmp_path / "runs",
    )
    state = loop.run(incident, UserContext(role="de"), run_id="run-fix")
    assert state.status == "AWAITING_APPROVAL"
    assert state.turn == 3
    assert len(executed_reads) == 1  # forbidden.txt was never executed
    tool_msgs = [m for m in state.messages if m.role == "tool"]
    assert "RBAC blocked" in tool_msgs[0].content
    assert "<REDACTED>" in tool_msgs[1].content
    assert "SECRET_ROW" not in tool_msgs[1].content
    assert model.recorded_requests[0].static_system_prompt == model.recorded_requests[2].static_system_prompt
    assert model.recorded_requests[0].tools == model.recorded_requests[2].tools
    assert state.usage.cache_hit_rate > 0

    # 2. Diagnosis-Only mode for `da` -> DIAGNOSED
    da_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [TurnResponse(content="EVIDENCE: mart analysis")]
        ),
        tool_registry=registry,
        hooks=HookPipeline(stop_hooks=[RequireTwoTurnStopHook()]),
        runs_dir=tmp_path / "runs",
    )
    da_state = da_loop.run(incident, UserContext(role="da"), run_id="run-da")
    assert da_state.user_context.resolution_mode == "diagnosis_only"
    assert da_state.status == "DIAGNOSED"

    # 3. PostTool abort -> ABORTED_THRASH
    thrash_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [TurnResponse(tool_calls=[ToolCall(id="c3", name="grep_search", arguments={"pattern": "x"})])]
        ),
        tool_registry=registry,
        hooks=HookPipeline(post_tool_hooks=[RedactingPostHook()]),
        runs_dir=tmp_path / "runs",
    )
    thrash_state = thrash_loop.run(incident, UserContext(role="de"), run_id="run-thrash")
    assert thrash_state.status == "ABORTED_THRASH"


def test_append_only_checkpoint_and_crash_safe_resume(tmp_path: Path) -> None:
    """Interrupted Run recovers from state.jsonl (ignoring torn trailing JSON) and completes."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("hello world\n", encoding="utf-8")
    runs_dir = tmp_path / "runs"
    registry = ToolRegistry(workspace_dir=ws)
    incident = IncidentInput(incident_id="INC-2", title="Crash test", summary="Simulate crash")

    # Run 1 exhausts max_turns=1 after executing 1 tool -> stops with ABORTED_BUDGET
    first_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [
                TurnResponse(
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a.txt"})],
                    usage=UsageMetrics(input_tokens=100, cached_input_tokens=0, output_tokens=20, cost_usd=0.01),
                )
            ]
        ),
        tool_registry=registry,
        runs_dir=runs_dir,
    )
    paused = first_loop.run(incident, UserContext(role="de"), run_id="run-crash", max_turns=1)
    assert paused.status == "ABORTED_BUDGET"
    assert paused.turn == 1

    # Simulate an abrupt mid-write crash appending a truncated JSON line
    store = CheckpointStore(runs_dir, "run-crash")
    with store.state_path.open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 999, "broken": ')

    # Fresh HarnessLoop instance resumes losslessly from checkpoint
    resumed_loop = HarnessLoop(
        model_adapter=FakeReplayModelAdapter(
            [
                TurnResponse(
                    content="Resumed RCA complete.",
                    usage=UsageMetrics(input_tokens=20, cached_input_tokens=100, output_tokens=30, cost_usd=0.02),
                )
            ]
        ),
        tool_registry=registry,
        runs_dir=runs_dir,
    )
    resumed = resumed_loop.resume(
        "run-crash",
        extra_message="Human feedback: continue with extra turns.",
        extra_turns=2,
    )
    assert resumed.status == "AWAITING_APPROVAL"
    assert resumed.turn == 2
    assert resumed.usage.input_tokens == 120
    assert resumed.usage.cached_input_tokens == 100
    assert resumed.usage.cost_usd == pytest.approx(0.03)
    assert any("hello world" in m.content for m in resumed.messages if m.role == "tool")
    assert resumed.final_output == "Resumed RCA complete."
