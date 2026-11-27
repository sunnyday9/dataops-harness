from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from src.core.context import Message, ToolCall, UsageMetrics

ApiProtocol = Literal[
    "anthropic_messages",
    "openai_chat",
    "openai_responses",
    "fake_replay",
]


class PricingConfig(BaseModel):
    """Per-million-token USD pricing for a model provider."""

    input: float = 0.0
    cached_input: float = 0.0
    cache_creation: float = 0.0
    output: float = 0.0


class ProviderConfig(BaseModel):
    """Configuration entry loaded from configs/models.yaml."""

    api_protocol: ApiProtocol
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    store: bool = False
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 3
    request_timeout_seconds: float = 120.0
    prompt_caching: dict[str, Any] = Field(default_factory=dict)
    pricing_per_1m_tokens: PricingConfig = Field(default_factory=PricingConfig)


class ToolDefinition(BaseModel):
    """JSON-schema tool contract sent identically across turns."""

    name: str
    description: str
    input_schema: dict[str, Any]


class TurnRequest(BaseModel):
    """Stateless turn request preserving an immutable static prefix."""

    static_system_prompt: str
    messages: list[Message]
    tools: list[ToolDefinition] = Field(default_factory=list)
    dynamic_tail_note: str = ""


class TurnResponse(BaseModel):
    """Normalized model response across all wire protocols."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: UsageMetrics = Field(default_factory=UsageMetrics)
    stop_reason: str = "end_turn"
    raw_provider: str = ""


def compute_usage_cost(
    *,
    input_tokens: int,
    cached_input_tokens: int,
    cache_creation_tokens: int,
    output_tokens: int,
    pricing: PricingConfig,
) -> float:
    """Compute USD cost from token counts and per-1M pricing."""
    cost = (
        (input_tokens / 1_000_000.0) * pricing.input
        + (cached_input_tokens / 1_000_000.0) * pricing.cached_input
        + (cache_creation_tokens / 1_000_000.0) * pricing.cache_creation
        + (output_tokens / 1_000_000.0) * pricing.output
    )
    return round(cost, 6)


def _append_tail_note_to_user_messages(
    messages: list[Message], tail_note: str
) -> list[Message]:
    if not tail_note:
        return list(messages)
    copied = [m.model_copy(deep=True) for m in messages]
    if copied and copied[-1].role == "user":
        sep = "\n\n" if copied[-1].content else ""
        copied[-1].content = f"{copied[-1].content}{sep}{tail_note}"
    else:
        copied.append(Message(role="user", content=tail_note))
    return copied


class ModelAdapter(ABC):
    """Stateless wire-API adapter protocol keeping all Run state local."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def _resolve_api_key(self, *, required: bool = False) -> str | None:
        if not self.config.api_key_env:
            return None
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if required and not api_key:
            raise ValueError(
                f"Missing required API key environment variable: {self.config.api_key_env}"
            )
        return api_key or None

    def _resolve_base_url(self) -> str | None:
        env_url = (
            os.environ.get(self.config.base_url_env, "").strip()
            if self.config.base_url_env
            else ""
        )
        return env_url or self.config.base_url

    @abstractmethod
    def complete(self, request: TurnRequest) -> TurnResponse:
        """Send a stateless turn request and return normalized output and usage."""


def _retry_delay_seconds(exc: Exception, attempt: int) -> float | None:
    """Return bounded exponential backoff seconds for transient HTTP/network failures."""
    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    cls_name = type(exc).__name__
    is_transient = (
        status_code in {429, 500, 502, 503, 504}
        or isinstance(exc, TimeoutError)
        or "Timeout" in cls_name
        or "Connection" in cls_name
    )
    if not is_transient:
        return None

    delay = float(3 * (2**attempt))
    headers = getattr(response, "headers", None) if response is not None else None
    if headers:
        retry_after = None
        for key, val in headers.items():
            if str(key).lower() == "retry-after":
                retry_after = val
                break
        if retry_after is not None:
            try:
                parsed = float(retry_after)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and parsed >= 0:
                delay = parsed
    return min(delay, 30.0)


class AnthropicMessagesAdapter(ModelAdapter):
    """Adapter for the Anthropic Messages API with explicit ephemeral cache_control."""

    def __init__(self, config: ProviderConfig, client: Any = None) -> None:
        super().__init__(config)
        self._client = client
        self.recorded_requests: list[TurnRequest] = []

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            api_key = self._resolve_api_key()
            base_url = self._resolve_base_url()
            self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        return self._client

    def build_payload(self, request: TurnRequest) -> dict[str, Any]:
        caching_on = bool(self.config.prompt_caching.get("enabled", True))
        system_block: dict[str, Any] = {
            "type": "text",
            "text": request.static_system_prompt,
        }
        if caching_on:
            system_block["cache_control"] = {"type": "ephemeral"}

        tools_payload: list[dict[str, Any]] = []
        for idx, tool in enumerate(request.tools):
            entry: dict[str, Any] = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            if caching_on and idx == len(request.tools) - 1:
                entry["cache_control"] = {"type": "ephemeral"}
            tools_payload.append(entry)

        wire_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        def flush_tool_results() -> None:
            if pending_tool_results:
                wire_messages.append(
                    {"role": "user", "content": list(pending_tool_results)}
                )
                pending_tool_results.clear()

        for msg in request.messages:
            if msg.role == "system":
                continue
            if msg.role == "tool":
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                )
                continue

            flush_tool_results()
            if msg.role == "user":
                wire_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": msg.content}],
                    }
                )
            elif msg.role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                if not blocks:
                    blocks.append({"type": "text", "text": ""})
                wire_messages.append({"role": "assistant", "content": blocks})

        flush_tool_results()
        if request.dynamic_tail_note:
            if wire_messages and wire_messages[-1]["role"] == "user":
                wire_messages[-1]["content"].append(
                    {"type": "text", "text": request.dynamic_tail_note}
                )
            else:
                wire_messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": request.dynamic_tail_note}],
                    }
                )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": [system_block],
            "messages": wire_messages,
        }
        if tools_payload:
            payload["tools"] = tools_payload
        return payload

    def parse_response(self, raw: dict[str, Any]) -> TurnResponse:
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in raw.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                texts.append(str(block.get("text") or ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=dict(block.get("input") or {}),
                    )
                )

        usage_raw = raw.get("usage") or {}
        uncached_in = int(usage_raw.get("input_tokens") or 0)
        cached_in = int(usage_raw.get("cache_read_input_tokens") or 0)
        cache_create = int(usage_raw.get("cache_creation_input_tokens") or 0)
        out_tokens = int(usage_raw.get("output_tokens") or 0)
        cost = compute_usage_cost(
            input_tokens=uncached_in,
            cached_input_tokens=cached_in,
            cache_creation_tokens=cache_create,
            output_tokens=out_tokens,
            pricing=self.config.pricing_per_1m_tokens,
        )
        return TurnResponse(
            content="\n".join(t for t in texts if t),
            tool_calls=tool_calls,
            usage=UsageMetrics(
                input_tokens=uncached_in,
                cached_input_tokens=cached_in,
                cache_creation_tokens=cache_create,
                output_tokens=out_tokens,
                cost_usd=cost,
            ),
            stop_reason=str(raw.get("stop_reason") or "end_turn"),
            raw_provider="anthropic_messages",
        )

    def complete(self, request: TurnRequest) -> TurnResponse:
        self.recorded_requests.append(request.model_copy(deep=True))
        payload = self.build_payload(request)
        resp = self._get_client().messages.create(**payload)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return self.parse_response(raw)


class OpenAIChatAdapter(ModelAdapter):
    """Adapter for OpenAI Chat Completions API using stateless prefix-first messages."""

    def __init__(self, config: ProviderConfig, client: Any = None) -> None:
        super().__init__(config)
        self._client = client
        self.recorded_requests: list[TurnRequest] = []

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = self._resolve_api_key(required=True)
            base_url = self._resolve_base_url()
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                default_headers=self.config.extra_headers or None,
                timeout=self.config.request_timeout_seconds,
            )
        return self._client

    def build_payload(self, request: TurnRequest) -> dict[str, Any]:
        wire_messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.static_system_prompt}
        ]
        effective_messages = _append_tail_note_to_user_messages(
            request.messages, request.dynamic_tail_note
        )
        for msg in effective_messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                wire_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(
                                    tc.arguments, ensure_ascii=False
                                ),
                            },
                            **(
                                {"extra_content": tc.extra_content}
                                if tc.extra_content is not None
                                else {}
                            ),
                        }
                        for tc in msg.tool_calls
                    ]
                wire_messages.append(entry)
            elif msg.role == "tool":
                wire_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }
                )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": wire_messages,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in request.tools
            ]
        return payload

    def parse_response(self, raw: dict[str, Any]) -> TurnResponse:
        choices = raw.get("choices") or [{}]
        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls: list[ToolCall] = []
        for idx, raw_tc in enumerate(message.get("tool_calls") or []):
            fn = raw_tc.get("function") or {}
            raw_args = fn.get("arguments")
            if isinstance(raw_args, dict):
                parsed_args = dict(raw_args)
            elif isinstance(raw_args, str) and raw_args.strip():
                try:
                    loaded = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {"_raw_arguments": raw_args}
                else:
                    parsed_args = (
                        loaded
                        if isinstance(loaded, dict)
                        else {"_raw_arguments": raw_args}
                    )
            else:
                parsed_args = {}
            tc_id = str(raw_tc.get("id") or "").strip() or f"tc-{idx}"
            raw_extra = raw_tc.get("extra_content")
            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    name=str(fn.get("name") or ""),
                    arguments=parsed_args,
                    extra_content=dict(raw_extra) if isinstance(raw_extra, dict) else None,
                )
            )

        usage_raw = raw.get("usage") or {}
        total_prompt = int(usage_raw.get("prompt_tokens") or 0)
        details = usage_raw.get("prompt_tokens_details") or {}
        cached_in = int(details.get("cached_tokens") or 0)
        uncached_in = max(total_prompt - cached_in, 0)
        out_tokens = int(usage_raw.get("completion_tokens") or 0)
        cost = compute_usage_cost(
            input_tokens=uncached_in,
            cached_input_tokens=cached_in,
            cache_creation_tokens=0,
            output_tokens=out_tokens,
            pricing=self.config.pricing_per_1m_tokens,
        )
        return TurnResponse(
            content=str(message.get("content") or ""),
            tool_calls=tool_calls,
            usage=UsageMetrics(
                input_tokens=uncached_in,
                cached_input_tokens=cached_in,
                output_tokens=out_tokens,
                cost_usd=cost,
            ),
            stop_reason=str(choice.get("finish_reason") or "stop"),
            raw_provider="openai_chat",
        )

    def complete(self, request: TurnRequest) -> TurnResponse:
        self.recorded_requests.append(request.model_copy(deep=True))
        payload = self.build_payload(request)
        client = self._get_client()
        max_retries = max(int(self.config.max_retries), 0)
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(**payload)
                break
            except Exception as exc:
                delay = _retry_delay_seconds(exc, attempt)
                if delay is None or attempt >= max_retries:
                    raise
                time.sleep(delay)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return self.parse_response(raw)


class OpenAIResponsesAdapter(ModelAdapter):
    """Stateless adapter for OpenAI Responses API (`store=False`, local state only)."""

    def __init__(self, config: ProviderConfig, client: Any = None) -> None:
        super().__init__(config)
        self._client = client
        self.recorded_requests: list[TurnRequest] = []

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            api_key = self._resolve_api_key()
            base_url = self._resolve_base_url()
            self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def build_payload(self, request: TurnRequest) -> dict[str, Any]:
        input_items: list[dict[str, Any]] = []
        effective_messages = _append_tail_note_to_user_messages(
            request.messages, request.dynamic_tail_note
        )
        for msg in effective_messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                input_items.append(
                    {"type": "message", "role": "user", "content": msg.content}
                )
            elif msg.role == "assistant":
                if msg.content:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": msg.content,
                        }
                    )
                for tc in msg.tool_calls:
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        }
                    )
            elif msg.role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id or "",
                        "output": msg.content,
                    }
                )

        payload: dict[str, Any] = {
            "model": self.config.model,
            "instructions": request.static_system_prompt,
            "input": input_items,
            "store": False,
            "max_output_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
                for t in request.tools
            ]
        return payload

    def parse_response(self, raw: dict[str, Any]) -> TurnResponse:
        texts: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in raw.get("output") or []:
            itype = item.get("type")
            if itype == "message":
                content = item.get("content")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") in ("output_text", "text"):
                            texts.append(str(part.get("text") or ""))
            elif itype == "function_call":
                raw_args = item.get("arguments") or "{}"
                parsed_args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else dict(raw_args)
                )
                tool_calls.append(
                    ToolCall(
                        id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=parsed_args,
                    )
                )

        usage_raw = raw.get("usage") or {}
        total_in = int(usage_raw.get("input_tokens") or 0)
        details = usage_raw.get("input_tokens_details") or {}
        cached_in = int(details.get("cached_tokens") or 0)
        uncached_in = max(total_in - cached_in, 0)
        out_tokens = int(usage_raw.get("output_tokens") or 0)
        cost = compute_usage_cost(
            input_tokens=uncached_in,
            cached_input_tokens=cached_in,
            cache_creation_tokens=0,
            output_tokens=out_tokens,
            pricing=self.config.pricing_per_1m_tokens,
        )
        return TurnResponse(
            content="\n".join(t for t in texts if t),
            tool_calls=tool_calls,
            usage=UsageMetrics(
                input_tokens=uncached_in,
                cached_input_tokens=cached_in,
                output_tokens=out_tokens,
                cost_usd=cost,
            ),
            stop_reason=str(raw.get("status") or "completed"),
            raw_provider="openai_responses",
        )

    def complete(self, request: TurnRequest) -> TurnResponse:
        self.recorded_requests.append(request.model_copy(deep=True))
        payload = self.build_payload(request)
        resp = self._get_client().responses.create(**payload)
        raw = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return self.parse_response(raw)


class FakeReplayModelAdapter(ModelAdapter):
    """Deterministic zero-token replay adapter for unit and Harness behavioral tests."""

    def __init__(
        self,
        responses: Sequence[TurnResponse] | Callable[[TurnRequest], TurnResponse],
        config: ProviderConfig | None = None,
    ) -> None:
        super().__init__(
            config
            or ProviderConfig(api_protocol="fake_replay", model="fake-replay-v1")
        )
        self._script = responses
        self._cursor = 0
        self.recorded_requests: list[TurnRequest] = []
        self._last_prefix: str | None = None

    def complete(self, request: TurnRequest) -> TurnResponse:
        self.recorded_requests.append(request.model_copy(deep=True))
        if callable(self._script):
            resp = self._script(request).model_copy(deep=True)
        else:
            if self._cursor >= len(self._script):
                raise IndexError(
                    f"FakeReplayModelAdapter exhausted after {self._cursor} turn(s)."
                )
            resp = self._script[self._cursor].model_copy(deep=True)
            self._cursor += 1

        if (
            resp.usage.input_tokens == 0
            and resp.usage.cached_input_tokens == 0
            and resp.usage.output_tokens == 0
        ):
            prefix_hit = (
                self._last_prefix is not None
                and self._last_prefix == request.static_system_prompt
            )
            prefix_tokens = max(len(request.static_system_prompt) // 4, 1)
            msg_tokens = sum(max(len(m.content) // 4, 1) for m in request.messages)
            resp.usage = UsageMetrics(
                input_tokens=msg_tokens if prefix_hit else prefix_tokens + msg_tokens,
                cached_input_tokens=prefix_tokens if prefix_hit else 0,
                output_tokens=max(len(resp.content) // 4, 10),
                cost_usd=0.0,
            )

        self._last_prefix = request.static_system_prompt
        if not resp.raw_provider:
            resp.raw_provider = "fake_replay"
        return resp


def load_model_configs(config_path: Path) -> tuple[str, dict[str, ProviderConfig]]:
    """Load default provider name and all ProviderConfig entries from YAML."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default_provider = str(raw.get("default_provider") or "fake_replay")
    providers: dict[str, ProviderConfig] = {}
    for name, entry in (raw.get("providers") or {}).items():
        providers[str(name)] = ProviderConfig.model_validate(entry)
    return default_provider, providers


_ADAPTER_CLASSES: dict[ApiProtocol, type[ModelAdapter]] = {
    "anthropic_messages": AnthropicMessagesAdapter,
    "openai_chat": OpenAIChatAdapter,
    "openai_responses": OpenAIResponsesAdapter,
}


def build_model_adapter(
    provider_name: str | None = None,
    config_path: Path | None = None,
    replay_responses: Sequence[TurnResponse]
    | Callable[[TurnRequest], TurnResponse]
    | None = None,
    client: Any = None,
) -> ModelAdapter:
    """Instantiate the configured ModelAdapter by provider name."""
    resolved_path = (
        config_path
        or Path(__file__).resolve().parent.parent.parent / "configs" / "models.yaml"
    )
    default_name, providers = load_model_configs(resolved_path)
    selected = provider_name or default_name
    if selected not in providers:
        raise KeyError(f"Unknown model provider {selected!r} in {resolved_path}")
    cfg = providers[selected]
    if cfg.api_protocol == "fake_replay":
        return FakeReplayModelAdapter(responses=replay_responses or [], config=cfg)
    adapter_cls = _ADAPTER_CLASSES[cfg.api_protocol]
    return adapter_cls(config=cfg, client=client)
