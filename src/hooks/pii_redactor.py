from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from typing import Any, Literal

from src.core.context import RunState, ToolCall, ToolResult
from src.core.policy import PoliciesConfig, load_policies
from src.hooks.base import PostToolDecision, PostToolUseHook

PiiCategory = Literal["EMAIL", "PHONE", "ID"]

_KEY_METADATA_KEY = "pii_hmac_key"

_ALREADY_REDACTED_EXACT_RE = re.compile(r"^<(EMAIL|PHONE|ID)_[0-9a-f]{8}>$")
_PLACEHOLDER_SPLIT_RE = re.compile(r"(<(?:EMAIL|PHONE|ID)_[0-9a-f]{8}>)")

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9._%+-])"
)
# Match 18-digit PRC resident IDs and XXX-XX-XXXX SSNs BEFORE phone numbers.
_PRC_ID_RE = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")

_CN_MOBILE_RE = re.compile(
    r"(?<!\d)(?:\+?86[-.\s]?)?1[3-9]\d(?:\d{8}|[-.\s]\d{4}[-.\s]\d{4})(?!\d)"
)
_INTL_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)[-.\s]?|\d{3}[-.\s])\d{3}[-.\s]\d{4}(?![A-Za-z0-9_.-])"
)


def _classify_column_category(column_name: str) -> PiiCategory:
    lowered = column_name.lower()
    if "email" in lowered or "mail" in lowered:
        return "EMAIL"
    if "phone" in lowered or "mobile" in lowered or "tel" in lowered:
        return "PHONE"
    return "ID"


def _canonicalize_value(category: PiiCategory, raw_value: str) -> str:
    if category == "EMAIL":
        return raw_value.strip().lower()
    if category == "PHONE":
        digits = re.sub(r"\D+", "", raw_value)
        return digits if digits else raw_value.strip()
    return raw_value.strip().upper()


class PiiRedactorHook(PostToolUseHook):
    """Two-layer (column-tag + regex) Run-scoped Keyed Pseudonym PII redactor."""

    def __init__(self, policies: PoliciesConfig | None = None) -> None:
        self.policies = policies or load_policies()
        self._pii_columns = {c.lower() for c in self.policies.pii_columns}

    @staticmethod
    def _get_or_create_run_key(state: RunState) -> bytes:
        existing = state.metadata.get(_KEY_METADATA_KEY)
        if isinstance(existing, str) and existing:
            return existing.encode("utf-8")
        key_hex = secrets.token_hex(32)
        state.metadata[_KEY_METADATA_KEY] = key_hex
        return key_hex.encode("utf-8")

    def pseudonymize(
        self, state: RunState, category: PiiCategory, raw_value: str
    ) -> str:
        """Compute deterministic Run-scoped HMAC-SHA256 pseudonym token."""
        key_bytes = self._get_or_create_run_key(state)
        canonical = _canonicalize_value(category, raw_value)
        digest = hmac.new(
            key_bytes, f"{category}:{canonical}".encode(), hashlib.sha256
        ).hexdigest()[:8]
        return f"<{category}_{digest}>"

    def _redact_plain_segment(self, state: RunState, text: str) -> str:
        out = _EMAIL_RE.sub(
            lambda m: self.pseudonymize(state, "EMAIL", m.group(0)), text
        )
        out = _PRC_ID_RE.sub(
            lambda m: self.pseudonymize(state, "ID", m.group(0)), out
        )
        out = _SSN_RE.sub(
            lambda m: self.pseudonymize(state, "ID", m.group(0)), out
        )
        out = _CN_MOBILE_RE.sub(
            lambda m: self.pseudonymize(state, "PHONE", m.group(0)), out
        )
        out = _INTL_PHONE_RE.sub(
            lambda m: self.pseudonymize(state, "PHONE", m.group(0)), out
        )
        return out

    def redact_text(self, state: RunState, text: str) -> str:
        """Apply Layer-2 regex PII sweep while keeping existing pseudonyms idempotent."""
        if not text:
            return text
        parts = _PLACEHOLDER_SPLIT_RE.split(text)
        redacted_parts = [
            part
            if _ALREADY_REDACTED_EXACT_RE.match(part)
            else self._redact_plain_segment(state, part)
            for part in parts
        ]
        return "".join(redacted_parts)

    def _redact_structured_sql_json(
        self, state: RunState, raw_output: str
    ) -> str | None:
        stripped = raw_output.lstrip()
        if not stripped.startswith("{"):
            return None
        try:
            payload: Any = json.loads(raw_output)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        columns = payload.get("columns")
        rows = payload.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return None

        changed = False
        col_categories: dict[int, PiiCategory] = {
            idx: _classify_column_category(str(col_name))
            for idx, col_name in enumerate(columns)
            if str(col_name).lower() in self._pii_columns
        }

        if isinstance(payload.get("sql"), str):
            redacted_sql = self.redact_text(state, payload["sql"])
            if redacted_sql != payload["sql"]:
                payload["sql"] = redacted_sql
                changed = True

        new_rows: list[Any] = []
        for row in rows:
            if not isinstance(row, list):
                new_rows.append(row)
                continue
            updated_row = list(row)
            for col_idx, cell in enumerate(row):
                if cell is None:
                    continue
                if col_idx in col_categories:
                    cell_text = str(cell)
                    if cell_text.strip() and not _ALREADY_REDACTED_EXACT_RE.match(
                        cell_text.strip()
                    ):
                        pseudonym = self.pseudonymize(
                            state, col_categories[col_idx], cell_text
                        )
                        if pseudonym != cell:
                            updated_row[col_idx] = pseudonym
                            changed = True
                elif isinstance(cell, str):
                    redacted_cell = self.redact_text(state, cell)
                    if redacted_cell != cell:
                        updated_row[col_idx] = redacted_cell
                        changed = True
            new_rows.append(updated_row)

        if not changed:
            return raw_output
        payload["rows"] = new_rows
        return json.dumps(payload, indent=2, ensure_ascii=False)

    def on_post_tool_use(
        self, state: RunState, tool_call: ToolCall, result: ToolResult
    ) -> PostToolDecision:
        """Scrub structured SQL columns (Layer 1) and free-text outputs (Layer 2)."""
        raw_output = result.output
        if not raw_output:
            return PostToolDecision(action="allow")

        structured = self._redact_structured_sql_json(state, raw_output)
        redacted = structured if structured is not None else self.redact_text(state, raw_output)

        if redacted != raw_output:
            return PostToolDecision(action="modify", modified_output=redacted)
        return PostToolDecision(action="allow")
