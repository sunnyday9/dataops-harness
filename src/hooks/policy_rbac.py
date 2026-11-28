from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

from sqlglot import exp

from src.core.context import RunState, ToolCall
from src.core.policy import PoliciesConfig, RolePolicy, load_policies, normalize_sql_dialect
from src.core.worktree import extract_raw_patch_paths
from src.hooks.base import PreToolDecision, PreToolUseHook
from src.hooks.sql_guard import (
    extract_referenced_tables,
    is_blocked_escape_func,
    is_explain_analyze,
    is_explain_statement,
    parse_single_statement,
    unwrap_explain,
)

_SPARK_LOG_SEGMENTS = frozenset(
    {
        "spark_events",
        "spark-events",
        "event_logs",
        "event-logs",
        "event_log",
        "event-log",
        "spark_event_logs",
        "spark-event-logs",
    }
)
_SPARK_LOG_FILENAME_RE = re.compile(r"(?:spark[-_]?events?|event[-_]?logs?)", re.IGNORECASE)


def _normalize_patch_entry(entry: str) -> str:
    return entry.strip().replace("\\", "/").lstrip("./")


def _patch_target_allowed(rel_path: str, allow_patch_paths: list[str]) -> bool:
    for raw_entry in allow_patch_paths:
        entry = _normalize_patch_entry(raw_entry)
        if not entry:
            continue
        if entry.endswith("/"):
            if rel_path.startswith(entry):
                return True
        elif rel_path == entry or rel_path.startswith(f"{entry}/"):
            return True
    return False


def _is_direct_agg(agg: exp.AggFunc, select_expr: exp.Select) -> bool:
    parent = agg.parent
    while parent is not None and parent is not select_expr:
        if isinstance(parent, (exp.Window, exp.Select, exp.Subquery, exp.CTE)):
            return False
        parent = parent.parent
    return parent is select_expr


def _is_direct_star(star: exp.Star, select_expr: exp.Select) -> bool:
    parent = star.parent
    while parent is not None and parent is not select_expr:
        if isinstance(parent, (exp.AggFunc, exp.Select, exp.Subquery, exp.CTE)):
            return False
        parent = parent.parent
    return parent is select_expr


def _is_inside_direct_agg_or_subquery(col: exp.Column, select_expr: exp.Select) -> bool:
    parent = col.parent
    while parent is not None and parent is not select_expr:
        if isinstance(parent, (exp.Select, exp.Subquery, exp.CTE)):
            return True
        if isinstance(parent, exp.AggFunc) and _is_direct_agg(parent, select_expr):
            return True
        parent = parent.parent
    return False


def _col_signature(col: exp.Column) -> str:
    return col.name.lower() if col.name else ""


_DISALLOWED_AGG_MODE_FUNCS = frozenset(
    {
        "list",
        "array_agg",
        "array_unique_agg",
        "collect_list",
        "collect_set",
        "string_agg",
        "group_concat",
        "first",
        "last",
        "any_value",
        "arg_min",
        "arg_max",
        "arg_min_null",
        "arg_max_null",
        "min_by",
        "max_by",
        "histogram",
        "map",
        "map_agg",
        "object_agg",
        "struct_pack",
        "json_group_array",
        "json_group_object",
        "json_agg",
        "jsonb_agg",
        "json_object_agg",
        "jsonb_object_agg",
        "j_s_o_n_array_agg",
        "j_s_o_n_object_agg",
        "xmlagg",
    }
)


def _iter_branch_selects(expr: exp.Expression | None) -> list[exp.Select]:
    if isinstance(expr, exp.Select):
        return [expr]
    if isinstance(expr, exp.SetOperation):
        return [*_iter_branch_selects(expr.this), *_iter_branch_selects(expr.expression)]
    return []


def _extract_alias_col_names(container: exp.Expression) -> list[str]:
    alias_node = container.args.get("alias")
    if isinstance(alias_node, exp.TableAlias):
        return [col.name.lower() for col in (alias_node.columns or []) if col.name]
    return []


def _collect_tainted_pii_aliases(stmt: exp.Expression, pii_cols: set[str]) -> set[str]:
    """Propagate PII taint through CTE chains, header column aliases, and derived subqueries."""
    tainted = set(pii_cols)
    targets: list[tuple[list[exp.Select], list[str]]] = []

    for cte in stmt.find_all(exp.CTE):
        targets.append((_iter_branch_selects(cte.this), _extract_alias_col_names(cte)))
    for sub in stmt.find_all(exp.Subquery):
        targets.append((_iter_branch_selects(sub.this), _extract_alias_col_names(sub)))

    seen_select_ids = {id(sel) for sels, _cols in targets for sel in sels}
    extra_selects: list[exp.Select] = []
    for node in [*stmt.find_all(exp.CTE), *stmt.find_all(exp.Subquery)]:
        for sel in node.find_all(exp.Select):
            if id(sel) not in seen_select_ids:
                seen_select_ids.add(id(sel))
                extra_selects.append(sel)
    if extra_selects:
        targets.append((extra_selects, []))

    changed = True
    while changed:
        changed = False
        for selects, col_aliases in targets:
            for sel in selects:
                for idx, proj in enumerate(sel.expressions):
                    referenced = {
                        _col_signature(col) for col in proj.find_all(exp.Column) if col.name
                    }
                    if referenced & tainted:
                        candidates: list[str] = []
                        if proj.alias:
                            candidates.append(proj.alias.lower())
                        elif isinstance(proj, exp.Column):
                            candidates.append(_col_signature(proj))
                        if idx < len(col_aliases):
                            candidates.append(col_aliases[idx])
                        for out_name in candidates:
                            if out_name and out_name not in tainted:
                                tainted.add(out_name)
                                changed = True
    return tainted


class PolicyRbacHook(PreToolUseHook):
    """Principal-role RBAC enforcement across patches, Spark event logs, and SQL scopes."""

    def __init__(
        self,
        policies: PoliciesConfig | None = None,
        workspace_dir: Path | None = None,
        *,
        dialect: str | None = None,
    ) -> None:
        self.policies = policies or load_policies()
        self.workspace_dir = workspace_dir.resolve() if workspace_dir is not None else None
        self.dialect = normalize_sql_dialect(dialect or self.policies.sql_guard.sql_dialect)

    def _resolve_relative_path(self, raw_path: str, *, allow_dot: bool = False) -> str:
        cleaned = raw_path.strip()
        if not cleaned:
            raise PermissionError("Empty path is not permitted.")

        slash_normalized = cleaned.replace("\\", "/")
        candidate = Path(cleaned)
        if candidate.is_absolute():
            if self.workspace_dir is None:
                raise PermissionError(f"Absolute path outside workspace is not allowed: {raw_path!r}")
            resolved = candidate.resolve()
            if not resolved.is_relative_to(self.workspace_dir):
                raise PermissionError(f"Path escapes workspace/: {raw_path!r}")
            rel = resolved.relative_to(self.workspace_dir).as_posix()
            return "." if rel == "." else rel

        norm = posixpath.normpath(slash_normalized)
        if norm == ".." or norm.startswith("../"):
            raise PermissionError(f"Path traversal outside workspace/ is forbidden: {raw_path!r}")
        if norm == ".":
            if allow_dot:
                return "."
            raise PermissionError("Target path must reference a concrete file inside workspace/.")
        return norm.lstrip("./")

    @staticmethod
    def _targets_spark_event_log(raw_path: str, rel_path: str | None) -> bool:
        candidates = [raw_path.replace("\\", "/")]
        if rel_path is not None:
            candidates.append(rel_path)

        for cand in candidates:
            parts = [p.lower() for p in cand.split("/") if p and p != "."]
            if any(part in _SPARK_LOG_SEGMENTS for part in parts):
                return True
            if "logs" in parts:
                idx = parts.index("logs")
                trailing = parts[idx + 1 :]
                if any("spark" in seg or "event" in seg for seg in trailing):
                    return True
                if not trailing:
                    return True
            if any(_SPARK_LOG_FILENAME_RE.search(part) for part in parts):
                return True
        return False

    def _check_read_or_grep(
        self,
        role: str,
        role_policy: RolePolicy,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PreToolDecision:
        raw_path = str(arguments.get("path") or ("." if tool_name == "grep_search" else ""))
        allow_dot = tool_name == "grep_search"

        # Read-only skills/*.md playbooks are readable by all roles via `read_file`.
        slash_rel = raw_path.strip().replace("\\", "/").lstrip("./")
        if tool_name == "read_file" and (
            slash_rel == "skills" or slash_rel.startswith("skills/")
        ):
            norm_skill = posixpath.normpath(slash_rel)
            if norm_skill != "skills" and not norm_skill.startswith("skills/"):
                return PreToolDecision(
                    action="block", reason=f"Invalid skills path traversal: {raw_path!r}"
                )
            if self.workspace_dir is not None:
                ws_resolved = self.workspace_dir.resolve()
                resolved_skill = (ws_resolved / norm_skill).resolve()
                if not resolved_skill.is_relative_to(ws_resolved):
                    return PreToolDecision(
                        action="block",
                        reason=f"Invalid skills symlink escape: {raw_path!r}",
                    )
                skill_rel = (
                    "."
                    if resolved_skill == ws_resolved
                    else resolved_skill.relative_to(ws_resolved).as_posix()
                )
                if not role_policy.allow_read_spark_event_logs and self._targets_spark_event_log(
                    skill_rel, None
                ):
                    return PreToolDecision(
                        action="block",
                        reason=f"Role {role!r} is not allowed to read Spark event logs ({raw_path!r}).",
                    )
            return PreToolDecision(action="allow")

        if role_policy.allow_read_spark_event_logs:
            if self.workspace_dir is not None:
                try:
                    rel_allowed = self._resolve_relative_path(raw_path, allow_dot=allow_dot)
                except PermissionError:
                    rel_allowed = None
                if rel_allowed is not None:
                    ws_resolved = self.workspace_dir.resolve()
                    cand_allowed = (
                        ws_resolved if rel_allowed == "." else (ws_resolved / rel_allowed)
                    ).resolve()
                    if not cand_allowed.is_relative_to(ws_resolved):
                        return PreToolDecision(
                            action="block",
                            reason=f"Path escapes workspace via symlink: {raw_path!r}",
                        )
            return PreToolDecision(action="allow")

        rel_path: str | None = None
        if not Path(raw_path.strip()).is_absolute():
            try:
                rel_path = self._resolve_relative_path(raw_path, allow_dot=allow_dot)
            except PermissionError as exc:
                return PreToolDecision(action="block", reason=str(exc))
        elif self.workspace_dir is not None:
            try:
                rel_path = self._resolve_relative_path(raw_path, allow_dot=allow_dot)
            except PermissionError as exc:
                # Even an external path like /tmp/spark-events/... is reported clearly.
                if self._targets_spark_event_log(raw_path, None):
                    return PreToolDecision(
                        action="block",
                        reason=f"Role {role!r} is not allowed to read Spark event logs ({raw_path!r}).",
                    )
                return PreToolDecision(action="block", reason=str(exc))

        resolved_rel: str | None = None
        if self.workspace_dir is not None and rel_path is not None:
            ws_resolved = self.workspace_dir.resolve()
            candidate = (ws_resolved if rel_path == "." else (ws_resolved / rel_path)).resolve()
            if not candidate.is_relative_to(ws_resolved):
                if self._targets_spark_event_log(candidate.as_posix(), None):
                    return PreToolDecision(
                        action="block",
                        reason=f"Role {role!r} is not allowed to read Spark event logs ({raw_path!r}).",
                    )
                return PreToolDecision(
                    action="block",
                    reason=f"Path escapes workspace via symlink: {raw_path!r}",
                )
            resolved_rel = (
                "."
                if candidate == ws_resolved
                else candidate.relative_to(ws_resolved).as_posix()
            )

        raw_check = (
            rel_path
            if Path(raw_path.strip()).is_absolute() and rel_path is not None
            else raw_path
        )
        if self._targets_spark_event_log(raw_check, rel_path) or (
            resolved_rel is not None and self._targets_spark_event_log(resolved_rel, None)
        ):
            return PreToolDecision(
                action="block",
                reason=f"Role {role!r} is not allowed to read Spark event logs ({raw_path!r}).",
            )
        return PreToolDecision(action="allow")

    def _extract_patch_targets(self, patch_text: str) -> list[str]:
        raw_paths = extract_raw_patch_paths(patch_text)
        targets: list[str] = []
        for raw_target in raw_paths:
            rel = self._resolve_relative_path(raw_target, allow_dot=False)
            if self.workspace_dir is not None:
                ws_resolved = self.workspace_dir.resolve()
                resolved_target = (ws_resolved / rel).resolve()
                if not resolved_target.is_relative_to(ws_resolved):
                    raise PermissionError(
                        f"Patch target escapes workspace via symlink: {raw_target!r}"
                    )
            if rel == ".git" or rel.startswith(".git/"):
                raise PermissionError(f"Patching .git/ is forbidden: {raw_target!r}")
            if rel not in targets:
                targets.append(rel)
        return targets

    def _check_git_inspect(
        self, role: str, role_policy: RolePolicy, arguments: dict[str, Any]
    ) -> PreToolDecision:
        extra_args = [str(a) for a in (arguments.get("args") or [])]
        for arg in extra_args:
            lowered = arg.lower()
            if (
                arg in {"-c", "-C", "-o", "--no-index"}
                or lowered.startswith(("-o", "--output", "--no-index", "--git-dir", "--work-tree"))
            ):
                return PreToolDecision(
                    action="block", reason=f"Disallowed git_inspect flag: {arg!r}"
                )
            if arg.startswith(("/", "~")) or arg == ".." or "../" in arg or "/.." in arg or "..\\" in arg:
                return PreToolDecision(
                    action="block",
                    reason=f"git_inspect path outside workspace is forbidden: {arg!r}",
                )
            if not role_policy.allow_read_spark_event_logs and self._targets_spark_event_log(
                arg, None
            ):
                return PreToolDecision(
                    action="block",
                    reason=f"Role {role!r} cannot inspect Spark event logs via git ({arg!r}).",
                )
        return PreToolDecision(action="allow")

    def _check_apply_patch(
        self,
        state: RunState,
        role: str,
        role_policy: RolePolicy,
        arguments: dict[str, Any],
    ) -> PreToolDecision:
        effective_mode = state.user_context.resolution_mode or role_policy.resolution_mode
        if role_policy.resolution_mode == "diagnosis_only" or effective_mode == "diagnosis_only":
            return PreToolDecision(
                action="block",
                reason=(
                    f"Role {role!r} runs in 'diagnosis_only' mode and cannot invoke apply_patch."
                ),
            )
        if not role_policy.allow_patch_paths:
            return PreToolDecision(
                action="block",
                reason=f"Role {role!r} has no writable patch paths configured.",
            )

        patch_text = str(arguments.get("patch") or "")
        if not patch_text.strip():
            return PreToolDecision(action="block", reason="Patch text must not be empty.")

        try:
            targets = self._extract_patch_targets(patch_text)
        except (PermissionError, ValueError) as exc:
            return PreToolDecision(action="block", reason=str(exc))

        for rel in targets:
            if not _patch_target_allowed(rel, role_policy.allow_patch_paths):
                return PreToolDecision(
                    action="block",
                    reason=(
                        f"Role {role!r} may only patch {role_policy.allow_patch_paths}, "
                        f"cannot patch {rel!r}."
                    ),
                )
        return PreToolDecision(action="allow")

    def _check_aggregate_only_stmt(
        self, role: str, stmt: exp.Expression, pii_cols: set[str], *, dialect: str = "duckdb"
    ) -> str | None:
        if isinstance(stmt, exp.Describe):
            return None
        if is_explain_statement(stmt):
            try:
                inner = unwrap_explain(stmt, dialect=dialect)
            except ValueError as exc:
                return str(exc)
            if is_explain_analyze(stmt):
                return self._check_aggregate_only_stmt(role, inner, pii_cols, dialect=dialect)
            if not isinstance(inner, (exp.Select, exp.SetOperation, exp.Describe)):
                return f"Role {role!r} cannot EXPLAIN non-query statement."
            return None

        tainted_cols = _collect_tainted_pii_aliases(stmt, pii_cols)

        if isinstance(stmt, exp.SetOperation):
            if reason := self._check_aggregate_only_stmt(
                role, stmt.this, tainted_cols, dialect=dialect
            ):
                return reason
            return self._check_aggregate_only_stmt(
                role, stmt.expression, tainted_cols, dialect=dialect
            )

        if not isinstance(stmt, exp.Select):
            return (
                f"Role {role!r} is restricted to aggregate or metadata-only SQL "
                f"(disallowed: {type(stmt).__name__.upper()})."
            )

        order_node = stmt.args.get("order")

        # Validate scalar/comparison subqueries in projections, HAVING, or ORDER BY.
        for container in [*stmt.expressions, stmt.args.get("having"), order_node]:
            if container is None:
                continue
            for sub in container.find_all(exp.Subquery):
                inner_q = sub.this
                if (
                    isinstance(inner_q, (exp.Select, exp.SetOperation))
                    and (
                        reason := self._check_aggregate_only_stmt(
                            role, inner_q, tainted_cols, dialect=dialect
                        )
                    )
                ):
                    return reason

        # Reject row/value-returning aggregates and PII inside non-COUNT aggregates across SELECT, HAVING, ORDER BY.
        for container in [*stmt.expressions, stmt.args.get("having"), order_node]:
            if container is None:
                continue
            for fn_node in container.find_all(exp.Func):
                fn = fn_node.name.lower() if isinstance(fn_node, exp.Anonymous) else fn_node.sql_name().lower()
                if fn in _DISALLOWED_AGG_MODE_FUNCS:
                    return (
                        f"Role {role!r} cannot use row/value-extracting function {fn!r} "
                        "in aggregate_or_metadata_only mode."
                    )
                if isinstance(fn_node, exp.AggFunc) and not isinstance(fn_node, exp.Count):
                    for col in fn_node.find_all(exp.Column):
                        sig = _col_signature(col)
                        if sig in tainted_cols:
                            return (
                                f"Role {role!r} cannot extract PII column {sig!r} via "
                                f"{fn.upper()} in aggregate_or_metadata_only mode."
                            )

        if order_node is not None:
            for ordered_item in order_node.expressions:
                inner_order = ordered_item.this
                for col in inner_order.find_all(exp.Column):
                    col_name = _col_signature(col)
                    if col_name in tainted_cols:
                        return (
                            f"Role {role!r} cannot ORDER BY PII column {col_name!r} "
                            "in aggregate_or_metadata_only mode."
                        )

        group_node = stmt.args.get("group")
        group_cols: set[str] = set()
        if group_node is not None:
            for g_expr in group_node.expressions:
                if isinstance(g_expr, exp.Literal) and not g_expr.is_string:
                    try:
                        pos = int(g_expr.this) - 1
                    except ValueError:
                        pos = -1
                    if 0 <= pos < len(stmt.expressions):
                        target_proj = stmt.expressions[pos]
                        for col in target_proj.find_all(exp.Column):
                            c_name = _col_signature(col)
                            if c_name in tainted_cols:
                                return (
                                    f"Role {role!r} cannot GROUP BY PII column "
                                    f"{c_name!r} in aggregate_or_metadata_only mode."
                                )
                            if c_name:
                                group_cols.add(c_name)
                        if target_proj.alias:
                            group_cols.add(target_proj.alias.lower())
                for col in g_expr.find_all(exp.Column):
                    col_name = _col_signature(col)
                    if col_name in tainted_cols:
                        return (
                            f"Role {role!r} cannot GROUP BY PII column {col_name!r} "
                            "in aggregate_or_metadata_only mode."
                        )
                    if col_name:
                        group_cols.add(col_name)

        if not stmt.args.get("from_") and not stmt.args.get("joins"):
            return None

        has_direct_agg = False
        for expr in stmt.expressions:
            for node in expr.walk():
                if isinstance(node, exp.Star) and _is_direct_star(node, stmt):
                    return (
                        f"Role {role!r} cannot select raw rows (`*`) in "
                        "aggregate_or_metadata_only mode."
                    )
                if isinstance(node, exp.AggFunc) and _is_direct_agg(node, stmt):
                    has_direct_agg = True
                if (
                    isinstance(node, exp.Column)
                    and not _is_inside_direct_agg_or_subquery(node, stmt)
                ):
                    col_name = _col_signature(node)
                    if col_name in tainted_cols:
                        return (
                            f"Role {role!r} cannot project PII column {col_name!r} "
                            "in aggregate_or_metadata_only mode."
                        )
                    if col_name not in group_cols:
                        return (
                            f"Role {role!r} cannot select unaggregated column {col_name!r} "
                            "in aggregate_or_metadata_only mode."
                        )

        if not has_direct_agg:
            return (
                f"Role {role!r} requires an aggregate function (COUNT/SUM/AVG/MIN/MAX) "
                "or EXPLAIN/DESCRIBE in aggregate_or_metadata_only mode."
            )
        return None

    def _check_sql(
        self, role: str, role_policy: RolePolicy, arguments: dict[str, Any]
    ) -> PreToolDecision:
        sql = str(arguments.get("sql") or "")
        engine_arg = str(arguments.get("engine") or "").strip().lower()
        effective_dialect = "spark" if engine_arg == "spark" else self.dialect
        try:
            stmt = parse_single_statement(sql, dialect=effective_dialect)
        except ValueError as exc:
            return PreToolDecision(action="block", reason=str(exc))

        if role_policy.sql_scope == "mart_only":
            # Prevent file-scan functions or direct path reads bypassing allowed Mart table scope.
            try:
                inspect_stmt = (
                    unwrap_explain(stmt, dialect=effective_dialect)
                    if is_explain_statement(stmt)
                    else stmt
                )
            except ValueError as exc:
                return PreToolDecision(action="block", reason=str(exc))

            for node in inspect_stmt.walk():
                if isinstance(node, exp.Table) and isinstance(node.this, exp.Literal):
                    return PreToolDecision(
                        action="block",
                        reason=f"Role {role!r} with sql_scope='mart_only' cannot scan raw files.",
                    )
                if isinstance(node, exp.Func):
                    fn = node.name.lower() if isinstance(node, exp.Anonymous) else node.sql_name().lower()
                    if (
                        fn.startswith(("read_", "parquet_"))
                        or fn.endswith("_scan")
                        or is_blocked_escape_func(fn)
                    ):
                        return PreToolDecision(
                            action="block",
                            reason=(
                                f"Role {role!r} with sql_scope='mart_only' cannot call "
                                f"restricted function {fn!r}."
                            ),
                        )

            allowed_tables = {t.lower() for t in role_policy.allowed_tables}
            tables = extract_referenced_tables(stmt, dialect=effective_dialect)
            disallowed = [
                ".".join(part for part in (cat, db, tbl) if part)
                for cat, db, tbl in tables
                if tbl not in allowed_tables
                and ".".join(part for part in (db, tbl) if part) not in allowed_tables
                and ".".join(part for part in (cat, db, tbl) if part) not in allowed_tables
            ]
            if disallowed:
                return PreToolDecision(
                    action="block",
                    reason=(
                        f"Role {role!r} with sql_scope='mart_only' may only query "
                        f"{sorted(allowed_tables)}; blocked: {disallowed}."
                    ),
                )

        if role_policy.sql_mode == "aggregate_or_metadata_only":
            pii_cols = {c.lower() for c in self.policies.pii_columns}
            if reason := self._check_aggregate_only_stmt(
                role, stmt, pii_cols, dialect=effective_dialect
            ):
                return PreToolDecision(action="block", reason=reason)

        return PreToolDecision(action="allow")

    def on_pre_tool_use(self, state: RunState, tool_call: ToolCall) -> PreToolDecision:
        """Enforce principal-role RBAC rules before executing a tool."""
        role = state.user_context.role
        role_policy = self.policies.roles.get(role)
        if role_policy is None:
            return PreToolDecision(
                action="block", reason=f"Unknown principal role: {role!r}"
            )

        if tool_call.name in {"read_file", "grep_search"}:
            return self._check_read_or_grep(
                role, role_policy, tool_call.name, tool_call.arguments
            )
        if tool_call.name == "apply_patch":
            return self._check_apply_patch(state, role, role_policy, tool_call.arguments)
        if tool_call.name == "git_inspect":
            return self._check_git_inspect(role, role_policy, tool_call.arguments)
        if tool_call.name == "run_sql_readonly":
            return self._check_sql(role, role_policy, tool_call.arguments)
        return PreToolDecision(action="allow")
