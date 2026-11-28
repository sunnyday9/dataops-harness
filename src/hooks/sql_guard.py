from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot.optimizer.scope import Scope, build_scope

from src.core.context import RunState, ToolCall
from src.core.policy import PoliciesConfig, load_policies, normalize_sql_dialect
from src.hooks.base import PreToolDecision, PreToolUseHook

_EXPLAIN_PREFIX_RE = re.compile(
    r"^(?:\([^)]*\)\s*|(?:ANALY[ZS]E|FORMATTED|EXTENDED|COSTS?|CODEGEN|VERBOSE|LOGICAL|PHYSICAL|QUERY\s+PLAN)\b\s*)",
    re.IGNORECASE,
)
_ANALYZE_OPTION_RE = re.compile(
    r"\bANALY[ZS]E\b(?:\s*(?:=|\s)\s*(TRUE|FALSE|ON|OFF|1|0)\b)?",
    re.IGNORECASE,
)
_SPARK_FILE_FORMATS = frozenset(
    {
        "parquet",
        "delta",
        "csv",
        "json",
        "text",
        "orc",
        "avro",
        "libsvm",
        "binaryfile",
        "iceberg",
    }
)

_BLOCKED_STMT_TYPES: dict[str, tuple[type[exp.Expression], ...]] = {
    "DROP": (exp.Drop,),
    "DELETE": (exp.Delete,),
    "UPDATE": (exp.Update,),
    "INSERT": (exp.Insert,),
    "COPY": (exp.Copy,),
    "ATTACH": (exp.Attach,),
    "DETACH": (exp.Detach,),
    "PRAGMA": (exp.Pragma,),
    "INSTALL": (exp.Install,),
    "SET": (exp.Set,),
    "ALTER": (exp.Alter,),
    "CREATE": (exp.Create,),
    "TRUNCATE": (exp.TruncateTable,),
    "MERGE": (exp.Merge,),
}

_ALWAYS_BLOCKED_FILE_FUNCS = frozenset(
    {
        "getenv",
        "duckdb_secrets",
        "duckdb_settings",
        "sqlite_scan",
        "sqlite_attach",
        "postgres_scan",
        "mysql_scan",
        "http_get",
        "query",
        "query_table",
        "json_execute_serialized_sql",
        "json_serialize_sql",
        "sniff_csv",
        "current_setting",
        "iceberg_scan",
        "iceberg_metadata",
        "iceberg_snapshots",
        "delta_metadata",
        "xlsx_scan",
        "st_read",
        "read_xlsx",
        "read_ndjson",
        "read_ndjson_auto",
        "read_ndjson_objects",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_ls_logdir",
        "pg_ls_waldir",
        "pg_ls_tmpdir",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "query_to_xml",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "set_config",
        "lo_get",
        "lo_put",
        "lo_from_bytea",
        "lo_unlink",
        "pg_rotate_logfile",
        "pg_switch_wal",
        "cursor_to_xml",
        "query_to_xml_and_xmlschema",
        "reflect",
        "java_method",
        "xpath",
        "xpath_string",
        "xpath_boolean",
        "xpath_double",
        "xpath_float",
        "xpath_int",
        "xpath_long",
        "xpath_number",
        "xpath_short",
        "postgres_query",
        "mysql_query",
        "sqlite_query",
        "http_post",
        "http_put",
        "http_delete",
        "http_patch",
        "http_head",
        "parquet_metadata",
        "parquet_schema",
        "parquet_file_metadata",
        "parquet_kv_metadata",
    }
)

_BLOCKED_FUNC_PREFIXES = ("pragma_", "duckdb_", "pg_ls_", "pg_file_", "dblink_")
_BLOCKED_FUNC_SUFFIXES = ("_to_xml", "_to_xmlschema", "_to_xml_and_xmlschema")


def is_blocked_escape_func(name: str) -> bool:
    """Return True if SQL function `name` invokes dynamic SQL, metadata, or external escapes."""
    lowered = name.lower()
    return (
        lowered in _ALWAYS_BLOCKED_FILE_FUNCS
        or lowered.startswith(_BLOCKED_FUNC_PREFIXES)
        or lowered.endswith(_BLOCKED_FUNC_SUFFIXES)
    )

_LAKEHOUSE_RELATIVE_PREFIX = "workspace/lakehouse/"


def parse_single_statement(sql: str, dialect: str = "duckdb") -> exp.Expression:
    """Parse `sql` in `dialect` and require exactly one non-empty AST statement."""
    if not sql or not sql.strip():
        raise ValueError("SQL query must not be empty.")
    read_dialect = normalize_sql_dialect(dialect)
    try:
        parsed = sqlglot.parse(sql, read=read_dialect)
    except ParseError as exc:
        raise ValueError(f"Unparseable SQL: {exc}") from exc

    if not parsed or len(parsed) != 1 or parsed[0] is None:
        raise ValueError("Exactly one SQL statement is permitted per call.")
    return parsed[0]


def is_explain_statement(stmt: exp.Expression) -> bool:
    """Return True if `stmt` represents an EXPLAIN command."""
    if isinstance(stmt, exp.Command) and str(stmt.this or "").upper() == "EXPLAIN":
        return True
    explain_cls = getattr(exp, "Explain", None)
    return bool(explain_cls is not None and isinstance(stmt, explain_cls))


def is_explain_analyze(stmt: exp.Expression) -> bool:
    """Return True if an EXPLAIN statement includes an active ANALYZE/ANALYSE option."""
    explain_cls = getattr(exp, "Explain", None)
    if explain_cls is not None and isinstance(stmt, explain_cls):
        return bool(stmt.args.get("analyze"))
    if isinstance(stmt, exp.Command) and str(stmt.this or "").upper() == "EXPLAIN":
        raw_expr = stmt.expression
        cleaned = (
            raw_expr.name if isinstance(raw_expr, exp.Literal) else str(raw_expr or "")
        ).strip()
        while cleaned:
            match = _EXPLAIN_PREFIX_RE.match(cleaned)
            if not match:
                break
            prefix_tok = match.group(0)
            for opt_match in _ANALYZE_OPTION_RE.finditer(prefix_tok):
                flag = opt_match.group(1)
                if flag is None or flag.upper() in {"TRUE", "ON", "1"}:
                    return True
            cleaned = cleaned[match.end() :].strip()
    return False


def unwrap_explain(stmt: exp.Expression, dialect: str = "duckdb") -> exp.Expression:
    """Extract and parse the inner SQL statement wrapped by an EXPLAIN command."""
    explain_cls = getattr(exp, "Explain", None)
    if (
        explain_cls is not None
        and isinstance(stmt, explain_cls)
        and isinstance(stmt.this, exp.Expression)
    ):
        return stmt.this

    raw_expr = stmt.expression
    inner = raw_expr.name if isinstance(raw_expr, exp.Literal) else str(raw_expr or "")
    cleaned = inner.strip()
    while cleaned:
        match = _EXPLAIN_PREFIX_RE.match(cleaned)
        if not match:
            break
        cleaned = cleaned[match.end() :].strip()
    if not cleaned:
        raise ValueError("EXPLAIN statement is missing an inner SQL query.")
    return parse_single_statement(cleaned, dialect=dialect)


def extract_referenced_tables(
    stmt: exp.Expression, dialect: str = "duckdb"
) -> list[tuple[str, str, str]]:
    """Return unique `(catalog, db, table_name)` tuples referenced by real tables in `stmt`."""
    if is_explain_statement(stmt):
        return extract_referenced_tables(unwrap_explain(stmt, dialect=dialect), dialect=dialect)
    if isinstance(stmt, exp.Describe):
        if isinstance(stmt.this, exp.Table) and stmt.this.name:
            t = stmt.this
            return [((t.catalog or "").lower(), (t.db or "").lower(), t.name.lower())]
        if isinstance(stmt.this, exp.Expression):
            return extract_referenced_tables(stmt.this, dialect=dialect)
        return []

    seen: list[tuple[str, str, str]] = []
    root = build_scope(stmt)
    if root is not None:
        for scope in root.traverse():
            for _node, source in scope.selected_sources.values():
                if isinstance(source, exp.Table) and source.name:
                    key = (
                        (source.catalog or "").lower(),
                        (source.db or "").lower(),
                        source.name.lower(),
                    )
                    if key not in seen:
                        seen.append(key)
        return seen

    for table in stmt.find_all(exp.Table):
        if table.name:
            key = (
                (table.catalog or "").lower(),
                (table.db or "").lower(),
                table.name.lower(),
            )
            if key not in seen:
                seen.append(key)
    return seen


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return node.name.lower()
    return node.sql_name().lower()


def _is_file_reading_func(name: str) -> bool:
    return name.startswith(("read_", "parquet_")) or name.endswith("_scan")


def _extract_string_literals(expr: exp.Expression) -> list[str]:
    values: list[str] = []
    for node in expr.walk():
        if isinstance(node, exp.Literal) and node.is_string:
            values.append(node.this)
        elif isinstance(node, exp.Identifier) and node.quoted:
            values.append(node.name)
    return values


def _is_path_inside_lakehouse(
    raw_path: str,
    lakehouse_dir: Path | None,
    workspace_dir: Path | None = None,
) -> bool:
    cleaned = raw_path.strip().replace("\\", "/")
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if "://" in lowered or cleaned.startswith("~"):
        return False

    raw_segments = [seg for seg in cleaned.split("/") if seg]
    if ".." in raw_segments:
        return False

    # Strip glob wildcards from suffix when checking parent directory containment.
    probe = re.split(r"[*?\[\]]", cleaned, maxsplit=1)[0].rstrip("/") or cleaned

    if cleaned.startswith("/") or (len(cleaned) >= 2 and cleaned[1] == ":"):
        if lakehouse_dir is None:
            return False
        try:
            resolved = Path(probe).resolve()
            return resolved.is_relative_to(lakehouse_dir)
        except OSError:
            return False

    if lakehouse_dir is not None and workspace_dir is not None:
        try:
            candidate = (workspace_dir / probe).resolve()
            if candidate.is_relative_to(lakehouse_dir):
                return True
            if workspace_dir.name == "workspace":
                parent_cand = (workspace_dir.parent / probe).resolve()
                if parent_cand.is_relative_to(lakehouse_dir):
                    return True
        except OSError:
            return False
        return False

    normalized = posixpath.normpath(cleaned).lstrip("./")
    if normalized == ".." or normalized.startswith("../"):
        return False
    return normalized.startswith(_LAKEHOUSE_RELATIVE_PREFIX)


def _strip_parens_and_casts(expr: exp.Expression) -> exp.Expression:
    cur = expr
    while isinstance(cur, (exp.Paren, exp.Cast, exp.TryCast)):
        cur = cur.this
    return cur


def _extract_col_ref(expr: exp.Expression) -> tuple[str, str] | None:
    inner = _strip_parens_and_casts(expr)
    if isinstance(inner, exp.Column) and inner.name:
        return (inner.table.lower() if inner.table else "", inner.name.lower())
    return None


def _col_matches_target(
    col_ref: tuple[str, str],
    qualifiers: set[str],
    partition_cols: set[str],
    allow_unqualified: bool,
) -> bool:
    tbl, col = col_ref
    if col not in partition_cols:
        return False
    if not tbl:
        return allow_unqualified
    return tbl in qualifiers


def _predicate_filters_target(
    pred: exp.Expression,
    qualifiers: set[str],
    partition_cols: set[str],
    allow_unqualified: bool,
    equivalences: dict[tuple[str, str], set[tuple[str, str]]] | None = None,
) -> bool:
    cur = pred.this if isinstance(pred, exp.Paren) else pred
    if isinstance(cur, exp.And):
        left_ok = _predicate_filters_target(
            cur.this, qualifiers, partition_cols, allow_unqualified, equivalences
        )
        right_ok = _predicate_filters_target(
            cur.expression, qualifiers, partition_cols, allow_unqualified, equivalences
        )
        return left_ok or right_ok
    if isinstance(cur, exp.Or):
        # Do not record equivalence edges inside OR branches (e.g. `a = b OR 1=1`).
        return _predicate_filters_target(
            cur.this, qualifiers, partition_cols, allow_unqualified, None
        ) and _predicate_filters_target(
            cur.expression, qualifiers, partition_cols, allow_unqualified, None
        )
    if isinstance(cur, (exp.Not, exp.NEQ, exp.Is)):
        return False

    if isinstance(cur, exp.Between):
        ref = _extract_col_ref(cur.this)
        low = cur.args.get("low")
        high = cur.args.get("high")
        if (
            ref is not None
            and low is not None
            and high is not None
            and not isinstance(low, exp.Null)
            and not isinstance(high, exp.Null)
            and not _has_any_column(low)
            and not _has_any_column(high)
        ):
            return _col_matches_target(ref, qualifiers, partition_cols, allow_unqualified)
        return False

    if isinstance(cur, exp.In):
        ref = _extract_col_ref(cur.this)
        sub_query = cur.args.get("query")
        if sub_query is not None:
            return ref is not None and _col_matches_target(
                ref, qualifiers, partition_cols, allow_unqualified
            )
        vals = cur.expressions or []
        if (
            ref is not None
            and vals
            and all(
                not isinstance(_strip_parens_and_casts(v), exp.Null)
                and not _has_any_column(v)
                for v in vals
            )
        ):
            return _col_matches_target(ref, qualifiers, partition_cols, allow_unqualified)
        return False

    if isinstance(cur, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        left_ref = _extract_col_ref(cur.this)
        right_ref = _extract_col_ref(cur.expression)
        left_is_null = isinstance(_strip_parens_and_casts(cur.this), exp.Null)
        right_is_null = isinstance(_strip_parens_and_casts(cur.expression), exp.Null)

        if (
            left_ref is not None
            and not right_is_null
            and _col_matches_target(left_ref, qualifiers, partition_cols, allow_unqualified)
        ):
            if right_ref is not None:
                if _col_matches_target(right_ref, qualifiers, partition_cols, allow_unqualified):
                    return False  # Reject tautologies like o.order_date = o.order_date.
                if isinstance(cur, exp.EQ) and equivalences is not None:
                    equivalences.setdefault(left_ref, set()).add(right_ref)
                return False
            # Reject expression tautologies like COALESCE(order_date, ...).
            return not _has_any_column(cur.expression)

        if (
            right_ref is not None
            and not left_is_null
            and _col_matches_target(right_ref, qualifiers, partition_cols, allow_unqualified)
        ):
            if left_ref is not None:
                if _col_matches_target(left_ref, qualifiers, partition_cols, allow_unqualified):
                    return False
                if isinstance(cur, exp.EQ) and equivalences is not None:
                    equivalences.setdefault(right_ref, set()).add(left_ref)
                return False
            return not _has_any_column(cur.this)
    return False


def _has_any_column(expr: exp.Expression) -> bool:
    return any(isinstance(node, exp.Column) for node in expr.walk())


def _extract_direct_filtered_cols(pred: exp.Expression) -> set[tuple[str, str]]:
    """Return column refs directly constrained by non-column predicates (excluding OR bypasses)."""
    cur = pred.this if isinstance(pred, exp.Paren) else pred
    if isinstance(cur, exp.And):
        return _extract_direct_filtered_cols(cur.this) | _extract_direct_filtered_cols(
            cur.expression
        )
    if isinstance(cur, exp.Or):
        return _extract_direct_filtered_cols(cur.this) & _extract_direct_filtered_cols(
            cur.expression
        )
    if isinstance(cur, (exp.Not, exp.NEQ, exp.Is)):
        return set()
    if isinstance(cur, exp.Between):
        ref = _extract_col_ref(cur.this)
        low = cur.args.get("low")
        high = cur.args.get("high")
        if (
            ref is not None
            and low is not None
            and high is not None
            and not isinstance(low, exp.Null)
            and not isinstance(high, exp.Null)
            and not _has_any_column(low)
            and not _has_any_column(high)
        ):
            return {ref}
        return set()
    if isinstance(cur, exp.In):
        ref = _extract_col_ref(cur.this)
        if ref is None:
            return set()
        if cur.args.get("query") is not None:
            return {ref}
        vals = cur.expressions or []
        if vals and all(
            not isinstance(_strip_parens_and_casts(v), exp.Null) and not _has_any_column(v)
            for v in vals
        ):
            return {ref}
        return set()
    if isinstance(cur, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        left_ref = _extract_col_ref(cur.this)
        right_ref = _extract_col_ref(cur.expression)
        left_is_null = isinstance(_strip_parens_and_casts(cur.this), exp.Null)
        right_is_null = isinstance(_strip_parens_and_casts(cur.expression), exp.Null)
        if left_ref is not None and not right_is_null and not _has_any_column(cur.expression):
            return {left_ref}
        if right_ref is not None and not left_is_null and not _has_any_column(cur.this):
            return {right_ref}
    return set()


def _extract_conjunctive_eq_pairs(
    pred: exp.Expression,
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Extract conjunctive column-to-column equalities (excluding OR branches)."""
    cur = pred.this if isinstance(pred, exp.Paren) else pred
    if isinstance(cur, exp.And):
        return _extract_conjunctive_eq_pairs(cur.this) + _extract_conjunctive_eq_pairs(
            cur.expression
        )
    if isinstance(cur, exp.EQ):
        left_ref = _extract_col_ref(cur.this)
        right_ref = _extract_col_ref(cur.expression)
        if left_ref is not None and right_ref is not None and left_ref != right_ref:
            return [(left_ref, right_ref)]
    return []


def _is_direct_agg_in_select(agg: exp.AggFunc, select_expr: exp.Select) -> bool:
    parent = agg.parent
    while parent is not None and parent is not select_expr:
        if isinstance(parent, (exp.Window, exp.Select, exp.Subquery, exp.CTE)):
            return False
        parent = parent.parent
    return parent is select_expr


def _is_direct_star_in_select(star: exp.Star, select_expr: exp.Select) -> bool:
    parent = star.parent
    while parent is not None and parent is not select_expr:
        if isinstance(parent, (exp.AggFunc, exp.Select, exp.Subquery, exp.CTE)):
            return False
        parent = parent.parent
    return parent is select_expr


_ROW_EXPANDING_FUNCS = frozenset({"unnest", "explode", "generate_series", "range"})


def _is_pure_scalar_or_aggregate(stmt: exp.Expression) -> bool:
    if not isinstance(stmt, exp.Select):
        return False
    if not stmt.args.get("from_") and not stmt.args.get("joins"):
        return not any(
            isinstance(node, (exp.Unnest, exp.Explode))
            or (isinstance(node, exp.Func) and _func_name(node) in _ROW_EXPANDING_FUNCS)
            for expr in stmt.expressions
            for node in expr.walk()
        )
    if stmt.args.get("group"):
        return False

    has_direct_agg = False
    for expr in stmt.expressions:
        for node in expr.walk():
            if isinstance(node, (exp.Unnest, exp.Explode)):
                return False
            if isinstance(node, exp.Func) and _func_name(node) in _ROW_EXPANDING_FUNCS:
                return False
            if isinstance(node, exp.Star) and _is_direct_star_in_select(node, stmt):
                return False
            if isinstance(node, exp.AggFunc) and _is_direct_agg_in_select(node, stmt):
                has_direct_agg = True
            if isinstance(node, exp.Column):
                parent = node.parent
                in_agg = False
                in_sub = False
                while parent is not None and parent is not stmt:
                    if isinstance(parent, (exp.Select, exp.Subquery)):
                        in_sub = True
                        break
                    if isinstance(parent, exp.AggFunc) and _is_direct_agg_in_select(parent, stmt):
                        in_agg = True
                        break
                    parent = parent.parent
                if not in_sub and not in_agg:
                    return False
    return has_direct_agg


class SqlGuardHook(PreToolUseHook):
    """AST-level SQL safety, escape prevention, partition-predicate, and LIMIT guard."""

    def __init__(
        self,
        policies: PoliciesConfig | None = None,
        workspace_dir: Path | None = None,
        *,
        dialect: str | None = None,
        lakehouse_dir: Path | None = None,
    ) -> None:
        self.policies = policies or load_policies()
        self.dialect = normalize_sql_dialect(dialect or self.policies.sql_guard.sql_dialect)
        self.workspace_dir: Path | None = (
            workspace_dir.resolve() if workspace_dir is not None else None
        )
        if lakehouse_dir is not None:
            self.lakehouse_dir: Path | None = lakehouse_dir.resolve()
        elif self.workspace_dir is not None:
            self.lakehouse_dir = self.workspace_dir / "lakehouse"
        else:
            self.lakehouse_dir = None

    def _validate_statement(self, stmt: exp.Expression, *, dialect: str = "duckdb") -> str | None:
        guard = self.policies.sql_guard
        allowed = {name.upper() for name in guard.allowed_top_level}

        if is_explain_statement(stmt):
            if "EXPLAIN" not in allowed:
                return "EXPLAIN statements are disabled by sql_guard policy."
            try:
                inner = unwrap_explain(stmt, dialect=dialect)
            except ValueError as exc:
                return str(exc)
            return self._validate_statement(inner, dialect=dialect)

        if isinstance(stmt, exp.Describe):
            if "DESCRIBE" not in allowed:
                return "DESCRIBE statements are disabled by sql_guard policy."
            if isinstance(stmt.this, exp.Expression) and not isinstance(stmt.this, exp.Table):
                return self._validate_statement(stmt.this, dialect=dialect)
            return self._check_ast_safety(stmt)

        if isinstance(stmt, exp.Select):
            if "SELECT" not in allowed:
                return "SELECT statements are disabled by sql_guard policy."
            if stmt.args.get("with_") and "WITH" not in allowed:
                return "WITH/CTE clauses are disabled by sql_guard policy."
            return self._check_ast_safety(stmt)

        if isinstance(stmt, exp.SetOperation):
            if "UNION" not in allowed:
                return "UNION/set-operation statements are disabled by sql_guard policy."
            if stmt.args.get("with_") and "WITH" not in allowed:
                return "WITH/CTE clauses are disabled by sql_guard policy."
            return self._check_ast_safety(stmt)

        stmt_name = type(stmt).__name__.upper()
        if isinstance(stmt, exp.Command):
            stmt_name = str(stmt.this or stmt_name).upper()
        return f"Disallowed SQL statement type: {stmt_name}"

    def _check_ast_safety(self, stmt: exp.Expression) -> str | None:
        guard = self.policies.sql_guard
        blocked_stmts = {s.upper() for s in guard.blocked_statements}
        blocked_funcs = {f.lower() for f in guard.blocked_functions}

        for node in stmt.walk():
            if isinstance(node, exp.Into):
                return "SELECT ... INTO is not allowed in read-only SQL mode."
            if isinstance(node, (exp.Summarize, exp.Show)):
                return f"Disallowed SQL operation: {type(node).__name__.upper()}"
            if isinstance(node, exp.Command):
                cmd = str(node.this or "").upper()
                if cmd in blocked_stmts or cmd != "EXPLAIN":
                    return f"Blocked SQL command: {cmd or 'COMMAND'}"
            for stmt_kw, node_types in _BLOCKED_STMT_TYPES.items():
                if stmt_kw in blocked_stmts and isinstance(node, node_types):
                    return f"Blocked SQL statement: {stmt_kw}"
            if "CALL" in blocked_stmts and node.__class__.__name__.upper() == "CALL":
                return "Blocked SQL statement: CALL"

            if isinstance(node, exp.Func):
                fn = _func_name(node)
                if fn in blocked_funcs:
                    return f"Blocked SQL function: {fn}"
                if is_blocked_escape_func(fn):
                    return f"Blocked external/system SQL function: {fn}"
                if _is_file_reading_func(fn):
                    literals = _extract_string_literals(node)
                    if not literals or any(
                        not _is_path_inside_lakehouse(p, self.lakehouse_dir, self.workspace_dir)
                        for p in literals
                    ):
                        return (
                            f"External file-reading function {fn!r} outside "
                            "workspace/lakehouse/ is blocked."
                        )

            if isinstance(node, exp.Table):
                is_literal_path = isinstance(node.this, exp.Literal) and node.this.is_string
                raw_name = node.name or ""
                looks_like_path = "/" in raw_name or "\\" in raw_name or "://" in raw_name
                is_format_table = (
                    (node.db or "").lower() in _SPARK_FILE_FORMATS
                    and isinstance(node.this, (exp.Identifier, exp.Literal))
                )
                if (
                    (is_literal_path or looks_like_path or is_format_table)
                    and not _is_path_inside_lakehouse(
                        raw_name, self.lakehouse_dir, self.workspace_dir
                    )
                ):
                    return (
                        f"Direct file scan outside workspace/lakehouse/ is blocked: "
                        f"{raw_name!r}"
                    )
        return None

    def _check_limits(self, stmt: exp.Expression, *, dialect: str = "duckdb") -> str | None:
        if is_explain_statement(stmt):
            if not is_explain_analyze(stmt):
                return None
            try:
                stmt = unwrap_explain(stmt, dialect=dialect)
            except ValueError as exc:
                return str(exc)
        max_limit = self.policies.sql_guard.max_limit
        for limit_node in [*stmt.find_all(exp.Limit), *stmt.find_all(exp.Fetch)]:
            limit_options = limit_node.args.get("limit_options")
            if limit_options is not None and limit_options.args.get("percent"):
                return f"PERCENT row limits are forbidden; use explicit LIMIT <= {max_limit}."
            expr = (
                limit_node.args.get("count")
                if isinstance(limit_node, exp.Fetch)
                else limit_node.expression
            )
            if not isinstance(expr, exp.Literal) or expr.is_string:
                return f"Explicit LIMIT must be an integer <= {max_limit}."
            try:
                value = int(expr.this)
            except ValueError:
                return f"Invalid LIMIT value: {expr.this!r}"
            if value < 0 or value > max_limit:
                return f"LIMIT {value} exceeds maximum allowed limit ({max_limit})."
        return None

    def _is_partition_discovery_scope(
        self,
        scope: Scope,
        table_name: str,
        qualifiers: set[str],
        partition_cols: set[str],
    ) -> bool:
        if not scope.is_root:
            return False
        select_expr = scope.expression
        if not isinstance(select_expr, exp.Select):
            return False
        if select_expr.args.get("joins") or len(scope.selected_sources) != 1:
            return False
        if not partition_cols:
            return False

        has_mode = bool(select_expr.args.get("distinct")) or bool(select_expr.args.get("group"))
        seen_partition_col = False

        check_roots: list[Any] = list(select_expr.expressions)
        for arg_name in ("group", "order", "having", "qualify"):
            arg_val = select_expr.args.get(arg_name)
            if arg_val is not None:
                check_roots.append(arg_val)

        for root_expr in check_roots:
            for node in root_expr.walk():
                if isinstance(node, (exp.Star, exp.Select, exp.Subquery)):
                    return False
                if isinstance(node, exp.AggFunc):
                    has_mode = True
                if isinstance(node, exp.Column) and node.name:
                    tbl = node.table.lower() if node.table else ""
                    if tbl and tbl not in qualifiers and tbl != table_name:
                        return False
                    if node.name.lower() not in partition_cols:
                        return False
                    seen_partition_col = True

        return has_mode and seen_partition_col

    def _check_partitions(self, stmt: exp.Expression, *, dialect: str = "duckdb") -> str | None:
        if is_explain_statement(stmt):
            if not is_explain_analyze(stmt):
                return None
            try:
                stmt = unwrap_explain(stmt, dialect=dialect)
            except ValueError as exc:
                return str(exc)
        if isinstance(stmt, exp.Describe):
            return None
        partitioned = {
            tbl.lower(): {c.lower() for c in cols if str(c).strip()}
            for tbl, cols in self.policies.partitioned_tables.items()
            if any(str(c).strip() for c in cols)
        }
        if not partitioned:
            return None

        short_partitioned: dict[str, set[str]] = {}
        for tbl_key, p_cols in partitioned.items():
            short = tbl_key.rsplit(".", 1)[-1]
            short_partitioned.setdefault(short, set()).update(p_cols)

        root = build_scope(stmt)
        if root is None:
            return None

        for scope in root.traverse():
            select_expr = scope.expression
            if not isinstance(select_expr, exp.Select):
                continue

            bindings: list[tuple[str, set[str], set[str], exp.Table]] = []
            node_to_quals: dict[int, set[str]] = {}
            for alias_key, (table_node, source) in scope.selected_sources.items():
                quals = {alias_key.lower()}
                if isinstance(source, exp.Table) and source.name:
                    cat = (source.catalog or "").lower()
                    db = (source.db or "").lower()
                    t_name = source.name.lower()
                    if not (isinstance(table_node, exp.Table) and table_node.alias):
                        quals.add(t_name)
                    node_to_quals[id(table_node)] = quals
                    candidates = [
                        ".".join(p for p in (cat, db, t_name) if p),
                        ".".join(p for p in (db, t_name) if p),
                        t_name,
                    ]
                    matched_cols = next(
                        (partitioned[cand] for cand in candidates if cand in partitioned), None
                    )
                    if matched_cols is None:
                        matched_cols = short_partitioned.get(t_name)
                    if matched_cols:
                        bindings.append((t_name, quals, matched_cols, table_node))

            if not bindings:
                continue
            if len(bindings) == 1 and self._is_partition_discovery_scope(
                scope, bindings[0][0], bindings[0][1], bindings[0][2]
            ):
                continue

            allow_unqualified = len(scope.selected_sources) == 1 or len(bindings) == 1
            where_node = select_expr.args.get("where")
            joins = select_expr.args.get("joins") or []

            # Build constraint graph from WHERE and constraining JOIN ONs.
            equivalences: dict[tuple[str, str], set[tuple[str, str]]] = {}
            directly_filtered_cols: set[tuple[str, str]] = set()
            directly_filtered: list[bool] = [False] * len(bindings)

            if where_node is not None and where_node.this is not None:
                directly_filtered_cols |= _extract_direct_filtered_cols(where_node.this)
                for left_ref, right_ref in _extract_conjunctive_eq_pairs(where_node.this):
                    equivalences.setdefault(left_ref, set()).add(right_ref)
                    equivalences.setdefault(right_ref, set()).add(left_ref)

            for join in joins:
                on_expr = join.args.get("on")
                if on_expr is None:
                    continue
                side = (join.side or "").upper()
                if side == "FULL":
                    continue
                if not side or side == "INNER":
                    directly_filtered_cols |= _extract_direct_filtered_cols(on_expr)
                    for left_ref, right_ref in _extract_conjunctive_eq_pairs(on_expr):
                        equivalences.setdefault(left_ref, set()).add(right_ref)
                        equivalences.setdefault(right_ref, set()).add(left_ref)
                elif side == "LEFT":
                    right_quals = node_to_quals.get(id(join.this), set())
                    for col in _extract_direct_filtered_cols(on_expr):
                        if col[0] in right_quals:
                            directly_filtered_cols.add(col)
                    for left_ref, right_ref in _extract_conjunctive_eq_pairs(on_expr):
                        if right_ref[0] in right_quals:
                            equivalences.setdefault(right_ref, set()).add(left_ref)
                        if left_ref[0] in right_quals:
                            equivalences.setdefault(left_ref, set()).add(right_ref)
                elif side == "RIGHT":
                    right_quals = node_to_quals.get(id(join.this), set())
                    for col in _extract_direct_filtered_cols(on_expr):
                        if col[0] and col[0] not in right_quals:
                            directly_filtered_cols.add(col)
                    for left_ref, right_ref in _extract_conjunctive_eq_pairs(on_expr):
                        if left_ref[0] and left_ref[0] not in right_quals:
                            equivalences.setdefault(left_ref, set()).add(right_ref)
                        if right_ref[0] and right_ref[0] not in right_quals:
                            equivalences.setdefault(right_ref, set()).add(left_ref)

            for idx, (_t_name, quals, p_cols, table_node) in enumerate(bindings):
                if (
                    where_node is not None
                    and where_node.this is not None
                    and _predicate_filters_target(
                        where_node.this, quals, p_cols, allow_unqualified, equivalences
                    )
                ):
                    directly_filtered[idx] = True
                    continue

                for join in joins:
                    on_expr = join.args.get("on")
                    if on_expr is None:
                        continue
                    side = (join.side or "").upper()
                    # LEFT JOIN ON constrains only the right joined table; FULL JOIN neither.
                    if side == "FULL":
                        continue
                    if side == "LEFT" and join.this is not table_node:
                        continue
                    if side == "RIGHT" and join.this is table_node:
                        continue
                    if _predicate_filters_target(
                        on_expr, quals, p_cols, allow_unqualified, equivalences
                    ):
                        directly_filtered[idx] = True
                        break

            # Propagate partition constraints through equivalence classes.
            for idx, (_t_name, quals, p_cols, _node) in enumerate(bindings):
                if directly_filtered[idx]:
                    continue
                if self._satisfied_by_equivalence(
                    quals,
                    p_cols,
                    allow_unqualified,
                    equivalences,
                    directly_filtered_cols,
                    bindings,
                    directly_filtered,
                ):
                    directly_filtered[idx] = True

            for idx, (t_name, _quals, p_cols, _node) in enumerate(bindings):
                if not directly_filtered[idx]:
                    cols_str = ", ".join(sorted(p_cols))
                    return (
                        f"Query on partitioned table {t_name!r} requires a partition "
                        f"predicate on ({cols_str}) in WHERE or JOIN ON."
                    )
        return None

    @staticmethod
    def _satisfied_by_equivalence(
        quals: set[str],
        p_cols: set[str],
        allow_unqualified: bool,
        equivalences: dict[tuple[str, str], set[tuple[str, str]]],
        directly_filtered_cols: set[tuple[str, str]],
        bindings: list[tuple[str, set[str], set[str], exp.Table]],
        directly_filtered: list[bool],
    ) -> bool:
        seeds = [
            ref
            for ref in equivalences
            if _col_matches_target(ref, quals, p_cols, allow_unqualified)
        ]
        if not seeds:
            return False

        visited: set[tuple[str, str]] = set(seeds)
        stack = list(seeds)
        while stack:
            cur = stack.pop()
            for nxt in equivalences.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
                if nxt in directly_filtered_cols:
                    return True
                for b_idx, (_tn, b_quals, b_cols, _tnode) in enumerate(bindings):
                    if directly_filtered[b_idx] and _col_matches_target(
                        nxt, b_quals, b_cols, len(bindings) == 1
                    ):
                        return True
        return False

    def on_pre_tool_use(self, state: RunState, tool_call: ToolCall) -> PreToolDecision:
        """Validate SQL AST and optionally inject `LIMIT max_limit` for unbounded row queries."""
        if tool_call.name != "run_sql_readonly":
            return PreToolDecision(action="allow")

        sql = str(tool_call.arguments.get("sql") or "")
        engine_arg = str(tool_call.arguments.get("engine") or "").strip().lower()
        effective_dialect = (
            "spark" if engine_arg == "spark" else self.dialect
        )
        try:
            stmt = parse_single_statement(sql, dialect=effective_dialect)
        except ValueError as exc:
            return PreToolDecision(action="block", reason=str(exc))

        if reason := self._validate_statement(stmt, dialect=effective_dialect):
            return PreToolDecision(action="block", reason=reason)
        if reason := self._check_limits(stmt, dialect=effective_dialect):
            return PreToolDecision(action="block", reason=reason)
        if reason := self._check_partitions(stmt, dialect=effective_dialect):
            return PreToolDecision(action="block", reason=reason)

        if (
            isinstance(stmt, (exp.Select, exp.SetOperation))
            and stmt.args.get("limit") is None
            and not _is_pure_scalar_or_aggregate(stmt)
        ):
            max_limit = self.policies.sql_guard.max_limit
            rewritten_sql = stmt.limit(max_limit).sql(dialect=effective_dialect)
            modified_args = dict(tool_call.arguments)
            modified_args["sql"] = rewritten_sql
            return PreToolDecision(
                action="modify",
                reason=f"Added default LIMIT {max_limit} to unbounded row query.",
                modified_arguments=modified_args,
            )

        return PreToolDecision(action="allow")
