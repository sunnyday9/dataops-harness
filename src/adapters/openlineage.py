from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.adapters.base import LineageNode
from src.core.aggregator import RawAlert
from src.core.policy import normalize_sql_dialect

_SQL_NAMESPACE_SCHEMES: dict[str, str] = {
    "postgres": "postgres",
    "postgresql": "postgres",
    "pg": "postgres",
    "duckdb": "duckdb",
    "spark": "spark",
    "databricks": "spark",
    "snowflake": "snowflake",
    "bigquery": "bigquery",
    "clickhouse": "clickhouse",
    "trino": "trino",
    "presto": "trino",
}

_PII_KEYWORDS = frozenset({"pii", "sensitive", "confidential", "restricted"})
_CRITICAL_ERROR_RE = re.compile(r"(?:outofmemoryerror|\boom\b|\bfatal\b)", re.IGNORECASE)


def _strip_quotes(token: str) -> str:
    return token.strip().strip('"`\'')


_DATA_FILE_EXTENSIONS = (
    ".parquet",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".orc",
    ".avro",
    ".delta",
    ".iceberg",
)


def normalize_dataset_name(name: str) -> str:
    """Return the canonical short table name for an OpenLineage dataset identifier."""
    cleaned = name.strip().replace("\\", "/")
    if not cleaned:
        return ""
    if "://" in cleaned:
        _, _, rest = cleaned.partition("://")
        rest = rest.strip("/")
        if "/" in rest:
            cleaned = rest.rsplit("/", 1)[-1]
        elif rest:
            cleaned = rest
    elif "/" in cleaned:
        cleaned = cleaned.strip("/")
        if "/" in cleaned:
            cleaned = cleaned.rsplit("/", 1)[-1]
    cleaned = _strip_quotes(cleaned)
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for ext in _DATA_FILE_EXTENSIONS:
        if lowered.endswith(ext) and len(cleaned) > len(ext):
            cleaned = cleaned[: -len(ext)]
            break
    parts = [_strip_quotes(part) for part in cleaned.split(".") if _strip_quotes(part)]
    if not parts:
        return _strip_quotes(cleaned)
    return parts[-1]


def infer_dialect_from_namespace(namespace: str | None) -> str | None:
    """Infer a SQLGlot dialect from an OpenLineage namespace URI scheme."""
    if not namespace or not namespace.strip():
        return None
    cleaned = namespace.strip().lower().removeprefix("jdbc:")
    if "://" in cleaned:
        scheme = cleaned.split("://", 1)[0].split("+", 1)[0].strip()
    else:
        scheme = cleaned.split(":", 1)[0].split("/", 1)[0].split("+", 1)[0].strip()
    mapped = _SQL_NAMESPACE_SCHEMES.get(scheme)
    return normalize_sql_dialect(mapped) if mapped is not None else None


def _parse_openlineage_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        events: list[dict[str, Any]] = []
        for idx, raw_line in enumerate(stripped.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed OpenLineage JSONL event on line {idx}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise TypeError(
                    f"OpenLineage event on line {idx} must be a JSON object."
                )
            events.append(item)
        return events

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError("OpenLineage JSON array must contain only event objects.")
        return [dict(item) for item in parsed]
    raise ValueError("OpenLineage payload must be a JSON object, JSON array, or JSONL stream.")


def load_openlineage_events(
    source: Path | str | dict[str, Any] | Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load OpenLineage RunEvents from a file, directory, JSON/JSONL string, dict, or list."""
    if isinstance(source, Path):
        resolved = source.resolve()
        if resolved.is_dir():
            files = sorted(
                {
                    *resolved.rglob("*.json"),
                    *resolved.rglob("*.jsonl"),
                }
            )
            collected: list[dict[str, Any]] = []
            for file_path in files:
                collected.extend(
                    _parse_openlineage_text(file_path.read_text(encoding="utf-8"))
                )
            return collected
        if not resolved.is_file():
            raise FileNotFoundError(f"OpenLineage source not found: {source}")
        return _parse_openlineage_text(resolved.read_text(encoding="utf-8"))

    if isinstance(source, dict):
        return [dict(source)]

    if isinstance(source, str):
        return _parse_openlineage_text(source)

    events: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            raise TypeError("OpenLineage event sequence items must be JSON objects.")
        events.append(dict(item))
    return events


def _extend_unique(target: list[str], additions: Sequence[str]) -> None:
    seen = set(target)
    for item in additions:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            target.append(cleaned)
            seen.add(cleaned)


def _infer_job_engine(job: dict[str, Any], event: dict[str, Any]) -> str:
    namespace = str(job.get("namespace") or "").lower()
    name = str(job.get("name") or "").lower()
    producer = str(event.get("producer") or "").lower()
    facets = job.get("facets") if isinstance(job.get("facets"), dict) else {}
    combined = f"{namespace} {name} {producer}"
    if "dbt" in combined:
        return "dbt"
    if "spark" in combined or any("spark" in str(k).lower() for k in facets):
        return "spark"
    return "external"


def _extract_relative_source_path(job: dict[str, Any]) -> str | None:
    facets = job.get("facets") if isinstance(job.get("facets"), dict) else {}
    src_facet = facets.get("sourceCodeLocation")
    if not isinstance(src_facet, dict):
        return None
    for key in ("path", "url"):
        val = src_facet.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        cleaned = val.strip().replace("\\", "/")
        if not cleaned or "://" in cleaned or cleaned.startswith(("/", "~")):
            continue
        if any(seg == ".." for seg in cleaned.split("/")):
            continue
        norm = posixpath.normpath(cleaned)
        if not norm or norm in (".", "..") or norm.startswith("../"):
            continue
        lower_norm = norm.lower()
        if lower_norm in (
            "dataops-harness.yaml",
            "dataops-harness.yml",
            ".git",
        ) or lower_norm.startswith((".git/", "configs/")):
            continue
        return norm
    return None


def _is_pii_field(field: dict[str, Any]) -> bool:
    for key in ("tags", "classification", "classifications", "labels"):
        val = field.get(key)
        if isinstance(val, str) and val.strip().lower() in _PII_KEYWORDS:
            return True
        if isinstance(val, list):
            for entry in val:
                if isinstance(entry, str) and entry.strip().lower() in _PII_KEYWORDS:
                    return True
                if isinstance(entry, dict):
                    name = str(entry.get("name") or entry.get("key") or entry.get("tag") or "")
                    if name.strip().lower() in _PII_KEYWORDS:
                        return True
    meta = field.get("meta") if isinstance(field.get("meta"), dict) else {}
    if meta.get("pii") is True or meta.get("is_pii") is True or meta.get("contains_pii") is True:
        return True
    for key in ("sensitivity", "classification"):
        s_val = str(meta.get(key) or field.get(key) or "").strip().lower()
        if s_val in _PII_KEYWORDS:
            return True
    return bool(field.get("pii") is True or field.get("is_pii") is True)


def _is_partition_field(field: dict[str, Any]) -> bool:
    if field.get("partition") is True or field.get("is_partition") is True:
        return True
    meta = field.get("meta") if isinstance(field.get("meta"), dict) else {}
    if meta.get("partition") is True or meta.get("is_partition") is True:
        return True
    tags = field.get("tags")
    if isinstance(tags, list):
        for entry in tags:
            label = entry.get("name", "") if isinstance(entry, dict) else str(entry)
            if label.strip().lower() in {"partition", "partition_key"}:
                return True
    return False


def _extract_dataset_details(
    dataset: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str], str | None]:
    facets = dataset.get("facets") if isinstance(dataset.get("facets"), dict) else {}
    columns: list[str] = []
    pii_columns: list[str] = []
    partition_columns: list[str] = []
    tags: list[str] = []

    schema_facet = facets.get("schema") if isinstance(facets.get("schema"), dict) else {}
    for field in schema_facet.get("fields") or []:
        if not isinstance(field, dict):
            continue
        col_name = str(field.get("name") or "").strip()
        if not col_name:
            continue
        _extend_unique(columns, [col_name])
        if _is_pii_field(field):
            _extend_unique(pii_columns, [col_name])
        if _is_partition_field(field):
            _extend_unique(partition_columns, [col_name])

    storage_facet = facets.get("storage") if isinstance(facets.get("storage"), dict) else {}
    for key in ("partitionColumns", "partitionBy", "partition_columns"):
        val = storage_facet.get(key)
        if isinstance(val, list):
            _extend_unique(partition_columns, [str(v) for v in val])
        elif isinstance(val, str):
            _extend_unique(partition_columns, [val])

    dataops_facet = facets.get("dataops") if isinstance(facets.get("dataops"), dict) else {}
    pii_facet = facets.get("pii") if isinstance(facets.get("pii"), dict) else {}
    for source_facet, pii_keys in (
        (dataops_facet, ("pii_columns", "piiColumns")),
        (pii_facet, ("pii_columns", "piiColumns", "columns")),
    ):
        for key in pii_keys:
            val = source_facet.get(key)
            if isinstance(val, list):
                _extend_unique(pii_columns, [str(v) for v in val])
        part_val = source_facet.get("partition_columns") or source_facet.get("partitionColumns")
        if isinstance(part_val, list):
            _extend_unique(partition_columns, [str(v) for v in part_val])

    tags_facet = facets.get("tags")
    if isinstance(tags_facet, dict):
        raw_tags = tags_facet.get("tags")
        if isinstance(raw_tags, list):
            for entry in raw_tags:
                if isinstance(entry, dict):
                    label = entry.get("key") or entry.get("name") or entry.get("value")
                    if label:
                        _extend_unique(tags, [str(label)])
                elif entry:
                    _extend_unique(tags, [str(entry)])
    raw_ds_tags = dataops_facet.get("tags")
    if isinstance(raw_ds_tags, list):
        _extend_unique(tags, [str(t) for t in raw_ds_tags])

    explicit_layer = dataops_facet.get("layer")
    return (
        columns,
        pii_columns,
        partition_columns,
        tags,
        str(explicit_layer).strip().lower() if isinstance(explicit_layer, str) and explicit_layer.strip() else None,
    )


def _infer_dataset_layer(raw_name: str, tags: Sequence[str], *, is_output: bool) -> str:
    lowered_tags = {t.lower() for t in tags}
    if lowered_tags & {"mart", "marts", "gold"}:
        return "mart"
    if "intermediate" in lowered_tags:
        return "intermediate"
    if "staging" in lowered_tags or "bronze" in lowered_tags:
        return "staging"
    if "silver" in lowered_tags:
        return "silver"

    lowered = raw_name.lower()
    short = normalize_dataset_name(lowered).lower()
    tokens = {part for part in lowered.replace(".", "/").split("/") if part}
    if (
        tokens & {"mart", "marts", "gold"}
        or short.startswith(("fct_", "dim_", "mart_"))
    ):
        return "mart"
    if tokens & {"intermediate"} or short.startswith("int_"):
        return "intermediate"
    if tokens & {"staging", "stg"} or short.startswith("stg_"):
        return "staging"
    if "silver" in tokens or short.startswith("silver_"):
        return "silver"
    if tokens & {"raw", "bronze", "landing"} or short.startswith("raw_"):
        return "raw"
    return "external" if is_output else "raw"


class OpenLineageGraphData(BaseModel):
    """Normalized lineage graph, name aliases, and SQL dialect discovered from OpenLineage."""

    nodes: dict[str, LineageNode] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    sql_dialect: str | None = None


def extract_openlineage_lineage(events: Sequence[dict[str, Any]]) -> OpenLineageGraphData:
    """Convert OpenLineage RunEvents into merged `LineageNode` records and alias mappings."""
    nodes: dict[str, LineageNode] = {}
    aliases: dict[str, str] = {}
    detected_dialect: str | None = None

    def _register_alias(raw_name: str, canonical: str) -> None:
        aliases.setdefault(raw_name, canonical)
        aliases.setdefault(raw_name.lower(), canonical)
        aliases.setdefault(canonical.lower(), canonical)

    for event in events:
        job = event.get("job") if isinstance(event.get("job"), dict) else {}
        job_ns = str(job.get("namespace") or "") or None
        if detected_dialect is None and job_ns:
            detected_dialect = infer_dialect_from_namespace(job_ns)

        job_engine = _infer_job_engine(job, event)
        job_file_path = _extract_relative_source_path(job)

        input_names: list[str] = []
        for raw_ds in event.get("inputs") or []:
            if not isinstance(raw_ds, dict):
                continue
            raw_name = str(raw_ds.get("name") or "").strip()
            if not raw_name:
                continue
            canonical = normalize_dataset_name(raw_name)
            ds_ns = str(raw_ds.get("namespace") or "") or None
            if detected_dialect is None and ds_ns:
                detected_dialect = infer_dialect_from_namespace(ds_ns)

            _register_alias(raw_name, canonical)
            cols, pii_cols, part_cols, tags, explicit_layer = _extract_dataset_details(raw_ds)
            layer = explicit_layer or _infer_dataset_layer(raw_name, tags, is_output=False)

            existing = nodes.get(canonical)
            if existing is None:
                nodes[canonical] = LineageNode(
                    name=canonical,
                    engine="raw",
                    layer=layer,
                    namespace=ds_ns,
                    tags=tags,
                    columns=cols,
                    partition_columns=part_cols,
                    pii_columns=pii_cols,
                )
            else:
                if explicit_layer is not None:
                    existing.layer = explicit_layer
                _extend_unique(existing.columns, cols)
                _extend_unique(existing.pii_columns, pii_cols)
                _extend_unique(existing.partition_columns, part_cols)
                _extend_unique(existing.tags, tags)
                if existing.namespace is None and ds_ns is not None:
                    existing.namespace = ds_ns
            if canonical not in input_names:
                input_names.append(canonical)

        for raw_ds in event.get("outputs") or []:
            if not isinstance(raw_ds, dict):
                continue
            raw_name = str(raw_ds.get("name") or "").strip()
            if not raw_name:
                continue
            canonical = normalize_dataset_name(raw_name)
            ds_ns = str(raw_ds.get("namespace") or "") or None
            if detected_dialect is None and ds_ns:
                detected_dialect = infer_dialect_from_namespace(ds_ns)

            _register_alias(raw_name, canonical)
            cols, pii_cols, part_cols, tags, explicit_layer = _extract_dataset_details(raw_ds)
            layer = explicit_layer or _infer_dataset_layer(raw_name, tags, is_output=True)
            upstream = [name for name in input_names if name != canonical]

            existing = nodes.get(canonical)
            if existing is None:
                nodes[canonical] = LineageNode(
                    name=canonical,
                    engine=job_engine,  # type: ignore[arg-type]
                    layer=layer,
                    file_path=job_file_path,
                    namespace=ds_ns or job_ns,
                    tags=tags,
                    columns=cols,
                    partition_columns=part_cols,
                    pii_columns=pii_cols,
                    upstream=upstream,
                )
            else:
                if existing.engine == "raw":
                    existing.engine = job_engine  # type: ignore[assignment]
                if explicit_layer is not None or (
                    existing.layer in ("raw", "external", "dbt")
                    and layer not in ("raw", "external", "dbt")
                ):
                    existing.layer = layer
                if existing.file_path is None and job_file_path is not None:
                    existing.file_path = job_file_path
                if existing.namespace is None:
                    existing.namespace = ds_ns or job_ns
                _extend_unique(existing.columns, cols)
                _extend_unique(existing.pii_columns, pii_cols)
                _extend_unique(existing.partition_columns, part_cols)
                _extend_unique(existing.tags, tags)
                _extend_unique(existing.upstream, upstream)

    for name, node in nodes.items():
        for up in node.upstream:
            up_node = nodes.get(up)
            if up_node is not None and name not in up_node.downstream:
                up_node.downstream.append(name)

    return OpenLineageGraphData(nodes=nodes, aliases=aliases, sql_dialect=detected_dialect)


def _parse_event_time(raw_ts: Any) -> datetime:
    if isinstance(raw_ts, datetime):
        return raw_ts.astimezone(UTC) if raw_ts.tzinfo else raw_ts.replace(tzinfo=UTC)
    if isinstance(raw_ts, str) and raw_ts.strip():
        candidate = raw_ts.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        parsed = datetime.fromisoformat(candidate)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


def parse_openlineage_fail_alerts(
    source: Path | str | dict[str, Any] | Sequence[dict[str, Any]],
) -> list[RawAlert]:
    """Normalize OpenLineage `FAIL`/`ABORT` RunEvents into canonical `RawAlert` records."""
    events = load_openlineage_events(source)
    alerts: list[RawAlert] = []

    for ev_idx, event in enumerate(events):
        event_type = str(event.get("eventType") or "").strip().upper()
        if event_type not in {"FAIL", "ABORT"}:
            continue

        run = event.get("run") if isinstance(event.get("run"), dict) else {}
        job = event.get("job") if isinstance(event.get("job"), dict) else {}
        run_id = str(run.get("runId") or f"run-{ev_idx}")
        job_name = str(job.get("name") or "unknown_job")
        job_ns = str(job.get("namespace") or "")

        run_facets = run.get("facets") if isinstance(run.get("facets"), dict) else {}
        err_facet = (
            run_facets.get("errorMessage")
            if isinstance(run_facets.get("errorMessage"), dict)
            else {}
        )
        err_msg = str(err_facet.get("message") or "").strip()
        stack_trace = str(err_facet.get("stackTrace") or "").strip()
        prog_lang = str(err_facet.get("programmingLanguage") or "").strip()

        combined_err = f"{err_msg}\n{stack_trace}"
        severity = "critical" if _CRITICAL_ERROR_RE.search(combined_err) else "high"
        ts = _parse_event_time(event.get("eventTime"))

        summary = err_msg or f"OpenLineage {event_type} for job {job_name}"
        outputs = [d for d in (event.get("outputs") or []) if isinstance(d, dict) and d.get("name")]
        inputs = [d for d in (event.get("inputs") or []) if isinstance(d, dict) and d.get("name")]

        target_assets = (
            [normalize_dataset_name(str(out["name"])) for out in outputs]
            if outputs
            else [normalize_dataset_name(job_name) or job_name]
        )

        for out_idx, asset in enumerate(target_assets):
            raw_asset_name = (
                str(outputs[out_idx]["name"]) if out_idx < len(outputs) else job_name
            )
            seed = f"{run_id}|{job_ns}|{job_name}|{raw_asset_name}|{event.get('eventTime') or ev_idx}"
            alert_id = f"ol-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"
            alerts.append(
                RawAlert(
                    alert_id=alert_id,
                    asset=asset,
                    timestamp=ts,
                    severity=severity,
                    alert_type="openlineage_fail",
                    summary=summary,
                    metadata={
                        "event_type": event_type,
                        "run_id": run_id,
                        "job_name": job_name,
                        "job_namespace": job_ns,
                        "raw_asset": raw_asset_name,
                        "programming_language": prog_lang,
                        "stack_trace": stack_trace[:1000] if stack_trace else "",
                        "inputs": [
                            normalize_dataset_name(str(inp["name"])) for inp in inputs
                        ],
                    },
                )
            )

    return alerts
