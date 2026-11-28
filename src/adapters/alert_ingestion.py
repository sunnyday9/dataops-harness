from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.adapters.base import LineageAdapterBase
from src.adapters.openlineage import normalize_dataset_name, parse_openlineage_fail_alerts
from src.core.aggregator import AggregatedIncident, AlertStormAggregator, RawAlert


def _load_json_dict(source: Path | str | dict[str, Any], *, label: str) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, Path):
        resolved = source.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{label} file not found: {source}")
        raw_text = resolved.read_text(encoding="utf-8")
    elif isinstance(source, str):
        raw_text = source
    else:
        raise TypeError(f"{label} must be a Path, JSON string, or dict.")

    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {label}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise TypeError(f"{label} root payload must be a JSON object.")
    return loaded


def _parse_iso_ts(raw_ts: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(raw_ts, datetime):
        return raw_ts.astimezone(UTC) if raw_ts.tzinfo else raw_ts.replace(tzinfo=UTC)
    if isinstance(raw_ts, str) and raw_ts.strip():
        candidate = raw_ts.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return fallback or datetime.now(UTC)


def _extract_result_timestamp(entry: dict[str, Any], default_ts: datetime) -> datetime:
    timings = entry.get("timing")
    if isinstance(timings, list):
        for phase in reversed(timings):
            if isinstance(phase, dict):
                ts_val = phase.get("completed_at") or phase.get("started_at")
                if ts_val:
                    return _parse_iso_ts(ts_val, fallback=default_ts)
    return default_ts


def _infer_asset_from_test_uid(
    uid: str, lineage: LineageAdapterBase | None = None
) -> str:
    parts = uid.split(".")
    if len(parts) >= 4 and parts[0] == "unit_test":
        raw_model = parts[2]
        if lineage is not None:
            resolved = lineage.resolve_asset_name(raw_model)
            if resolved:
                return resolved
        return raw_model

    body = parts[2] if len(parts) >= 3 else uid
    stripped = body
    for prefix in (
        "not_null_",
        "unique_",
        "accepted_values_",
        "relationships_",
        "dbt_utils_",
        "dbt_expectations_",
    ):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :]
            break

    if lineage is not None:
        all_nodes_fn = getattr(lineage, "all_nodes", None)
        known_nodes = all_nodes_fn() if callable(all_nodes_fn) else {}
        all_aliases_fn = getattr(lineage, "all_aliases", None)
        known_aliases = all_aliases_fn() if callable(all_aliases_fn) else {}
        candidates: dict[str, str] = {}
        if isinstance(known_nodes, dict):
            for node_name in known_nodes:
                candidates[node_name] = node_name
        if isinstance(known_aliases, dict):
            for alias_key, canonical in known_aliases.items():
                if "." not in alias_key:
                    candidates.setdefault(alias_key, str(canonical))

        stripped_lower = stripped.lower()
        for surface in sorted(candidates, key=lambda s: (-len(s), s)):
            surf_lower = surface.lower()
            if stripped_lower == surf_lower or stripped_lower.startswith(f"{surf_lower}_"):
                canonical = candidates[surface]
                resolved = lineage.resolve_asset_name(canonical)
                return resolved or canonical

    return stripped


def ingest_openlineage_alerts(
    source: Path | str | dict[str, Any] | Sequence[dict[str, Any]],
) -> list[RawAlert]:
    """Normalize OpenLineage `FAIL`/`ABORT` events into canonical `RawAlert` records."""
    return parse_openlineage_fail_alerts(source)


def ingest_dbt_run_results(
    run_results: Path | str | dict[str, Any],
    *,
    manifest: Path | str | dict[str, Any] | None = None,
    lineage: LineageAdapterBase | None = None,
    include_warnings: bool = False,
) -> list[RawAlert]:
    """Normalize failing entries in dbt `run_results.json` into canonical `RawAlert` records."""
    rr_data = _load_json_dict(run_results, label="dbt run_results")
    manifest_data = (
        _load_json_dict(manifest, label="dbt manifest") if manifest is not None else None
    )

    manifest_nodes: dict[str, Any] = {}
    manifest_sources: dict[str, Any] = {}
    manifest_unit_tests: dict[str, Any] = {}
    if manifest_data is not None:
        raw_nodes = manifest_data.get("nodes")
        raw_sources = manifest_data.get("sources")
        raw_unit_tests = manifest_data.get("unit_tests")
        if isinstance(raw_nodes, dict):
            manifest_nodes = raw_nodes
        if isinstance(raw_sources, dict):
            manifest_sources = raw_sources
        if isinstance(raw_unit_tests, dict):
            manifest_unit_tests = raw_unit_tests

    rr_meta = rr_data.get("metadata") if isinstance(rr_data.get("metadata"), dict) else {}
    invocation_id = str(rr_meta.get("invocation_id") or "")
    default_ts = _parse_iso_ts(rr_meta.get("generated_at"))

    target_statuses = {"fail", "error"}
    if include_warnings:
        target_statuses.add("warn")

    alerts: list[RawAlert] = []
    for entry in rr_data.get("results") or []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status not in target_statuses:
            continue

        uid = str(entry.get("unique_id") or "").strip()
        m_node = (
            manifest_nodes.get(uid)
            or manifest_sources.get(uid)
            or manifest_unit_tests.get(uid)
        )
        m_node_dict = m_node if isinstance(m_node, dict) else {}
        resource_type = str(
            m_node_dict.get("resource_type") or (uid.split(".", 1)[0] if "." in uid else "model")
        )

        target_assets: list[str] = []
        test_name = str(m_node_dict.get("name") or "")
        column_name = m_node_dict.get("column_name")

        if resource_type in {"test", "unit_test"}:
            depends_nodes = (m_node_dict.get("depends_on") or {}).get("nodes") or []
            for dep_uid in depends_nodes:
                dep_obj = manifest_nodes.get(dep_uid) or manifest_sources.get(dep_uid)
                if isinstance(dep_obj, dict) and dep_obj.get("name"):
                    dep_name = str(dep_obj["name"])
                    if dep_name not in target_assets:
                        target_assets.append(dep_name)
            if not target_assets:
                relation = entry.get("relation_name")
                if isinstance(relation, str) and relation.strip() and "_test" not in relation:
                    normalized_rel = normalize_dataset_name(relation)
                    if lineage is not None:
                        normalized_rel = (
                            lineage.resolve_asset_name(normalized_rel) or normalized_rel
                        )
                    target_assets.append(normalized_rel)
                else:
                    target_assets.append(_infer_asset_from_test_uid(uid, lineage=lineage))
        else:
            if m_node_dict.get("name"):
                target_assets.append(str(m_node_dict["name"]))
            elif isinstance(entry.get("relation_name"), str) and entry["relation_name"].strip():
                normalized_rel = normalize_dataset_name(str(entry["relation_name"]))
                if lineage is not None:
                    normalized_rel = lineage.resolve_asset_name(normalized_rel) or normalized_rel
                target_assets.append(normalized_rel)
            elif uid:
                short_uid = uid.rsplit(".", 1)[-1]
                if lineage is not None:
                    short_uid = lineage.resolve_asset_name(short_uid) or short_uid
                target_assets.append(short_uid)
            else:
                target_assets.append("unknown_dbt_asset")

        failures = entry.get("failures")
        raw_message = str(entry.get("message") or "").strip()
        ts = _extract_result_timestamp(entry, default_ts)
        alert_type = "dbt_test_failure" if resource_type in {"test", "unit_test"} else "dbt_run_error"

        if status == "warn":
            severity = "medium"
        elif (
            status == "error" and resource_type not in {"test", "unit_test"}
        ) or "fatal" in raw_message.lower():
            severity = "critical"
        else:
            severity = "high"

        label = test_name or uid or alert_type
        detail_bits = [f"status={status}"]
        if failures is not None:
            detail_bits.append(f"failures={failures}")
        if raw_message:
            detail_bits.append(raw_message)
        summary = f"{label} ({', '.join(detail_bits)})"

        for asset in target_assets:
            seed = f"{invocation_id}|{uid}|{asset}|{status}"
            alert_id = f"dbt-{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}"
            alerts.append(
                RawAlert(
                    alert_id=alert_id,
                    asset=asset,
                    timestamp=ts,
                    severity=severity,
                    alert_type=alert_type,
                    summary=summary,
                    metadata={
                        "unique_id": uid,
                        "resource_type": resource_type,
                        "status": status,
                        "failures": failures,
                        "invocation_id": invocation_id,
                        "column_name": str(column_name) if column_name else None,
                        "test_name": test_name or None,
                    },
                )
            )

    return alerts


def ingest_text_alert(
    text: str,
    *,
    asset: str | None = None,
    lineage: LineageAdapterBase | None = None,
    severity: str = "high",
    alert_id: str | None = None,
    timestamp: datetime | None = None,
) -> list[RawAlert]:
    """Normalize a free-form failure description into canonical `RawAlert` records."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Alert description text must not be empty.")

    ts = _parse_iso_ts(timestamp) if timestamp is not None else datetime.now(UTC)

    matched_assets: list[str] = []
    if asset is not None and asset.strip():
        raw_explicit = asset.strip()
        if lineage is not None:
            resolved_explicit = lineage.resolve_asset_name(raw_explicit)
            matched_assets = [resolved_explicit or raw_explicit]
        else:
            matched_assets = [raw_explicit]
    else:
        all_nodes_fn = getattr(lineage, "all_nodes", None)
        known_nodes = all_nodes_fn() if callable(all_nodes_fn) else {}
        all_aliases_fn = getattr(lineage, "all_aliases", None)
        known_aliases = all_aliases_fn() if callable(all_aliases_fn) else {}
        surface_map: dict[str, str] = {}
        if isinstance(known_nodes, dict):
            for name in known_nodes:
                surface_map[name] = name
        if isinstance(known_aliases, dict):
            for alias_key, canonical in known_aliases.items():
                if not isinstance(alias_key, str) or not alias_key.strip():
                    continue
                resolved = (
                    lineage.resolve_asset_name(canonical)
                    if lineage is not None
                    else canonical
                )
                target = resolved or str(canonical)
                surface_map.setdefault(alias_key, target)

        if surface_map:
            hits: list[tuple[int, int, str]] = []
            for surface in sorted(surface_map.keys(), key=lambda n: (-len(n), n)):
                match = re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(surface)}(?![A-Za-z0-9_])",
                    cleaned,
                    re.IGNORECASE,
                )
                if match is not None:
                    hits.append((match.start(), -len(surface), surface_map[surface]))
            hits.sort()
            for _pos, _neg_len, target_name in hits:
                if target_name not in matched_assets:
                    matched_assets.append(target_name)

    if not matched_assets:
        matched_assets = ["unknown_asset"]

    alerts: list[RawAlert] = []
    for idx, target_asset in enumerate(matched_assets):
        if alert_id:
            resolved_id = alert_id if len(matched_assets) == 1 else f"{alert_id}-{idx + 1}"
        else:
            digest = hashlib.sha1(f"{target_asset}|{cleaned}".encode()).hexdigest()[:12]
            resolved_id = f"txt-{digest}"
        alerts.append(
            RawAlert(
                alert_id=resolved_id,
                asset=target_asset,
                timestamp=ts,
                severity=severity,
                alert_type="text_alert",
                summary=cleaned,
                metadata={"source": "text"},
            )
        )
    return alerts


def ingest_and_converge(
    *,
    lineage: LineageAdapterBase,
    openlineage_events: Path | str | dict[str, Any] | Sequence[dict[str, Any]] | None = None,
    dbt_run_results: Path | str | dict[str, Any] | None = None,
    dbt_manifest: Path | str | dict[str, Any] | None = None,
    text: str | None = None,
    text_asset: str | None = None,
    text_timestamp: datetime | None = None,
    raw_alerts: Sequence[RawAlert] = (),
    window_seconds: int | None = None,
) -> tuple[list[RawAlert], list[AggregatedIncident]]:
    """Normalize heterogeneous failure signals and converge them into deduplicated `Incident`s."""
    alerts: list[RawAlert] = []
    if openlineage_events is not None:
        alerts.extend(ingest_openlineage_alerts(openlineage_events))
    if dbt_run_results is not None:
        alerts.extend(
            ingest_dbt_run_results(dbt_run_results, manifest=dbt_manifest, lineage=lineage)
        )
    alerts.extend(raw_alerts)
    if text is not None and text.strip():
        anchor_ts = text_timestamp
        if anchor_ts is None and alerts:
            anchor_ts = max(alert.timestamp for alert in alerts)
        alerts.extend(
            ingest_text_alert(
                text,
                asset=text_asset,
                lineage=lineage,
                timestamp=anchor_ts,
            )
        )

    aggregator = AlertStormAggregator(lineage=lineage, window_seconds=window_seconds)
    incidents = aggregator.ingest(alerts)
    return alerts, incidents
