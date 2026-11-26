from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from src.adapters.base import LineageAdapterBase
from src.core.context import IncidentInput, RunState, UserContext
from src.core.policy import load_policies

_SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "low": 1,
    "p4": 1,
    "medium": 2,
    "warning": 2,
    "warn": 2,
    "p3": 2,
    "high": 3,
    "error": 3,
    "p2": 3,
    "critical": 4,
    "p1": 4,
    "fatal": 5,
    "p0": 5,
}
_DEFAULT_LINEAGE_DEPTH = 10


def _to_utc(ts: datetime) -> datetime:
    """Normalize naive or aware timestamps to UTC for safe ordering and deltas."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _stable_incident_id(alert_ids: Sequence[str]) -> str:
    """Build a deterministic short Incident identifier from correlated alert IDs."""
    digest = hashlib.sha1("|".join(alert_ids).encode("utf-8")).hexdigest()[:10]
    return f"inc-{digest}"


class RawAlert(BaseModel):
    """Single incoming telemetry Alert emitted by a job, test, or monitor."""

    alert_id: str
    asset: str
    timestamp: datetime
    severity: str = "high"
    alert_type: str = "anomaly"
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("alert_id", "asset")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field must not be empty or whitespace.")
        return cleaned


class AggregatedIncident(BaseModel):
    """Deduplicated Incident converged from one or more lineage-related RawAlerts."""

    incident_id: str = ""
    alert_ids: list[str]
    alerts: list[RawAlert]
    affected_assets: list[str]
    root_cause_candidates: list[str]
    shared_ancestors: list[str] = Field(default_factory=list)
    window_start: datetime
    window_end: datetime
    severity: str = "high"

    @model_validator(mode="after")
    def _ensure_incident_id(self) -> AggregatedIncident:
        if not self.incident_id:
            self.incident_id = _stable_incident_id(self.alert_ids)
        return self

    def to_incident_input(
        self,
        title: str | None = None,
        summary: str | None = None,
    ) -> IncidentInput:
        """Convert this converged bundle into the core `IncidentInput` loop contract."""
        assets_str = ", ".join(self.affected_assets) or "unknown"
        if title is not None:
            resolved_title = title
        elif len(self.affected_assets) > 1 and self.root_cause_candidates:
            suspects = ", ".join(self.root_cause_candidates)
            resolved_title = (
                f"[{self.severity.upper()}] Alert storm on {assets_str} "
                f"(suspected root cause: {suspects})"
            )
        elif self.alerts and self.alerts[0].summary:
            resolved_title = (
                f"[{self.severity.upper()}] {assets_str}: {self.alerts[0].summary}"
            )
        else:
            resolved_title = f"[{self.severity.upper()}] Incident on {assets_str}"

        if summary is not None:
            resolved_summary = summary
        else:
            lines = [
                (
                    f"Converged {len(self.alerts)} alert(s) across "
                    f"[{self.window_start.isoformat()} .. {self.window_end.isoformat()}]."
                ),
                f"Affected assets: {assets_str}.",
                f"Root-cause candidates: {', '.join(self.root_cause_candidates) or 'none'}.",
            ]
            for alert in self.alerts:
                detail = alert.summary or "(no summary)"
                lines.append(
                    f"- [{alert.alert_id}] {alert.asset} "
                    f"({alert.alert_type}/{alert.severity}): {detail}"
                )
            resolved_summary = "\n".join(lines)

        advisory_ancestors = (
            list(self.shared_ancestors)
            if self.shared_ancestors
            else (list(self.root_cause_candidates) if len(self.affected_assets) > 1 else [])
        )
        return IncidentInput(
            incident_id=self.incident_id,
            title=resolved_title,
            summary=resolved_summary,
            alerts=[alert.model_dump(mode="json") for alert in self.alerts],
            affected_assets=list(self.affected_assets),
            shared_ancestors=advisory_ancestors,
        )


class IncidentLoopProtocol(Protocol):
    """Protocol matching `HarnessLoop.run` for single-Run Incident dispatch."""

    def run(
        self,
        incident: IncidentInput,
        user_context: UserContext,
        run_id: str | None = None,
        max_turns: int = 20,
    ) -> RunState: ...


class _UnionFind:
    """Disjoint-set structure with path compression and union by size."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._size = [1] * size

    def find(self, idx: int) -> int:
        while self._parent[idx] != idx:
            self._parent[idx] = self._parent[self._parent[idx]]
            idx = self._parent[idx]
        return idx

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]


class _LineageProbe:
    """Per-ingest cached wrapper around `LineageAdapterBase` safe for unknown assets."""

    def __init__(self, lineage: LineageAdapterBase) -> None:
        self._lineage = lineage
        self._known_cache: dict[str, bool] = {}
        self._resolved_name_cache: dict[str, str] = {}
        self._neighbors_cache: dict[tuple[str, Literal["upstream", "downstream"]], set[str]] = {}
        self._pair_lca_cache: dict[tuple[str, str], list[str]] = {}

    def is_known(self, asset: str) -> bool:
        if asset not in self._known_cache:
            try:
                self._lineage.query_lineage(asset, direction="both", depth=1)
                self._known_cache[asset] = True
            except KeyError:
                self._known_cache[asset] = False
        return self._known_cache[asset]

    def normalize_asset(self, raw_asset: str) -> str:
        if raw_asset in self._resolved_name_cache:
            return self._resolved_name_cache[raw_asset]

        candidates = [raw_asset]
        lowered = raw_asset.lower()
        if lowered != raw_asset:
            candidates.append(lowered)
        if "." in raw_asset:
            short = raw_asset.rsplit(".", 1)[-1]
            candidates.append(short)
            if short.lower() != short:
                candidates.append(short.lower())

        resolved = raw_asset
        for candidate in candidates:
            if self.is_known(candidate):
                resolved = candidate
                break

        self._resolved_name_cache[raw_asset] = resolved
        return resolved

    def neighborhood(
        self, asset: str, direction: Literal["upstream", "downstream"]
    ) -> set[str]:
        key = (asset, direction)
        if key not in self._neighbors_cache:
            if not self.is_known(asset):
                self._neighbors_cache[key] = set()
            else:
                subgraph = self._lineage.query_lineage(
                    asset, direction=direction, depth=_DEFAULT_LINEAGE_DEPTH
                )
                self._neighbors_cache[key] = set(subgraph.nodes.keys())
        return self._neighbors_cache[key]

    def lowest_common_ancestors(self, assets: list[str]) -> list[str]:
        if not assets or any(not self.is_known(a) for a in assets):
            return []
        if len(assets) == 2:
            pair = (assets[0], assets[1]) if assets[0] <= assets[1] else (assets[1], assets[0])
            if pair not in self._pair_lca_cache:
                self._pair_lca_cache[pair] = list(
                    self._lineage.lowest_common_ancestors(list(pair))
                )
            return self._pair_lca_cache[pair]
        return list(self._lineage.lowest_common_ancestors(assets))

    def shared_ancestors(self, assets: list[str]) -> list[str]:
        if not assets or any(not self.is_known(a) for a in assets):
            return []
        return list(self._lineage.shared_ancestors(assets))

    def is_ancestor_descendant(self, asset_a: str, asset_b: str) -> bool:
        if not (self.is_known(asset_a) and self.is_known(asset_b)):
            return False
        if asset_a in self.neighborhood(asset_b, "upstream"):
            return True
        if asset_b in self.neighborhood(asset_a, "upstream"):
            return True
        if asset_a in self.neighborhood(asset_b, "downstream"):
            return True
        if asset_b in self.neighborhood(asset_a, "downstream"):
            return True
        pair_lcas = self.lowest_common_ancestors([asset_a, asset_b])
        return asset_a in pair_lcas or asset_b in pair_lcas

    def are_convergent(self, asset_a: str, asset_b: str) -> bool:
        if asset_a == asset_b:
            return True
        if self.is_ancestor_descendant(asset_a, asset_b):
            return True
        return bool(self.lowest_common_ancestors([asset_a, asset_b]))


def _pick_severity(alerts: Sequence[RawAlert]) -> str:
    """Return the highest-severity label among alerts in a converged component."""
    best = alerts[0].severity
    best_rank = _SEVERITY_RANK.get(best.lower(), 2)
    for alert in alerts[1:]:
        rank = _SEVERITY_RANK.get(alert.severity.lower(), 2)
        if rank > best_rank:
            best = alert.severity
            best_rank = rank
    return best


class AlertStormAggregator:
    """Converges concurrent raw Alerts into deduplicated root-cause Incidents."""

    def __init__(
        self,
        lineage: LineageAdapterBase,
        window_seconds: int | None = None,
        policies_path: Path | None = None,
    ) -> None:
        resolved_window = (
            window_seconds
            if window_seconds is not None
            else load_policies(policies_path).aggregator.window_seconds
        )
        if resolved_window < 0:
            raise ValueError(f"window_seconds must be non-negative, got {resolved_window}")
        self._lineage = lineage
        self.window_seconds = resolved_window

    @property
    def lineage(self) -> LineageAdapterBase:
        """Lineage adapter used for Alert convergence and LCA resolution."""
        return self._lineage

    def _resolve_root_causes(
        self,
        distinct_assets: list[str],
        probe: _LineageProbe,
    ) -> tuple[list[str], list[str]]:
        if len(distinct_assets) == 1:
            return [distinct_assets[0]], []

        known_assets = [a for a in distinct_assets if probe.is_known(a)]
        if not known_assets:
            return list(distinct_assets), []

        alerted_set = set(known_assets)
        top_alerted = [
            asset
            for asset in known_assets
            if not ((probe.neighborhood(asset, "upstream") - {asset}) & alerted_set)
        ]

        lcas = probe.lowest_common_ancestors(known_assets)
        shared = probe.shared_ancestors(known_assets)

        if lcas:
            # Prefer an alerted upstream ancestor first if a single root alerted,
            # while retaining any additional LCA nodes cleanly.
            ordered_lcas = list(lcas)
            if len(top_alerted) == 1 and top_alerted[0] in ordered_lcas:
                root = top_alerted[0]
                candidates = [root, *(item for item in ordered_lcas if item != root)]
            elif len(top_alerted) == 1 and any(
                top_alerted[0] in probe.neighborhood(item, "downstream") for item in ordered_lcas
            ):
                candidates = [top_alerted[0], *(item for item in ordered_lcas if item != top_alerted[0])]
            else:
                candidates = ordered_lcas
        else:
            candidates = top_alerted or list(known_assets)

        ordered_shared = list(dict.fromkeys([*candidates, *shared]))
        return candidates, ordered_shared

    def ingest(self, alerts: Sequence[RawAlert]) -> list[AggregatedIncident]:
        """Group chronological RawAlerts within `window_seconds` via lineage convergence."""
        if not alerts:
            return []

        ordered = sorted(alerts, key=lambda alert: _to_utc(alert.timestamp))
        ordered_utc = [_to_utc(alert.timestamp) for alert in ordered]
        probe = _LineageProbe(self._lineage)
        norm_assets = [probe.normalize_asset(alert.asset) for alert in ordered]

        uf = _UnionFind(len(ordered))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                if (ordered_utc[j] - ordered_utc[i]).total_seconds() > self.window_seconds:
                    break
                if uf.find(i) == uf.find(j):
                    continue
                if probe.are_convergent(norm_assets[i], norm_assets[j]):
                    uf.union(i, j)

        groups: dict[int, list[int]] = {}
        for idx in range(len(ordered)):
            root = uf.find(idx)
            groups.setdefault(root, []).append(idx)

        incidents: list[AggregatedIncident] = []
        for indices in groups.values():
            comp_alerts = [ordered[idx] for idx in indices]
            distinct_assets = list(dict.fromkeys(norm_assets[idx] for idx in indices))
            root_causes, shared = self._resolve_root_causes(distinct_assets, probe)
            incidents.append(
                AggregatedIncident(
                    incident_id=_stable_incident_id([a.alert_id for a in comp_alerts]),
                    alert_ids=[a.alert_id for a in comp_alerts],
                    alerts=comp_alerts,
                    affected_assets=distinct_assets,
                    root_cause_candidates=root_causes,
                    shared_ancestors=shared,
                    window_start=comp_alerts[0].timestamp,
                    window_end=comp_alerts[-1].timestamp,
                    severity=_pick_severity(comp_alerts),
                )
            )
        return incidents

    def aggregate(self, alerts: Sequence[RawAlert]) -> list[AggregatedIncident]:
        """Alias for `ingest` converting RawAlerts into deduplicated AggregatedIncidents."""
        return self.ingest(alerts)

    def dispatch_runs(
        self,
        items: Sequence[RawAlert] | Sequence[AggregatedIncident],
        runner: IncidentLoopProtocol | Callable[[IncidentInput, UserContext], RunState],
        user_context: UserContext | None = None,
        max_turns: int | None = None,
    ) -> list[RunState]:
        """Dispatch exactly one troubleshooting Run per converged Incident."""
        if not items:
            return []

        if isinstance(items[0], RawAlert):
            incidents = self.ingest(items)  # type: ignore[arg-type]
        else:
            incidents = list(items)  # type: ignore[arg-type]

        ctx = user_context or UserContext()
        dispatched_ids: set[str] = set()
        results: list[RunState] = []

        for incident in incidents:
            if incident.incident_id in dispatched_ids:
                continue
            dispatched_ids.add(incident.incident_id)
            incident_input = incident.to_incident_input()

            run_fn = getattr(runner, "run", None)
            if callable(run_fn):
                if max_turns is not None:
                    state = run_fn(incident_input, ctx, max_turns=max_turns)
                else:
                    state = run_fn(incident_input, ctx)
            else:
                state = runner(incident_input, ctx)  # type: ignore[operator]
            results.append(state)

        return results
