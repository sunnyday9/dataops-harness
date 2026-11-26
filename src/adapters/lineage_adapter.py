from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

import yaml

from src.adapters.base import (
    AffectedSubgraph,
    LakehousePaths,
    LineageAdapterBase,
    LineageEdge,
    LineageNode,
    LineageSubgraph,
    LineageTestRef,
)


def _normalize_workspace_path(raw_path: str) -> str:
    cleaned = raw_path.replace("\\", "/").lstrip("./")
    return cleaned.removeprefix("workspace/")


class LineageAdapter(LineageAdapterBase):
    """Cross-engine DAG unifying raw, PySpark Delta silver, and dbt-duckdb assets."""

    def __init__(
        self,
        spark_lineage_path: Path,
        dbt_manifest_path: Path,
    ) -> None:
        self.spark_lineage_path = spark_lineage_path.resolve()
        self.dbt_manifest_path = dbt_manifest_path.resolve()
        self._nodes: dict[str, LineageNode] = {}
        self._file_to_assets: dict[str, list[str]] = defaultdict(list)
        self._macro_to_models: dict[str, list[str]] = defaultdict(list)
        self._load()

    @classmethod
    def from_workspace(cls, workspace_dir: Path) -> LineageAdapter:
        paths = LakehousePaths.from_workspace(workspace_dir)
        return cls(
            spark_lineage_path=paths.spark_jobs_dir / "lineage.yaml",
            dbt_manifest_path=paths.dbt_project_dir / "target" / "manifest.json",
        )

    def _register_file_index(self, file_path: str | None, asset_name: str) -> None:
        if not file_path:
            return
        normalized = _normalize_workspace_path(file_path)
        if asset_name not in self._file_to_assets[normalized]:
            self._file_to_assets[normalized].append(asset_name)

    def _load(self) -> None:
        spark_raw: dict[str, Any] = {}
        if self.spark_lineage_path.is_file():
            loaded = yaml.safe_load(self.spark_lineage_path.read_text(encoding="utf-8"))
            spark_raw = (loaded or {}).get("assets", {})

        for name, spec in spark_raw.items():
            engine = spec.get("engine", "spark")
            file_path = spec.get("job_path") or spec.get("path")
            normalized_file = (
                _normalize_workspace_path(file_path) if file_path else None
            )
            node = LineageNode(
                name=name,
                engine="raw" if engine == "raw" else "spark",
                layer=str(spec.get("layer", engine)),
                file_path=normalized_file,
                columns=list(spec.get("columns") or []),
                partition_columns=list(spec.get("partition_columns") or []),
                pii_columns=list(spec.get("pii_columns") or []),
                upstream=list(spec.get("upstream") or []),
            )
            self._nodes[name] = node
            self._register_file_index(normalized_file, name)

        if self.dbt_manifest_path.is_file():
            manifest = json.loads(self.dbt_manifest_path.read_text(encoding="utf-8"))
            self._load_dbt_manifest(manifest)

        # Build reverse downstream adjacency cleanly from upstream lists.
        for name, node in self._nodes.items():
            for up in node.upstream:
                up_node = self._nodes.get(up)
                if up_node is not None and name not in up_node.downstream:
                    up_node.downstream.append(name)

    def _load_dbt_manifest(self, manifest: dict[str, Any]) -> None:
        nodes = manifest.get("nodes") or {}
        macros = manifest.get("macros") or {}
        uid_to_name: dict[str, str] = {}

        macro_paths: dict[str, str] = {}
        for macro_uid, macro_meta in macros.items():
            orig = macro_meta.get("original_file_path")
            if orig:
                macro_paths[macro_uid] = f"dbt_project/{_normalize_workspace_path(orig)}"

        model_nodes = [
            (uid, meta)
            for uid, meta in nodes.items()
            if meta.get("resource_type") == "model"
        ]
        for uid, meta in model_nodes:
            uid_to_name[uid] = str(meta["name"])

        for _uid, meta in model_nodes:
            name = str(meta["name"])
            orig_path = str(meta.get("original_file_path") or "")
            file_path = (
                f"dbt_project/{_normalize_workspace_path(orig_path)}"
                if orig_path
                else None
            )
            layer = self._infer_dbt_layer(orig_path)

            upstream_assets: list[str] = []
            node_meta = meta.get("meta") or (meta.get("config") or {}).get("meta") or {}
            spark_upstream = node_meta.get("upstream_spark_table")
            if isinstance(spark_upstream, str) and spark_upstream in self._nodes:
                upstream_assets.append(spark_upstream)

            depends_on = meta.get("depends_on") or {}
            for dep_uid in depends_on.get("nodes") or []:
                dep_name = uid_to_name.get(dep_uid)
                if dep_name and dep_name not in upstream_assets:
                    upstream_assets.append(dep_name)

            for macro_uid in depends_on.get("macros") or []:
                m_path = macro_paths.get(macro_uid)
                if m_path and name not in self._macro_to_models[m_path]:
                    self._macro_to_models[m_path].append(name)

            columns = list((meta.get("columns") or {}).keys())
            self._nodes[name] = LineageNode(
                name=name,
                engine="dbt",
                layer=layer,
                file_path=file_path,
                columns=columns,
                upstream=upstream_assets,
            )
            self._register_file_index(file_path, name)

            patch_path = str(meta.get("patch_path") or "")
            if "://" in patch_path:
                schema_rel = patch_path.split("://", 1)[1]
                self._register_file_index(f"dbt_project/{schema_rel}", name)

        for test_meta in nodes.values():
            if test_meta.get("resource_type") != "test":
                continue
            generic_meta = test_meta.get("test_metadata") or {}
            test_type = str(generic_meta.get("name") or "singular")
            col_name = test_meta.get("column_name")
            test_orig = test_meta.get("original_file_path")
            test_path = (
                f"dbt_project/{_normalize_workspace_path(str(test_orig))}"
                if test_orig
                else None
            )
            test_ref = LineageTestRef(
                name=str(test_meta.get("name") or ""),
                test_type=test_type,
                column_name=str(col_name) if col_name else None,
                file_path=test_path,
            )
            for dep_uid in (test_meta.get("depends_on") or {}).get("nodes") or []:
                model_name = uid_to_name.get(dep_uid)
                if model_name and model_name in self._nodes:
                    self._nodes[model_name].tests.append(test_ref)
                    if test_type == "singular" and test_path:
                        self._register_file_index(test_path, model_name)

    @staticmethod
    def _infer_dbt_layer(orig_path: str) -> str:
        norm = orig_path.replace("\\", "/")
        if "/staging/" in norm or norm.startswith("models/staging/"):
            return "staging"
        if "/intermediate/" in norm or norm.startswith("models/intermediate/"):
            return "intermediate"
        if "/marts/" in norm or norm.startswith("models/marts/"):
            return "mart"
        return "dbt"

    def get_node(self, table_name: str) -> LineageNode:
        if table_name not in self._nodes:
            raise KeyError(f"Unknown asset in lineage graph: {table_name}")
        return self._nodes[table_name]

    def all_nodes(self) -> dict[str, LineageNode]:
        return dict(self._nodes)

    def _bfs_distances(
        self,
        start: str,
        step_attr: Literal["upstream", "downstream"],
        max_depth: int | None = None,
    ) -> dict[str, int]:
        distances: dict[str, int] = {start: 0}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        while queue:
            current, dist = queue.popleft()
            if max_depth is not None and dist >= max_depth:
                continue
            neighbors = getattr(self.get_node(current), step_attr)
            for nxt in neighbors:
                if nxt not in distances:
                    distances[nxt] = dist + 1
                    queue.append((nxt, dist + 1))
        return distances

    def query_lineage(
        self,
        table_name: str,
        direction: Literal["upstream", "downstream", "both"] = "both",
        depth: int = 3,
    ) -> LineageSubgraph:
        self.get_node(table_name)
        reachable: set[str] = {table_name}
        if direction in ("upstream", "both"):
            reachable.update(self._bfs_distances(table_name, "upstream", depth).keys())
        if direction in ("downstream", "both"):
            reachable.update(self._bfs_distances(table_name, "downstream", depth).keys())

        nodes = {name: self._nodes[name] for name in sorted(reachable)}
        edges: list[LineageEdge] = []
        for src in sorted(reachable):
            for dst in self._nodes[src].downstream:
                if dst in reachable:
                    edges.append(LineageEdge(source=src, target=dst))

        return LineageSubgraph(
            root=table_name,
            direction=direction,
            depth=depth,
            nodes=nodes,
            edges=edges,
        )

    def shared_ancestors(self, table_names: list[str]) -> list[str]:
        if not table_names:
            return []
        dist_maps = [self._bfs_distances(name, "upstream") for name in table_names]
        common = set(dist_maps[0].keys())
        for dmap in dist_maps[1:]:
            common &= set(dmap.keys())
        return sorted(
            common,
            key=lambda anc: (
                max(dmap[anc] for dmap in dist_maps),
                min(dmap[anc] for dmap in dist_maps),
                anc,
            ),
        )

    def lowest_common_ancestors(self, table_names: list[str]) -> list[str]:
        ordered_common = self.shared_ancestors(table_names)
        if not ordered_common:
            return []
        common_set = set(ordered_common)
        dist_maps = [self._bfs_distances(name, "upstream") for name in table_names]

        lcas: list[str] = []
        for candidate in ordered_common:
            descendants = set(self._bfs_distances(candidate, "downstream").keys())
            descendants.discard(candidate)
            if not (descendants & common_set):
                lcas.append(candidate)

        return sorted(
            lcas,
            key=lambda anc: (max(dmap[anc] for dmap in dist_maps), anc),
        )

    def affected_downstream(self, changed_paths: list[str]) -> AffectedSubgraph:
        seeds: set[str] = set()
        normalized_inputs = [_normalize_workspace_path(p) for p in changed_paths]

        for norm in normalized_inputs:
            if norm in self._file_to_assets:
                seeds.update(self._file_to_assets[norm])
            if norm in self._macro_to_models:
                seeds.update(self._macro_to_models[norm])
            if norm == "spark_jobs/spark_config.py":
                seeds.update(
                    name
                    for name, node in self._nodes.items()
                    if node.engine == "spark"
                )
            if norm in ("dbt_project/dbt_project.yml", "dbt_project/profiles.yml"):
                seeds.update(
                    name
                    for name, node in self._nodes.items()
                    if node.engine == "dbt"
                )

        affected_assets: set[str] = set()
        for seed in seeds:
            affected_assets.update(self._bfs_distances(seed, "downstream").keys())

        ordered_assets = sorted(affected_assets)
        spark_jobs: list[str] = []
        dbt_models: list[str] = []
        for asset in ordered_assets:
            node = self._nodes[asset]
            if node.engine == "spark" and node.file_path and node.file_path not in spark_jobs:
                spark_jobs.append(node.file_path)
            elif node.engine == "dbt" and node.name not in dbt_models:
                dbt_models.append(node.name)

        return AffectedSubgraph(
            changed_paths=normalized_inputs,
            spark_jobs=spark_jobs,
            dbt_models=dbt_models,
            affected_assets=ordered_assets,
        )
