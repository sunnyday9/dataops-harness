from __future__ import annotations

import heapq
import json
from collections import defaultdict, deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml

from src.adapters.base import (
    AffectedSubgraph,
    LineageAdapterBase,
    LineageEdge,
    LineageNode,
    LineageSubgraph,
    LineageTestRef,
)
from src.adapters.openlineage import extract_openlineage_lineage, load_openlineage_events
from src.core.policy import normalize_sql_dialect
from src.core.project import (
    ResolvedProject,
    discover_project,
    extract_dbt_node_partition_columns,
    extract_dbt_node_pii_columns,
    infer_dbt_layer,
)


def _normalize_workspace_path(raw_path: str) -> str:
    cleaned = raw_path.replace("\\", "/").lstrip("./")
    return cleaned.removeprefix("workspace/")


def _join_rel(prefix_rel: str, rel_path: str) -> str:
    norm = _normalize_workspace_path(rel_path)
    if not prefix_rel:
        return norm
    if norm == prefix_rel or norm.startswith(f"{prefix_rel}/"):
        return norm
    return f"{prefix_rel}/{norm}"


def _extend_unique(target: list[str], additions: Sequence[str]) -> None:
    seen = set(target)
    for item in additions:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            target.append(cleaned)
            seen.add(cleaned)


class LineageAdapter(LineageAdapterBase):
    """Cross-engine DAG unifying raw, Spark, dbt artifacts, and OpenLineage assets."""

    def __init__(
        self,
        spark_lineage_path: Path | None = None,
        dbt_manifest_path: Path | None = None,
        *,
        openlineage_paths: Sequence[Path] | None = None,
        dbt_project_rel: str = "dbt_project",
    ) -> None:
        self.spark_lineage_path = (
            spark_lineage_path.resolve() if spark_lineage_path is not None else None
        )
        self.dbt_manifest_path = (
            dbt_manifest_path.resolve() if dbt_manifest_path is not None else None
        )
        self.openlineage_paths = [Path(p).resolve() for p in (openlineage_paths or ())]
        self.dbt_project_rel = dbt_project_rel.strip("/").replace("\\", "/")
        self.sql_dialect: str | None = None
        self._nodes: dict[str, LineageNode] = {}
        self._lower_to_canonical: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._pending_spark_links: list[tuple[str, str]] = []
        self._file_to_assets: dict[str, list[str]] = defaultdict(list)
        self._macro_to_models: dict[str, list[str]] = defaultdict(list)
        self._load()

    @classmethod
    def from_project(cls, project: ResolvedProject) -> LineageAdapter:
        """Construct a `LineageAdapter` from a discovered `ResolvedProject`."""
        return cls(
            spark_lineage_path=project.spark_lineage_path,
            dbt_manifest_path=project.dbt_manifest_path,
            openlineage_paths=project.openlineage_paths,
            dbt_project_rel=project.dbt_project_rel,
        )

    @classmethod
    def from_workspace(cls, workspace_dir: Path) -> LineageAdapter:
        """Discover project artifacts inside `workspace_dir` and build a `LineageAdapter`."""
        project = discover_project(workspace_dir)
        return cls.from_project(project)

    def _register_file_index(self, file_path: str | None, asset_name: str) -> None:
        if not file_path:
            return
        normalized = _normalize_workspace_path(file_path)
        if asset_name not in self._file_to_assets[normalized]:
            self._file_to_assets[normalized].append(asset_name)

    def _register_alias(self, alias: str, canonical: str) -> None:
        cleaned = alias.strip()
        if cleaned:
            self._aliases.setdefault(cleaned, canonical)
            self._aliases.setdefault(cleaned.lower(), canonical)

    def _load(self) -> None:
        if self.spark_lineage_path is not None and self.spark_lineage_path.is_file():
            loaded = yaml.safe_load(self.spark_lineage_path.read_text(encoding="utf-8"))
            spark_raw: dict[str, Any] = (loaded or {}).get("assets", {})
            for name, spec in spark_raw.items():
                if not isinstance(spec, dict):
                    continue
                engine = spec.get("engine", "spark")
                file_path = spec.get("job_path") or spec.get("path")
                normalized_file = (
                    _normalize_workspace_path(str(file_path)) if file_path else None
                )
                node = LineageNode(
                    name=name,
                    engine="raw" if engine == "raw" else "spark",
                    layer=str(spec.get("layer", engine)),
                    file_path=normalized_file,
                    tags=[str(t) for t in (spec.get("tags") or [])],
                    columns=list(spec.get("columns") or []),
                    partition_columns=list(spec.get("partition_columns") or []),
                    pii_columns=list(spec.get("pii_columns") or []),
                    upstream=list(spec.get("upstream") or []),
                )
                self._nodes[name] = node
                self._lower_to_canonical.setdefault(name.lower(), name)
                self._register_alias(name, name)
                self._register_file_index(normalized_file, name)

        if self.dbt_manifest_path is not None and self.dbt_manifest_path.is_file():
            manifest = json.loads(self.dbt_manifest_path.read_text(encoding="utf-8"))
            self._load_dbt_manifest(manifest)

        if self.openlineage_paths:
            self._load_openlineage()

        for child_name, spark_upstream in self._pending_spark_links:
            resolved_child = self.resolve_asset_name(child_name)
            resolved_up = self.resolve_asset_name(spark_upstream)
            if (
                resolved_child is not None
                and resolved_up is not None
                and resolved_up != resolved_child
            ):
                _extend_unique(self._nodes[resolved_child].upstream, [resolved_up])

        # Build reverse downstream adjacency cleanly from canonicalized upstream lists.
        for name, node in self._nodes.items():
            canonical_ups: list[str] = []
            for up in node.upstream:
                resolved_up = self.resolve_asset_name(up) or up
                if resolved_up != name and resolved_up not in canonical_ups:
                    canonical_ups.append(resolved_up)
            node.upstream = canonical_ups
            for up in node.upstream:
                up_node = self._nodes.get(up)
                if up_node is not None and name not in up_node.downstream:
                    up_node.downstream.append(name)

    def _merge_or_create_node(self, candidate: LineageNode) -> LineageNode:
        existing_key = (
            candidate.name
            if candidate.name in self._nodes
            else self._lower_to_canonical.get(candidate.name.lower())
        )
        existing = self._nodes.get(existing_key) if existing_key is not None else None
        if existing is None:
            self._nodes[candidate.name] = candidate
            self._lower_to_canonical[candidate.name.lower()] = candidate.name
            self._register_alias(candidate.name, candidate.name)
            self._register_file_index(candidate.file_path, candidate.name)
            return candidate

        if candidate.name != existing.name:
            self._register_alias(candidate.name, existing.name)
        if existing.engine == "raw" and candidate.engine != "raw":
            existing.engine = candidate.engine
        if (existing.layer == "raw" and candidate.layer != "raw") or (
            existing.layer in ("raw", "dbt")
            and candidate.layer not in ("raw", "dbt", "external")
        ):
            existing.layer = candidate.layer
        if existing.file_path is None and candidate.file_path is not None:
            existing.file_path = candidate.file_path
            self._register_file_index(candidate.file_path, existing.name)
        if existing.namespace is None and candidate.namespace is not None:
            existing.namespace = candidate.namespace
        _extend_unique(existing.tags, candidate.tags)
        _extend_unique(existing.columns, candidate.columns)
        _extend_unique(existing.partition_columns, candidate.partition_columns)
        _extend_unique(existing.pii_columns, candidate.pii_columns)
        normalized_ups = [
            self.resolve_asset_name(u) or self._lower_to_canonical.get(u.lower(), u)
            for u in candidate.upstream
        ]
        _extend_unique(
            existing.upstream,
            [u for u in normalized_ups if u.lower() != existing.name.lower()],
        )
        return existing

    def _load_dbt_manifest(self, manifest: dict[str, Any]) -> None:
        adapter_type = (manifest.get("metadata") or {}).get("adapter_type")
        if isinstance(adapter_type, str) and adapter_type.strip():
            self.sql_dialect = normalize_sql_dialect(adapter_type)

        nodes = manifest.get("nodes") or {}
        sources = manifest.get("sources") or {}
        macros = manifest.get("macros") or {}
        uid_to_name: dict[str, str] = {}

        asset_nodes = [
            (uid, meta)
            for uid, meta in nodes.items()
            if isinstance(meta, dict)
            and meta.get("resource_type") in {"model", "seed", "snapshot"}
            and meta.get("name")
        ]
        model_names_lower = {str(meta["name"]).strip().lower() for _uid, meta in asset_nodes}

        macro_paths: dict[str, str] = {}
        for macro_uid, macro_meta in macros.items():
            if not isinstance(macro_meta, dict):
                continue
            orig = macro_meta.get("original_file_path")
            if orig:
                macro_paths[macro_uid] = _join_rel(self.dbt_project_rel, str(orig))

        for src_uid, src_meta in sources.items():
            if not isinstance(src_meta, dict):
                continue
            raw_name = str(src_meta.get("name") or "").strip()
            if not raw_name:
                continue
            source_name = str(src_meta.get("source_name") or "").strip()
            schema_name = str(src_meta.get("schema") or "").strip()
            if raw_name.lower() in model_names_lower:
                if source_name:
                    name = f"{source_name}.{raw_name}"
                elif schema_name:
                    name = f"{schema_name}.{raw_name}"
                else:
                    name = src_uid
            else:
                name = raw_name
            uid_to_name[src_uid] = name
            if source_name:
                self._register_alias(f"{source_name}.{raw_name}", name)
            if schema_name and schema_name.lower() != source_name.lower():
                self._register_alias(f"{schema_name}.{raw_name}", name)

            orig_path = str(src_meta.get("original_file_path") or "")
            file_path = _join_rel(self.dbt_project_rel, orig_path) if orig_path else None
            columns = list((src_meta.get("columns") or {}).keys())
            src_node = LineageNode(
                name=name,
                engine="raw",
                layer="raw",
                file_path=file_path,
                tags=[str(t) for t in (src_meta.get("tags") or [])],
                columns=columns,
                partition_columns=extract_dbt_node_partition_columns(src_meta),
                pii_columns=extract_dbt_node_pii_columns(src_meta),
            )
            self._merge_or_create_node(src_node)
            patch_path = str(src_meta.get("patch_path") or "")
            if "://" in patch_path:
                schema_rel = patch_path.split("://", 1)[1]
                self._register_file_index(_join_rel(self.dbt_project_rel, schema_rel), name)

        for uid, meta in asset_nodes:
            uid_to_name[uid] = str(meta["name"])

        for _uid, meta in asset_nodes:
            res_type = str(meta.get("resource_type") or "model")
            name = str(meta["name"])
            orig_path = str(meta.get("original_file_path") or "")
            file_path = _join_rel(self.dbt_project_rel, orig_path) if orig_path else None
            layer = infer_dbt_layer(meta, orig_path, name)
            engine: Literal["raw", "dbt"] = "raw" if res_type == "seed" else "dbt"

            upstream_assets: list[str] = []
            cfg_meta = (meta.get("config") or {}).get("meta") or {}
            direct_meta = meta.get("meta") or {}
            node_meta = {
                **(cfg_meta if isinstance(cfg_meta, dict) else {}),
                **(direct_meta if isinstance(direct_meta, dict) else {}),
            }
            spark_upstream = node_meta.get("upstream_spark_table")
            if isinstance(spark_upstream, str) and spark_upstream.strip():
                self._pending_spark_links.append((name, spark_upstream.strip()))
            elif isinstance(spark_upstream, list):
                for item in spark_upstream:
                    if isinstance(item, str) and item.strip():
                        self._pending_spark_links.append((name, item.strip()))

            depends_on = meta.get("depends_on") or {}
            for dep_uid in depends_on.get("nodes") or []:
                dep_name = uid_to_name.get(dep_uid)
                if dep_name and dep_name != name and dep_name not in upstream_assets:
                    upstream_assets.append(dep_name)

            for macro_uid in depends_on.get("macros") or []:
                m_path = macro_paths.get(macro_uid)
                if m_path and name not in self._macro_to_models[m_path]:
                    self._macro_to_models[m_path].append(name)

            schema_name = str(meta.get("schema") or "").strip()
            alias_name = str(meta.get("alias") or "").strip()
            if schema_name:
                self._register_alias(f"{schema_name}.{alias_name or name}", name)
            if alias_name and alias_name != name:
                self._register_alias(alias_name, name)

            columns = list((meta.get("columns") or {}).keys())
            dbt_node = LineageNode(
                name=name,
                engine=engine,
                layer=layer,
                file_path=file_path,
                tags=[str(t) for t in (meta.get("tags") or [])],
                columns=columns,
                partition_columns=extract_dbt_node_partition_columns(meta),
                pii_columns=extract_dbt_node_pii_columns(meta),
                upstream=upstream_assets,
            )
            self._merge_or_create_node(dbt_node)

            patch_path = str(meta.get("patch_path") or "")
            if "://" in patch_path:
                schema_rel = patch_path.split("://", 1)[1]
                self._register_file_index(_join_rel(self.dbt_project_rel, schema_rel), name)

        for test_meta in nodes.values():
            if not isinstance(test_meta, dict) or test_meta.get("resource_type") != "test":
                continue
            generic_meta = test_meta.get("test_metadata") or {}
            test_type = str(generic_meta.get("name") or "singular")
            col_name = test_meta.get("column_name")
            test_orig = test_meta.get("original_file_path")
            test_path = (
                _join_rel(self.dbt_project_rel, str(test_orig)) if test_orig else None
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

    def _load_openlineage(self) -> None:
        events: list[dict[str, Any]] = []
        for ol_path in self.openlineage_paths:
            if ol_path.exists():
                events.extend(load_openlineage_events(ol_path))
        if not events:
            return
        graph = extract_openlineage_lineage(events)
        if self.sql_dialect is None and graph.sql_dialect is not None:
            self.sql_dialect = graph.sql_dialect
        for node in graph.nodes.values():
            copy_node = node.model_copy(update={"downstream": []})
            self._merge_or_create_node(copy_node)
        for alias, canonical in graph.aliases.items():
            resolved = self.resolve_asset_name(canonical) or self._lower_to_canonical.get(
                canonical.lower()
            )
            if resolved is not None:
                self._register_alias(alias, resolved)

    @staticmethod
    def _infer_dbt_layer(orig_path: str) -> str:
        return infer_dbt_layer({}, orig_path)

    def resolve_asset_name(self, table_name: str) -> str | None:
        """Resolve an asset name or dotted FQDN alias to its canonical node key."""
        if table_name in self._nodes:
            return table_name
        if table_name in self._aliases and self._aliases[table_name] in self._nodes:
            return self._aliases[table_name]
        lowered = table_name.lower()
        if lowered in self._aliases and self._aliases[lowered] in self._nodes:
            return self._aliases[lowered]
        if "." in table_name:
            short = table_name.rsplit(".", 1)[-1]
            if short in self._nodes:
                return short
            if short.lower() in self._aliases and self._aliases[short.lower()] in self._nodes:
                return self._aliases[short.lower()]
        return None

    def get_node(self, table_name: str) -> LineageNode:
        resolved = self.resolve_asset_name(table_name)
        if resolved is None:
            raise KeyError(f"Unknown asset in lineage graph: {table_name}")
        return self._nodes[resolved]

    def all_nodes(self) -> dict[str, LineageNode]:
        return dict(self._nodes)

    def all_aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def topological_order(self, assets: list[str] | None = None) -> list[str]:
        """Return assets in deterministic upstream-to-downstream order via Kahn's algorithm."""
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        for name, node in self._nodes.items():
            unique_ups = {up for up in node.upstream if up in self._nodes and up != name}
            in_degree[name] = len(unique_ups)

        heap = [name for name, deg in in_degree.items() if deg == 0]
        heapq.heapify(heap)
        ordered: list[str] = []
        visited: set[str] = set()

        while heap:
            current = heapq.heappop(heap)
            ordered.append(current)
            visited.add(current)
            for nxt in sorted(self._nodes[current].downstream):
                if nxt not in in_degree or nxt in visited:
                    continue
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    heapq.heappush(heap, nxt)

        for name in sorted(self._nodes):
            if name not in visited:
                ordered.append(name)

        if assets is None:
            return ordered

        requested = {a for a in assets}
        result = [name for name in ordered if name in requested]
        for asset in assets:
            if asset not in result:
                result.append(asset)
        return result

    def spark_job_table_map(self) -> dict[str, str]:
        """Return `{normalized_job_path: asset_name}` for all discovered Spark assets."""
        mapping: dict[str, str] = {}
        for asset_name in self.topological_order():
            node = self._nodes[asset_name]
            if node.engine == "spark" and node.file_path:
                mapping[node.file_path] = asset_name
        return mapping

    def ordered_spark_jobs(self, spark_jobs: Sequence[str]) -> list[str]:
        """Order Spark script paths deterministically from upstream to downstream via lineage."""
        norm_inputs = [_normalize_workspace_path(p) for p in spark_jobs]
        input_set = set(norm_inputs)
        ordered: list[str] = []
        for asset_name in self.topological_order():
            node = self._nodes[asset_name]
            if (
                node.engine == "spark"
                and node.file_path
                and node.file_path in input_set
                and node.file_path not in ordered
            ):
                ordered.append(node.file_path)
        for job in sorted(norm_inputs):
            if job not in ordered:
                ordered.append(job)
        return ordered

    def _bfs_distances(
        self,
        start: str,
        step_attr: Literal["upstream", "downstream"],
        max_depth: int | None = None,
    ) -> dict[str, int]:
        canonical_start = self.get_node(start).name
        distances: dict[str, int] = {canonical_start: 0}
        queue: deque[tuple[str, int]] = deque([(canonical_start, 0)])
        while queue:
            current, dist = queue.popleft()
            if max_depth is not None and dist >= max_depth:
                continue
            neighbors = getattr(self.get_node(current), step_attr)
            for nxt in neighbors:
                resolved_nxt = self.resolve_asset_name(nxt)
                if resolved_nxt is not None and resolved_nxt not in distances:
                    distances[resolved_nxt] = dist + 1
                    queue.append((resolved_nxt, dist + 1))
        return distances

    def query_lineage(
        self,
        table_name: str,
        direction: Literal["upstream", "downstream", "both"] = "both",
        depth: int = 3,
    ) -> LineageSubgraph:
        canonical = self.get_node(table_name).name
        reachable: set[str] = {canonical}
        if direction in ("upstream", "both"):
            reachable.update(self._bfs_distances(canonical, "upstream", depth).keys())
        if direction in ("downstream", "both"):
            reachable.update(self._bfs_distances(canonical, "downstream", depth).keys())

        nodes = {name: self._nodes[name] for name in sorted(reachable)}
        edges: list[LineageEdge] = []
        for src in sorted(reachable):
            for dst in self._nodes[src].downstream:
                if dst in reachable:
                    edges.append(LineageEdge(source=src, target=dst))

        return LineageSubgraph(
            root=canonical,
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
        dbt_project_yml = _join_rel(self.dbt_project_rel, "dbt_project.yml")
        profiles_yml = _join_rel(self.dbt_project_rel, "profiles.yml")

        for norm in normalized_inputs:
            if norm in self._file_to_assets:
                seeds.update(self._file_to_assets[norm])
            if norm in self._macro_to_models:
                seeds.update(self._macro_to_models[norm])
            if norm == "spark_jobs/spark_config.py" or norm.endswith("/spark_config.py"):
                seeds.update(
                    name
                    for name, node in self._nodes.items()
                    if node.engine == "spark"
                )
            if norm in (
                dbt_project_yml,
                profiles_yml,
                "dbt_project/dbt_project.yml",
                "dbt_project/profiles.yml",
            ):
                seeds.update(
                    name
                    for name, node in self._nodes.items()
                    if node.engine == "dbt"
                )

        affected_assets: set[str] = set()
        for seed in seeds:
            affected_assets.update(self._bfs_distances(seed, "downstream").keys())

        ordered_assets = self.topological_order(list(affected_assets))
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
