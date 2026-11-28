from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from src.adapters.openlineage import extract_openlineage_lineage, load_openlineage_events
from src.core.context import PrincipalRole, ResolutionMode
from src.core.policy import normalize_sql_dialect

MANIFEST_FILENAMES: tuple[str, ...] = ("dataops-harness.yaml", "dataops-harness.yml")
_SKIP_SCAN_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "target",
        "dbt_packages",
        "node_modules",
        "logs",
        "runs",
    }
)
_PII_TAGS = frozenset({"pii", "sensitive", "confidential", "restricted"})
_PARTITION_TAGS = frozenset({"partition", "partition_key"})
_ENV_VAR_DEFAULT_RE = re.compile(
    r"\{\{\s*env_var\s*\(\s*['\"][^'\"]+['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"
)


class RolePolicyOverride(BaseModel):
    """Per-role permission overrides declared inside a Project Manifest."""

    model_config = ConfigDict(extra="forbid")

    resolution_mode: ResolutionMode | None = None
    allow_patch_paths: list[str] | None = None
    allow_read_spark_event_logs: bool | None = None
    sql_scope: Literal["all_layers", "mart_only"] | None = None
    allowed_tables: list[str] | None = None
    sql_mode: Literal["rows_redacted", "aggregate_or_metadata_only"] | None = None


class ManifestShadowConfig(BaseModel):
    """Target-repository Shadow Backend configuration inside `dataops-harness.yaml`."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["duckdb", "postgres", "spark", "none"] | None = None
    connection_env: str | None = None
    duckdb_path: str | None = None
    schema_prefix: str = "shadow_"


class ManifestGovernanceOverrides(BaseModel):
    """Governance policy and asset overrides declared inside a Project Manifest."""

    model_config = ConfigDict(extra="forbid")

    default_resolution_mode: ResolutionMode | None = None
    sql_dialect: str | None = None
    pii_columns: list[str] = Field(default_factory=list)
    partitioned_tables: dict[str, list[str]] = Field(default_factory=dict)
    mart_tables: list[str] = Field(default_factory=list)
    max_limit: int | None = None
    blocked_functions: list[str] = Field(default_factory=list)
    role_overrides: dict[PrincipalRole, RolePolicyOverride] = Field(default_factory=dict)


class ProjectManifest(BaseModel):
    """Minimal declarative manifest (`dataops-harness.yaml`) for an external repository."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    name: str | None = None
    sql_dialect: str | None = None
    dbt_project_dir: str | None = None
    dbt_manifest_path: str | None = None
    dbt_run_results_path: str | None = None
    openlineage_paths: list[str] = Field(default_factory=list)
    spark_jobs_dir: str | None = None
    spark_lineage_path: str | None = None
    lakehouse_dir: str | None = None
    shadow: ManifestShadowConfig = Field(default_factory=ManifestShadowConfig)
    governance: ManifestGovernanceOverrides = Field(default_factory=ManifestGovernanceOverrides)


class DiscoveredMetadata(BaseModel):
    """Auto-discovered metadata merged from dbt artifacts, OpenLineage, and repository layout."""

    sql_dialect: str | None = None
    mart_tables: list[str] = Field(default_factory=list)
    pii_columns: list[str] = Field(default_factory=list)
    partitioned_tables: dict[str, list[str]] = Field(default_factory=dict)
    patchable_paths: list[str] = Field(default_factory=list)
    discovered_assets: list[str] = Field(default_factory=list)


class ResolvedProject(BaseModel):
    """Resolved repository layout, artifact locations, dialect, and discovered governance signals."""

    repo_root: Path
    manifest_path: Path | None = None
    has_manifest: bool = False
    manifest: ProjectManifest = Field(default_factory=ProjectManifest)
    default_resolution_mode: ResolutionMode = "diagnosis_only"
    dbt_project_dir: Path | None = None
    dbt_project_rel: str = ""
    dbt_manifest_path: Path | None = None
    dbt_run_results_path: Path | None = None
    openlineage_paths: list[Path] = Field(default_factory=list)
    spark_jobs_dir: Path | None = None
    spark_lineage_path: Path | None = None
    lakehouse_dir: Path | None = None
    duckdb_path: Path | None = None
    sql_dialect: str = "duckdb"
    discovered: DiscoveredMetadata = Field(default_factory=DiscoveredMetadata)


def _append_unique(target: list[str], values: Sequence[str]) -> None:
    seen = set(target)
    for val in values:
        cleaned = str(val).strip()
        if cleaned and cleaned not in seen:
            target.append(cleaned)
            seen.add(cleaned)


def _append_unique_ci(target: list[str], values: Sequence[str]) -> None:
    seen_lower = {item.lower() for item in target}
    for val in values:
        cleaned = str(val).strip()
        if cleaned and cleaned.lower() not in seen_lower:
            target.append(cleaned)
            seen_lower.add(cleaned.lower())


def _merge_partition_columns_ci(
    partitioned: dict[str, list[str]], table_name: str, cols: Sequence[str]
) -> None:
    cleaned = [str(c).strip() for c in cols if str(c).strip()]
    if not cleaned:
        return
    existing_key = next((k for k in partitioned if k.lower() == table_name.lower()), None)
    key = existing_key or table_name
    _append_unique_ci(partitioned.setdefault(key, []), cleaned)


def _resolve_contained_path(repo_root: Path, raw_path: str, *, label: str) -> Path:
    cleaned = raw_path.strip()
    if not cleaned:
        raise ValueError(f"{label} path must not be empty.")
    candidate = Path(cleaned)
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError(f"{label} escapes repository root: {raw_path!r}")
    return resolved


def load_project_manifest(manifest_path: Path) -> ProjectManifest:
    """Load and validate a `ProjectManifest` YAML file."""
    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Project manifest not found: {manifest_path}")
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in project manifest {manifest_path}: {exc}") from exc
    if raw is None:
        return ProjectManifest()
    if not isinstance(raw, dict):
        raise TypeError(
            f"Project manifest {manifest_path} must deserialize to a mapping, got {type(raw).__name__}."
        )
    return ProjectManifest.model_validate(raw)


def _iter_tag_labels(meta_obj: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    config = meta_obj.get("config") if isinstance(meta_obj.get("config"), dict) else {}
    for container in (meta_obj, config):
        for key in ("tags", "labels", "classifications"):
            val = container.get(key)
            if isinstance(val, str):
                labels.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        lbl = item.get("name") or item.get("key") or item.get("tag")
                        if lbl:
                            labels.append(str(lbl))
                    elif item:
                        labels.append(str(item))
    return labels


def _merged_meta(meta_obj: dict[str, Any]) -> dict[str, Any]:
    config = meta_obj.get("config") if isinstance(meta_obj.get("config"), dict) else {}
    cfg_meta = config.get("meta") if isinstance(config.get("meta"), dict) else {}
    direct_meta = meta_obj.get("meta") if isinstance(meta_obj.get("meta"), dict) else {}
    merged = dict(cfg_meta)
    merged.update(direct_meta)
    return merged


def is_dbt_column_pii(col_meta: dict[str, Any]) -> bool:
    """Return True if a dbt column carries PII/sensitive tags or meta annotations."""
    labels = {lbl.strip().lower() for lbl in _iter_tag_labels(col_meta)}
    if labels & _PII_TAGS:
        return True
    meta = _merged_meta(col_meta)
    if meta.get("pii") is True or meta.get("is_pii") is True or meta.get("contains_pii") is True:
        return True
    for key in ("sensitivity", "classification", "privacy"):
        val = str(meta.get(key) or "").strip().lower()
        if val in _PII_TAGS:
            return True
    return False


def _extract_partition_names(raw: Any) -> list[str]:
    if isinstance(raw, str):
        cleaned = raw.strip()
        return [cleaned] if cleaned else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(_extract_partition_names(item))
        return out
    if isinstance(raw, dict):
        field = raw.get("field") or raw.get("column") or raw.get("name")
        if isinstance(field, str) and field.strip():
            return [field.strip()]
    return []


def extract_dbt_node_partition_columns(node_meta: dict[str, Any]) -> list[str]:
    """Extract partition columns declared via dbt config, meta, or column tags."""
    cols: list[str] = []
    config = node_meta.get("config") if isinstance(node_meta.get("config"), dict) else {}
    meta = _merged_meta(node_meta)
    for raw_part in (
        config.get("partition_by"),
        config.get("partition_columns"),
        meta.get("partition_columns"),
        meta.get("partition_by"),
    ):
        _append_unique(cols, _extract_partition_names(raw_part))

    columns = node_meta.get("columns") if isinstance(node_meta.get("columns"), dict) else {}
    for col_name, col_spec in columns.items():
        if not isinstance(col_spec, dict):
            continue
        col_labels = {lbl.strip().lower() for lbl in _iter_tag_labels(col_spec)}
        col_meta = _merged_meta(col_spec)
        if (
            (col_labels & _PARTITION_TAGS)
            or col_meta.get("partition") is True
            or col_meta.get("is_partition") is True
        ):
            _append_unique(cols, [str(col_spec.get("name") or col_name)])
    return cols


def extract_dbt_node_pii_columns(node_meta: dict[str, Any]) -> list[str]:
    """Extract PII columns declared on a dbt model, source, seed, or snapshot node."""
    pii_cols: list[str] = []
    meta = _merged_meta(node_meta)
    raw_model_pii = meta.get("pii_columns") or meta.get("sensitive_columns")
    if isinstance(raw_model_pii, list):
        _append_unique(pii_cols, [str(c) for c in raw_model_pii])
    elif isinstance(raw_model_pii, str):
        _append_unique(pii_cols, [raw_model_pii])

    columns = node_meta.get("columns") if isinstance(node_meta.get("columns"), dict) else {}
    for col_name, col_spec in columns.items():
        if isinstance(col_spec, dict) and is_dbt_column_pii(col_spec):
            _append_unique(pii_cols, [str(col_spec.get("name") or col_name)])
    return pii_cols


def infer_dbt_layer(node_meta: dict[str, Any], orig_path: str = "", name: str = "") -> str:
    """Infer pipeline layer (`staging`, `intermediate`, `mart`, `raw`, `dbt`) for a dbt asset."""
    res_type = str(node_meta.get("resource_type") or "")
    if res_type == "source":
        return "raw"
    if res_type == "seed":
        return "raw"

    meta = _merged_meta(node_meta)
    explicit = meta.get("layer")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()

    tags = {lbl.strip().lower() for lbl in _iter_tag_labels(node_meta)}
    if tags & {"mart", "marts", "gold"}:
        return "mart"
    if "intermediate" in tags:
        return "intermediate"
    if tags & {"staging", "stg"}:
        return "staging"
    if tags & {"raw", "bronze"}:
        return "raw"

    norm = orig_path.replace("\\", "/").lower()
    if "/staging/" in norm or norm.startswith(("models/staging/", "staging/")):
        return "staging"
    if "/intermediate/" in norm or norm.startswith(("models/intermediate/", "intermediate/")):
        return "intermediate"
    if "/marts/" in norm or "/mart/" in norm or norm.startswith(("models/marts/", "marts/")):
        return "mart"

    lowered_name = name.lower()
    if lowered_name.startswith(("fct_", "dim_", "mart_")):
        return "mart"
    if lowered_name.startswith("int_"):
        return "intermediate"
    if lowered_name.startswith("stg_"):
        return "staging"
    return "dbt"


def _is_contained_dir(repo_root: Path, cand_dir: Path) -> bool:
    try:
        resolved_dir = cand_dir.resolve()
    except OSError:
        return False
    return resolved_dir.is_dir() and resolved_dir.is_relative_to(repo_root.resolve())


def _find_dbt_project_dir(repo_root: Path) -> Path | None:
    resolved_root = repo_root.resolve()
    root_yml = resolved_root / "dbt_project.yml"
    if root_yml.is_file() and root_yml.resolve().is_relative_to(resolved_root):
        return resolved_root
    preferred = resolved_root / "dbt_project"
    preferred_yml = preferred / "dbt_project.yml"
    if (
        _is_contained_dir(resolved_root, preferred)
        and preferred_yml.is_file()
        and preferred_yml.resolve().is_relative_to(resolved_root)
    ):
        return preferred.resolve()

    for depth_one in sorted(resolved_root.iterdir()):
        if not _is_contained_dir(resolved_root, depth_one) or depth_one.name in _SKIP_SCAN_DIRS:
            continue
        yml_one = depth_one / "dbt_project.yml"
        if yml_one.is_file() and yml_one.resolve().is_relative_to(resolved_root):
            return depth_one.resolve()
        for depth_two in sorted(depth_one.iterdir()):
            if not _is_contained_dir(resolved_root, depth_two) or depth_two.name in _SKIP_SCAN_DIRS:
                continue
            yml_two = depth_two / "dbt_project.yml"
            if yml_two.is_file() and yml_two.resolve().is_relative_to(resolved_root):
                return depth_two.resolve()
    return None


def _safe_load_dbt_yaml(yml_path: Path) -> dict[str, Any]:
    raw_text = yml_path.read_text(encoding="utf-8")
    try:
        loaded = yaml.safe_load(raw_text)
        return loaded if isinstance(loaded, dict) else {}
    except yaml.YAMLError:
        sanitized = re.sub(r"\{#.*?#\}", "", raw_text, flags=re.DOTALL)
        sanitized = re.sub(r"^[ \t]*\{%.*?%\}[ \t]*$", "", sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r"\{\{.*?\}\}", "__jinja__", sanitized, flags=re.DOTALL)
        loaded = yaml.safe_load(sanitized)
        return loaded if isinstance(loaded, dict) else {}


def _read_dbt_project_yml(dbt_dir: Path | None, repo_root: Path | None = None) -> dict[str, Any]:
    if dbt_dir is None:
        return {}
    yml_path = dbt_dir / "dbt_project.yml"
    if not yml_path.is_file():
        return {}
    if repo_root is not None and not yml_path.resolve().is_relative_to(repo_root.resolve()):
        return {}
    return _safe_load_dbt_yaml(yml_path)


def _discover_profile_hints(
    repo_root: Path,
    dbt_dir: Path | None,
    dbt_cfg: dict[str, Any],
) -> tuple[str | None, Path | None]:
    resolved_root = repo_root.resolve()
    search_dirs = [d for d in (dbt_dir, resolved_root) if d is not None]
    profiles_data: dict[str, Any] = {}
    profiles_dir: Path | None = None
    for candidate_dir in search_dirs:
        cand = candidate_dir / "profiles.yml"
        if cand.is_file() and cand.resolve().is_relative_to(resolved_root):
            loaded = _safe_load_dbt_yaml(cand)
            if loaded:
                profiles_data = loaded
                profiles_dir = candidate_dir.resolve()
                break
    if not profiles_data:
        return None, None

    profile_name = str(dbt_cfg.get("profile") or "").strip()
    profile_entry = (
        profiles_data.get(profile_name)
        if profile_name and isinstance(profiles_data.get(profile_name), dict)
        else None
    )
    if profile_entry is None:
        for key, val in profiles_data.items():
            if key != "config" and isinstance(val, dict) and isinstance(val.get("outputs"), dict):
                profile_entry = val
                break
    if not isinstance(profile_entry, dict):
        return None, None

    outputs = profile_entry.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return None, None
    target_name = str(profile_entry.get("target") or "").strip()
    target_cfg = (
        outputs.get(target_name)
        if target_name and isinstance(outputs.get(target_name), dict)
        else (
            outputs.get("dev")
            if isinstance(outputs.get("dev"), dict)
            else next((v for v in outputs.values() if isinstance(v, dict)), None)
        )
    )
    if not isinstance(target_cfg, dict):
        return None, None

    raw_type = target_cfg.get("type")
    adapter_dialect = (
        normalize_sql_dialect(str(raw_type))
        if isinstance(raw_type, str) and raw_type.strip()
        else None
    )

    duckdb_path: Path | None = None
    if adapter_dialect == "duckdb" and profiles_dir is not None:
        raw_db_path = target_cfg.get("path")
        if isinstance(raw_db_path, str) and raw_db_path.strip():
            cleaned_path = raw_db_path.strip()
            env_match = _ENV_VAR_DEFAULT_RE.fullmatch(cleaned_path)
            if env_match:
                cleaned_path = env_match.group(1)
            if "{{" not in cleaned_path and "__jinja__" not in cleaned_path and cleaned_path != ":memory:":
                cand_db = Path(cleaned_path)
                resolved_db = (
                    cand_db.resolve()
                    if cand_db.is_absolute()
                    else (profiles_dir / cand_db).resolve()
                )
                if resolved_db.is_relative_to(resolved_root):
                    duckdb_path = resolved_db

    return adapter_dialect, duckdb_path


def _discover_openlineage_paths(
    repo_root: Path, manifest: ProjectManifest
) -> list[Path]:
    resolved_root = repo_root.resolve()
    if manifest.openlineage_paths:
        paths = [
            _resolve_contained_path(resolved_root, raw, label="openlineage_paths")
            for raw in manifest.openlineage_paths
        ]
        return [p for p in paths if p.exists()]

    discovered: list[Path] = []
    ol_dir = resolved_root / "openlineage"
    if _is_contained_dir(resolved_root, ol_dir):
        for cand in sorted({*ol_dir.rglob("*.json"), *ol_dir.rglob("*.jsonl")}):
            if not cand.is_file():
                continue
            resolved_cand = cand.resolve()
            if resolved_cand.is_relative_to(resolved_root) and resolved_cand not in discovered:
                discovered.append(resolved_cand)
    for filename in ("openlineage.json", "openlineage.jsonl"):
        cand = resolved_root / filename
        if cand.is_file():
            resolved_cand = cand.resolve()
            if resolved_cand.is_relative_to(resolved_root) and resolved_cand not in discovered:
                discovered.append(resolved_cand)
    return discovered


def _compute_patchable_paths(
    repo_root: Path,
    dbt_project_dir: Path | None,
    dbt_project_rel: str,
    dbt_cfg: dict[str, Any],
    spark_jobs_dir: Path | None,
    ol_file_paths: Sequence[str],
) -> list[str]:
    paths: list[str] = []
    resolved_root = repo_root.resolve()
    if spark_jobs_dir is not None and spark_jobs_dir.is_dir():
        resolved_spark = spark_jobs_dir.resolve()
        if resolved_spark.is_relative_to(resolved_root):
            rel_spark = resolved_spark.relative_to(resolved_root).as_posix()
            if rel_spark and rel_spark != ".":
                _append_unique(paths, [f"{rel_spark}/"])

    if dbt_project_dir is not None:
        resolved_dbt_dir = dbt_project_dir.resolve()
        for key, default_dirs in (
            ("model-paths", ["models"]),
            ("macro-paths", ["macros"]),
            ("test-paths", ["tests"]),
            ("seed-paths", ["seeds"]),
            ("snapshot-paths", ["snapshots"]),
        ):
            raw_dirs = dbt_cfg.get(key)
            dir_list = raw_dirs if isinstance(raw_dirs, list) else default_dirs
            for entry in dir_list:
                raw_entry = str(entry).strip().replace("\\", "/")
                if not raw_entry:
                    continue
                if raw_entry.startswith("~"):
                    raise ValueError(f"dbt {key} escapes repository root: {entry!r}")
                cand_entry = Path(raw_entry)
                resolved_entry = (
                    cand_entry.resolve()
                    if cand_entry.is_absolute()
                    else (resolved_dbt_dir / cand_entry).resolve()
                )
                if not resolved_entry.is_relative_to(resolved_root):
                    raise ValueError(f"dbt {key} escapes repository root: {entry!r}")
                if not dbt_project_rel:
                    rel_entry = (
                        "."
                        if resolved_entry == resolved_root
                        else resolved_entry.relative_to(resolved_root).as_posix()
                    )
                    if rel_entry and rel_entry != ".":
                        _append_unique(paths, [f"{rel_entry}/"])
        if dbt_project_rel:
            _append_unique(paths, [f"{dbt_project_rel}/"])

    for rel_file in ol_file_paths:
        raw_ol = rel_file.replace("\\", "/").strip()
        if not raw_ol or "://" in raw_ol or raw_ol.startswith(("/", "~")):
            continue
        if any(seg == ".." for seg in raw_ol.split("/")):
            continue
        norm = posixpath.normpath(raw_ol)
        if not norm or norm in (".", "..") or norm.startswith("../"):
            continue
        lower_norm = norm.lower()
        if lower_norm in (
            "dataops-harness.yaml",
            "dataops-harness.yml",
            ".git",
        ) or lower_norm.startswith((".git/", "configs/", "logs/")):
            continue
        resolved_ol_target = (resolved_root / norm).resolve()
        if not resolved_ol_target.is_relative_to(resolved_root):
            continue
        top = norm.split("/", 1)[0]
        _append_unique(paths, [f"{top}/" if "/" in norm else norm])

    return paths


def discover_project(
    repo_root: Path,
    manifest_path: Path | None = None,
) -> ResolvedProject:
    """Discover repository layout, artifacts, SQL dialect, and governance metadata."""
    resolved_root = repo_root.resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")

    resolved_manifest_path: Path | None = None
    has_manifest = False
    if manifest_path is not None:
        resolved_manifest_path = manifest_path.resolve()
        manifest = load_project_manifest(resolved_manifest_path)
        has_manifest = True
    else:
        for fname in MANIFEST_FILENAMES:
            cand = resolved_root / fname
            if cand.is_file() or cand.is_symlink():
                resolved_manifest_path = _resolve_contained_path(
                    resolved_root, fname, label=f"Manifest {fname}"
                )
                manifest = load_project_manifest(resolved_manifest_path)
                has_manifest = True
                break
        else:
            manifest = ProjectManifest()

    default_resolution_mode: ResolutionMode = (
        (manifest.governance.default_resolution_mode or "fix")
        if has_manifest
        else "diagnosis_only"
    )

    if manifest.dbt_project_dir:
        dbt_project_dir: Path | None = _resolve_contained_path(
            resolved_root, manifest.dbt_project_dir, label="dbt_project_dir"
        )
    else:
        dbt_project_dir = _find_dbt_project_dir(resolved_root)

    dbt_project_rel = (
        ""
        if dbt_project_dir is None or dbt_project_dir == resolved_root
        else dbt_project_dir.relative_to(resolved_root).as_posix()
    )
    dbt_cfg = _read_dbt_project_yml(dbt_project_dir, resolved_root)
    target_rel = str(dbt_cfg.get("target-path") or "target").strip() or "target"
    if dbt_project_dir is not None:
        target_cand = Path(target_rel)
        resolved_target = (
            target_cand.resolve()
            if target_cand.is_absolute()
            else (dbt_project_dir / target_cand).resolve()
        )
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(f"dbt target-path escapes repository root: {target_rel!r}")

    dbt_manifest_path: Path | None = None
    if manifest.dbt_manifest_path:
        cand_manifest = _resolve_contained_path(
            resolved_root, manifest.dbt_manifest_path, label="dbt_manifest_path"
        )
        if cand_manifest.is_file():
            dbt_manifest_path = cand_manifest
    elif dbt_project_dir is not None:
        raw_manifest_cand = dbt_project_dir / target_rel / "manifest.json"
        if raw_manifest_cand.is_file() or raw_manifest_cand.is_symlink():
            cand_manifest = raw_manifest_cand.resolve()
            if not cand_manifest.is_relative_to(resolved_root):
                raise ValueError(
                    f"dbt manifest path escapes repository root: {raw_manifest_cand}"
                )
            if cand_manifest.is_file():
                dbt_manifest_path = cand_manifest

    dbt_run_results_path: Path | None = None
    if manifest.dbt_run_results_path:
        cand_rr = _resolve_contained_path(
            resolved_root, manifest.dbt_run_results_path, label="dbt_run_results_path"
        )
        if cand_rr.is_file():
            dbt_run_results_path = cand_rr
    elif dbt_project_dir is not None:
        raw_rr_cand = dbt_project_dir / target_rel / "run_results.json"
        if raw_rr_cand.is_file() or raw_rr_cand.is_symlink():
            cand_rr = raw_rr_cand.resolve()
            if not cand_rr.is_relative_to(resolved_root):
                raise ValueError(
                    f"dbt run_results path escapes repository root: {raw_rr_cand}"
                )
            if cand_rr.is_file():
                dbt_run_results_path = cand_rr

    openlineage_paths = _discover_openlineage_paths(resolved_root, manifest)

    spark_jobs_dir: Path | None = None
    if manifest.spark_jobs_dir:
        cand_spark = _resolve_contained_path(
            resolved_root, manifest.spark_jobs_dir, label="spark_jobs_dir"
        )
        if cand_spark.is_dir():
            spark_jobs_dir = cand_spark
    else:
        cand_spark_default = resolved_root / "spark_jobs"
        if _is_contained_dir(resolved_root, cand_spark_default):
            spark_jobs_dir = cand_spark_default.resolve()

    spark_lineage_path: Path | None = None
    if manifest.spark_lineage_path:
        cand_sl = _resolve_contained_path(
            resolved_root, manifest.spark_lineage_path, label="spark_lineage_path"
        )
        if cand_sl.is_file():
            spark_lineage_path = cand_sl
    elif spark_jobs_dir is not None:
        cand_sl_default = spark_jobs_dir / "lineage.yaml"
        if (
            cand_sl_default.is_file()
            and cand_sl_default.resolve().is_relative_to(resolved_root)
        ):
            spark_lineage_path = cand_sl_default.resolve()

    lakehouse_dir: Path | None = None
    if manifest.lakehouse_dir:
        lakehouse_dir = _resolve_contained_path(
            resolved_root, manifest.lakehouse_dir, label="lakehouse_dir"
        )
    elif _is_contained_dir(resolved_root, resolved_root / "lakehouse"):
        lakehouse_dir = (resolved_root / "lakehouse").resolve()

    profile_dialect, profile_duckdb = _discover_profile_hints(resolved_root, dbt_project_dir, dbt_cfg)

    duckdb_path: Path | None = None
    if manifest.shadow.duckdb_path:
        duckdb_path = _resolve_contained_path(
            resolved_root, manifest.shadow.duckdb_path, label="shadow.duckdb_path"
        )
    elif lakehouse_dir is not None and (lakehouse_dir / "prod" / "mart" / "warehouse.duckdb").is_file():
        duckdb_path = (lakehouse_dir / "prod" / "mart" / "warehouse.duckdb").resolve()
    elif profile_duckdb is not None:
        duckdb_path = profile_duckdb

    discovered_assets: list[str] = []
    mart_tables: list[str] = []
    pii_columns: list[str] = []
    partitioned_tables: dict[str, list[str]] = {}
    ol_file_paths: list[str] = []
    manifest_dialect: str | None = None
    ol_dialect: str | None = None

    if spark_lineage_path is not None:
        loaded_spark = yaml.safe_load(spark_lineage_path.read_text(encoding="utf-8")) or {}
        spark_assets = loaded_spark.get("assets") if isinstance(loaded_spark, dict) else {}
        if isinstance(spark_assets, dict):
            for name, spec in spark_assets.items():
                if not isinstance(spec, dict):
                    continue
                _append_unique_ci(discovered_assets, [str(name)])
                _append_unique_ci(pii_columns, [str(c) for c in (spec.get("pii_columns") or [])])
                p_cols = [str(c) for c in (spec.get("partition_columns") or []) if str(c).strip()]
                if p_cols:
                    _merge_partition_columns_ci(partitioned_tables, str(name), p_cols)

    if dbt_manifest_path is not None:
        raw_manifest = json.loads(dbt_manifest_path.read_text(encoding="utf-8"))
        adapter_type = (raw_manifest.get("metadata") or {}).get("adapter_type")
        if isinstance(adapter_type, str) and adapter_type.strip():
            manifest_dialect = normalize_sql_dialect(adapter_type)

        for section_key in ("sources", "nodes"):
            entries = raw_manifest.get(section_key) or {}
            if not isinstance(entries, dict):
                continue
            for meta_obj in entries.values():
                if not isinstance(meta_obj, dict):
                    continue
                res_type = str(meta_obj.get("resource_type") or "")
                if res_type not in {"model", "source", "seed", "snapshot"}:
                    continue
                asset_name = str(meta_obj.get("name") or "").strip()
                if not asset_name:
                    continue
                alias_name = str(meta_obj.get("alias") or "").strip()
                asset_names = [asset_name]
                if (
                    res_type != "source"
                    and alias_name
                    and alias_name.lower() != asset_name.lower()
                ):
                    asset_names.append(alias_name)
                _append_unique_ci(discovered_assets, asset_names)
                orig_path = str(meta_obj.get("original_file_path") or "")
                if infer_dbt_layer(meta_obj, orig_path, asset_name) == "mart":
                    _append_unique_ci(mart_tables, asset_names)
                _append_unique_ci(pii_columns, extract_dbt_node_pii_columns(meta_obj))
                part_cols = extract_dbt_node_partition_columns(meta_obj)
                if part_cols:
                    for reg_name in asset_names:
                        _merge_partition_columns_ci(partitioned_tables, reg_name, part_cols)

    if openlineage_paths:
        ol_events: list[dict[str, Any]] = []
        for ol_path in openlineage_paths:
            ol_events.extend(load_openlineage_events(ol_path))
        ol_graph = extract_openlineage_lineage(ol_events)
        if ol_graph.sql_dialect is not None:
            ol_dialect = ol_graph.sql_dialect
        for name, node in ol_graph.nodes.items():
            _append_unique_ci(discovered_assets, [name])
            if node.layer == "mart":
                _append_unique_ci(mart_tables, [name])
            _append_unique_ci(pii_columns, node.pii_columns)
            if node.partition_columns:
                _merge_partition_columns_ci(partitioned_tables, name, node.partition_columns)
            if node.file_path:
                ol_file_paths.append(node.file_path)

    shadow_dialect = (
        normalize_sql_dialect(manifest.shadow.backend)
        if manifest.shadow.backend and manifest.shadow.backend != "none"
        else None
    )
    discovered_dialect = manifest_dialect or profile_dialect or ol_dialect
    if manifest.governance.sql_dialect:
        resolved_dialect = normalize_sql_dialect(manifest.governance.sql_dialect)
    elif manifest.sql_dialect:
        resolved_dialect = normalize_sql_dialect(manifest.sql_dialect)
    elif discovered_dialect:
        resolved_dialect = discovered_dialect
    elif shadow_dialect:
        resolved_dialect = shadow_dialect
    else:
        resolved_dialect = "duckdb"

    patchable_paths = _compute_patchable_paths(
        resolved_root,
        dbt_project_dir,
        dbt_project_rel,
        dbt_cfg,
        spark_jobs_dir,
        ol_file_paths,
    )

    return ResolvedProject(
        repo_root=resolved_root,
        manifest_path=resolved_manifest_path,
        has_manifest=has_manifest,
        manifest=manifest,
        default_resolution_mode=default_resolution_mode,
        dbt_project_dir=dbt_project_dir,
        dbt_project_rel=dbt_project_rel,
        dbt_manifest_path=dbt_manifest_path,
        dbt_run_results_path=dbt_run_results_path,
        openlineage_paths=openlineage_paths,
        spark_jobs_dir=spark_jobs_dir,
        spark_lineage_path=spark_lineage_path,
        lakehouse_dir=lakehouse_dir,
        duckdb_path=duckdb_path,
        sql_dialect=resolved_dialect,
        discovered=DiscoveredMetadata(
            sql_dialect=discovered_dialect,
            mart_tables=mart_tables,
            pii_columns=pii_columns,
            partitioned_tables=partitioned_tables,
            patchable_paths=patchable_paths,
            discovered_assets=discovered_assets,
        ),
    )


__all__ = [
    "MANIFEST_FILENAMES",
    "DiscoveredMetadata",
    "ManifestGovernanceOverrides",
    "ManifestShadowConfig",
    "ProjectManifest",
    "ResolvedProject",
    "RolePolicyOverride",
    "discover_project",
    "extract_dbt_node_partition_columns",
    "extract_dbt_node_pii_columns",
    "infer_dbt_layer",
    "is_dbt_column_pii",
    "load_project_manifest",
    "normalize_sql_dialect",
]
