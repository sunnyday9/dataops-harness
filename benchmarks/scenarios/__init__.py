from __future__ import annotations

from benchmarks.scenarios.base import FaultScenario, InjectionOutcome
from benchmarks.scenarios.spark_scenarios import (
    SPARK_SCENARIOS,
    SparkAppendGmvDoublingScenario,
    SparkDriverOomScenario,
    SparkJoinSkewScenario,
    SparkJsonSchemaDriftScenario,
)

SCENARIO_REGISTRY: dict[str, FaultScenario] = {
    **SPARK_SCENARIOS,
}


def list_scenarios() -> list[FaultScenario]:
    """Return all registered fault scenarios in canonical order."""
    return list(SCENARIO_REGISTRY.values())


def get_scenario(scenario_id: str) -> FaultScenario:
    """Look up a registered FaultScenario by `scenario_id`."""
    if scenario_id not in SCENARIO_REGISTRY:
        available = ", ".join(sorted(SCENARIO_REGISTRY))
        raise KeyError(f"Unknown scenario {scenario_id!r}. Available: {available}")
    return SCENARIO_REGISTRY[scenario_id]


__all__ = [
    "SCENARIO_REGISTRY",
    "SPARK_SCENARIOS",
    "FaultScenario",
    "InjectionOutcome",
    "SparkAppendGmvDoublingScenario",
    "SparkDriverOomScenario",
    "SparkJoinSkewScenario",
    "SparkJsonSchemaDriftScenario",
    "get_scenario",
    "list_scenarios",
]
