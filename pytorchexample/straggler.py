"""Deterministic straggler simulation utilities for async/sync FL experiments."""

from dataclasses import dataclass

TIER_MULTIPLIERS: dict[str, float] = {
    "FAST": 1.0,
    "MEDIUM": 1.5,
    "SLOW": 2.0,
}

SCENARIO_PROPORTIONS: dict[str, dict[str, float]] = {
    "balanced": {"FAST": 0.33, "MEDIUM": 0.33},
    "slow_dominant": {"FAST": 0.20, "MEDIUM": 0.10},
    "fast_dominant": {"FAST": 0.70, "MEDIUM": 0.20},
}


@dataclass
class StragglerResult:
    partition_id: int
    scenario: str
    tier: str
    multiplier: float
    baseline_mode: str
    baseline_T: float
    train_duration: float
    target_completion_time: float
    sleep_duration: float
    total_effective_time: float


def get_tier(partition_id: int, total_clients: int, scenario: str) -> tuple[str, float]:
    """Deterministically map partition ID to straggler tier."""
    if scenario not in SCENARIO_PROPORTIONS:
        raise ValueError(
            f"Unknown straggler scenario '{scenario}'. "
            f"Valid options: {list(SCENARIO_PROPORTIONS.keys())}"
        )

    props = SCENARIO_PROPORTIONS[scenario]
    n_fast = round(total_clients * props["FAST"])
    n_medium = round(total_clients * props["MEDIUM"])

    if partition_id < n_fast:
        tier = "FAST"
    elif partition_id < n_fast + n_medium:
        tier = "MEDIUM"
    else:
        tier = "SLOW"

    return tier, TIER_MULTIPLIERS[tier]


def compute_straggler(
    partition_id: int,
    total_clients: int,
    scenario: str,
    baseline_mode: str,
    baseline_T: float,
    train_duration: float,
    additive_delays: dict[str, float] | None = None,
) -> StragglerResult:
    """Compute delay based on tier.

    baseline_mode='additive': fixed extra sleep per tier (additive_delays dict).
    baseline_mode='measured' or 'fixed': sleep = max(0, multiplier * baseline_T - train_duration).
    """
    tier, multiplier = get_tier(partition_id, total_clients, scenario)

    if baseline_mode == "additive":
        delays = additive_delays or {"FAST": 0.0, "MEDIUM": 10.0, "SLOW": 20.0}
        sleep_duration = delays.get(tier, 0.0)
        target = train_duration + sleep_duration
    else:
        target = multiplier * baseline_T
        sleep_duration = max(0.0, target - train_duration)

    total_effective = train_duration + sleep_duration

    return StragglerResult(
        partition_id=partition_id,
        scenario=scenario,
        tier=tier,
        multiplier=multiplier,
        baseline_mode=baseline_mode,
        baseline_T=baseline_T,
        train_duration=train_duration,
        target_completion_time=target,
        sleep_duration=sleep_duration,
        total_effective_time=total_effective,
    )


def describe_tier_mapping(total_clients: int, scenario: str) -> str:
    """Return deterministic partition-to-tier mapping summary string."""
    if scenario not in SCENARIO_PROPORTIONS:
        raise ValueError(
            f"Unknown straggler scenario '{scenario}'. "
            f"Valid options: {list(SCENARIO_PROPORTIONS.keys())}"
        )

    props = SCENARIO_PROPORTIONS[scenario]
    n_fast = round(total_clients * props["FAST"])
    n_medium = round(total_clients * props["MEDIUM"])
    n_slow = total_clients - n_fast - n_medium

    fast_range = f"0-{n_fast - 1}" if n_fast > 0 else "(none)"
    medium_start = n_fast
    medium_end = n_fast + n_medium - 1
    medium_range = f"{medium_start}-{medium_end}" if n_medium > 0 else "(none)"
    slow_start = n_fast + n_medium
    slow_end = total_clients - 1
    slow_range = f"{slow_start}-{slow_end}" if n_slow > 0 else "(none)"

    lines = [
        f"Scenario: {scenario}  |  {total_clients} clients",
        f"  FAST   (x{TIER_MULTIPLIERS['FAST']:.1f}): partitions {fast_range}   ({n_fast} clients)",
        f"  MEDIUM (x{TIER_MULTIPLIERS['MEDIUM']:.1f}): partitions {medium_range}   ({n_medium} clients)",
        f"  SLOW   (x{TIER_MULTIPLIERS['SLOW']:.1f}): partitions {slow_range}   ({n_slow} clients)",
    ]
    return "\n".join(lines)
