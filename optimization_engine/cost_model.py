"""
Cost Model
==========
Maps AWS instance types to hourly on-demand pricing and computes
cost-per-convergence-step — the novel metric bridging infra cost
and ML training progress.

Cost per convergence step = cumulative_cost_usd / Δval_accuracy
"""

import time as _time
from dataclasses import dataclass, field

# AWS on-demand pricing (USD/hr) — us-east-1, as of 2024
INSTANCE_COSTS: dict[str, float] = {
    "g4dn.xlarge":   0.526,
    "g4dn.2xlarge":  0.752,
    "g4dn.4xlarge":  1.204,
    "g4dn.8xlarge":  2.264,
    "g4dn.12xlarge": 3.912,
    "p3.2xlarge":    3.060,
    "p3.8xlarge":   12.240,
    "p3.16xlarge":  24.480,
    "p2.xlarge":     0.900,
    "p2.8xlarge":    7.200,
}

# Upgrade/downgrade recommendations when action is triggered
INSTANCE_UPGRADE_MAP: dict[str, str] = {
    "g4dn.xlarge":  "g4dn.2xlarge",
    "g4dn.2xlarge": "g4dn.4xlarge",
    "g4dn.4xlarge": "g4dn.8xlarge",
    "p2.xlarge":    "p3.2xlarge",
}

INSTANCE_DOWNGRADE_MAP: dict[str, str] = {
    "g4dn.2xlarge":  "g4dn.xlarge",
    "g4dn.4xlarge":  "g4dn.2xlarge",
    "g4dn.8xlarge":  "g4dn.4xlarge",
    "g4dn.12xlarge": "g4dn.8xlarge",
    "p3.8xlarge":    "p3.2xlarge",
    "p3.16xlarge":   "p3.8xlarge",
}


def hourly_cost(instance_type: str) -> float:
    return INSTANCE_COSTS.get(instance_type, 0.526)


def cost_per_second(instance_type: str) -> float:
    return hourly_cost(instance_type) / 3600


@dataclass
class CostTracker:
    instance_type:      str
    start_time:         float = field(default_factory=_time.time)
    _last_sample_time:  float = field(default_factory=_time.time)
    _cumulative_cost:   float = 0.0
    _baseline_val_acc:  float = 0.0
    _cost_per_step:     float = 0.0

    def tick(self, current_time: float) -> float:
        """Accumulate cost since last call. Returns cumulative cost."""
        elapsed = current_time - self._last_sample_time
        self._cumulative_cost += elapsed * cost_per_second(self.instance_type)
        self._last_sample_time = current_time
        return self._cumulative_cost

    def update_convergence(self, val_accuracy: float) -> float:
        """
        Recompute cost_per_convergence_step each time val_accuracy is updated.
        Returns USD per unit of validation accuracy gained since training started.
        """
        delta_acc = val_accuracy - self._baseline_val_acc
        if delta_acc <= 0.0001:
            return self._cost_per_step  # no progress; return last known value

        self._cost_per_step    = self._cumulative_cost / delta_acc
        return self._cost_per_step

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def cost_per_convergence_step(self) -> float:
        return self._cost_per_step

    def savings_from_pause(self, idle_seconds: float) -> float:
        """Estimated USD saved by pausing instead of continuing to idle."""
        return idle_seconds * cost_per_second(self.instance_type)

    def upgrade_recommendation(self) -> str | None:
        return INSTANCE_UPGRADE_MAP.get(self.instance_type)

    def downgrade_recommendation(self) -> str | None:
        return INSTANCE_DOWNGRADE_MAP.get(self.instance_type)
