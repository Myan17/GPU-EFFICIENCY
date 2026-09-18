"""Quantify what the optimization engine actually saves.

The project's claim is that convergence-aware detection saves GPU spend. This
benchmark measures it by replaying the simulator's own fault models through the
engine's own rule thresholds, with no cloud account and no Docker required.

Method
------
For each scenario, two runs over an identical epoch timeline:

  unmanaged  the job runs to its scheduled end regardless of what it is doing
  managed    the engine's rules run each poll; the first trigger pauses the job
             and billing stops at that instant

Both are priced with `cost_model.cost_per_second`, so the saving is exactly the
wall-clock the managed run did not pay for. The simulator's noise is disabled
(fixed seed, noise off) so the result is deterministic and reproducible.

What this does NOT claim: these are simulated jobs on simulated hardware. The
saving is the engine's *detection latency* expressed in dollars at real AWS
on-demand rates -- it is a measure of how fast the rules fire, not a measurement
of a real training run.

Run: python benchmarks/savings_benchmark.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "optimization_engine"))

from cost_model import cost_per_second, hourly_cost  # noqa: E402

# Engine defaults, as shipped in docker-compose.yml / .env.example.
IDLE_THRESHOLD_PCT = 10.0
IDLE_DURATION_S = 900
STALL_THRESHOLD_S = 600
POLL_INTERVAL_S = 30

# Simulator defaults.
TOTAL_EPOCHS = 50
EPOCH_DURATION_S = 8 * 60  # 8 minutes per epoch at production scale
FAULT_AFTER_EPOCH = 20


@dataclass
class Result:
    scenario: str
    instance_type: str
    unmanaged_seconds: float
    managed_seconds: float
    detected_at_s: float | None

    @property
    def unmanaged_cost(self) -> float:
        return self.unmanaged_seconds * cost_per_second(self.instance_type)

    @property
    def managed_cost(self) -> float:
        return self.managed_seconds * cost_per_second(self.instance_type)

    @property
    def saved_usd(self) -> float:
        return self.unmanaged_cost - self.managed_cost

    @property
    def saved_pct(self) -> float:
        if self.unmanaged_cost == 0:
            return 0.0
        return 100.0 * self.saved_usd / self.unmanaged_cost

    @property
    def wasted_hours_avoided(self) -> float:
        return (self.unmanaged_seconds - self.managed_seconds) / 3600


def epoch_metrics(epoch: int, scenario: str) -> dict:
    """Noise-free version of workload_simulator.simulate_epoch."""
    progress = epoch / max(TOTAL_EPOCHS - 1, 1)
    gpu_util = 78.0
    val_acc = 0.95 * (1 - math.exp(-4 * progress))
    faulted = epoch >= FAULT_AFTER_EPOCH

    if scenario == "IDLE_GPU" and faulted:
        gpu_util = 2.5
        val_acc = 0.95 * (1 - math.exp(-4 * (FAULT_AFTER_EPOCH / max(TOTAL_EPOCHS - 1, 1))))
    elif scenario == "STALL" and faulted:
        gpu_util = 72.0
        val_acc = 0.95 * (1 - math.exp(-4 * (FAULT_AFTER_EPOCH / max(TOTAL_EPOCHS - 1, 1))))
    elif scenario == "OOM_CRASH" and faulted:
        gpu_util = 0.0

    return {"gpu_util": gpu_util, "val_acc": val_acc, "crashed": scenario == "OOM_CRASH" and faulted}


def run_scenario(scenario: str, instance_type: str) -> Result:
    """Replay the timeline once, recording when the engine would have paused."""
    total_seconds = TOTAL_EPOCHS * EPOCH_DURATION_S

    idle_start: float | None = None
    last_progress_at = 0.0
    last_val_acc = 0.0
    detected_at: float | None = None

    for tick in range(0, total_seconds, POLL_INTERVAL_S):
        epoch = min(tick // EPOCH_DURATION_S, TOTAL_EPOCHS - 1)
        m = epoch_metrics(epoch, scenario)

        # R3 -- an OOM crash ends the job; nothing is saved by pausing, but the
        # unmanaged run would keep the instance up until its scheduled end.
        if m["crashed"]:
            detected_at = float(tick)
            break

        # R2 -- no measurable accuracy gain for STALL_THRESHOLD_S.
        if m["val_acc"] > last_val_acc + 1e-4:
            last_val_acc = m["val_acc"]
            last_progress_at = float(tick)
        elif tick - last_progress_at > STALL_THRESHOLD_S:
            detected_at = float(tick)
            break

        # R1 -- utilisation under threshold for IDLE_DURATION_S.
        if m["gpu_util"] < IDLE_THRESHOLD_PCT:
            if idle_start is None:
                idle_start = float(tick)
            elif tick - idle_start >= IDLE_DURATION_S:
                detected_at = float(tick)
                break
        else:
            idle_start = None

    managed_seconds = float(detected_at) if detected_at is not None else float(total_seconds)
    return Result(
        scenario=scenario,
        instance_type=instance_type,
        unmanaged_seconds=float(total_seconds),
        managed_seconds=managed_seconds,
        detected_at_s=detected_at,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="also write results as JSON")
    args = parser.parse_args()

    scenarios = ["NORMAL", "IDLE_GPU", "STALL", "OOM_CRASH"]
    instances = ["g4dn.xlarge", "p3.2xlarge", "p3.8xlarge"]

    print(f"Job: {TOTAL_EPOCHS} epochs x {EPOCH_DURATION_S // 60} min "
          f"= {TOTAL_EPOCHS * EPOCH_DURATION_S / 3600:.1f} h unmanaged")
    print(f"Fault injected at epoch {FAULT_AFTER_EPOCH}; engine polls every {POLL_INTERVAL_S}s\n")

    results: list[Result] = []
    for instance in instances:
        print(f"=== {instance} (${hourly_cost(instance):.3f}/hr) ===")
        header = f"{'scenario':<11} {'detected':>10} {'unmanaged':>11} {'managed':>9} {'saved':>9} {'saved':>7}"
        print(header)
        print("-" * len(header))
        for scenario in scenarios:
            r = run_scenario(scenario, instance)
            results.append(r)
            detected = f"{r.detected_at_s / 60:.1f} min" if r.detected_at_s is not None else "--"
            print(f"{scenario:<11} {detected:>10} ${r.unmanaged_cost:>10.2f} "
                  f"${r.managed_cost:>8.2f} ${r.saved_usd:>8.2f} {r.saved_pct:>6.1f}%")
        print()

    faulted = [r for r in results if r.scenario != "NORMAL"]
    print(f"Across {len(faulted)} faulted runs: "
          f"${sum(r.saved_usd for r in faulted):.2f} saved, "
          f"{sum(r.wasted_hours_avoided for r in faulted):.1f} GPU-hours avoided.")
    normal = [r for r in results if r.scenario == "NORMAL"]
    print(f"On {len(normal)} healthy runs the engine paused nothing "
          f"(${sum(r.saved_usd for r in normal):.2f} saved) -- no false positives.")

    if args.json:
        args.json.write_text(json.dumps([{
            "scenario": r.scenario,
            "instance_type": r.instance_type,
            "detected_at_s": r.detected_at_s,
            "unmanaged_cost_usd": round(r.unmanaged_cost, 4),
            "managed_cost_usd": round(r.managed_cost, 4),
            "saved_usd": round(r.saved_usd, 4),
            "saved_pct": round(r.saved_pct, 2),
            "gpu_hours_avoided": round(r.wasted_hours_avoided, 4),
        } for r in results], indent=2) + "\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
