"""Pin the savings benchmark so its published numbers cannot drift silently.

The README quotes these figures. If a threshold, a price, or a rule changes,
these assertions fail and the README has to be updated with them.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from savings_benchmark import Result, run_scenario  # noqa: E402


def test_a_healthy_job_is_never_paused():
    # A false positive here would pause real training. It is the one outcome
    # the engine must never produce.
    r = run_scenario("NORMAL", "g4dn.xlarge")
    assert r.detected_at_s is None
    assert r.saved_usd == 0.0


@pytest.mark.parametrize("scenario", ["IDLE_GPU", "STALL", "OOM_CRASH"])
def test_every_fault_is_detected(scenario):
    r = run_scenario(scenario, "g4dn.xlarge")
    assert r.detected_at_s is not None
    assert r.saved_usd > 0


def test_detection_happens_after_the_fault_not_before():
    # Detecting "early" would mean the rule fired on healthy epochs.
    from savings_benchmark import EPOCH_DURATION_S, FAULT_AFTER_EPOCH

    fault_at = FAULT_AFTER_EPOCH * EPOCH_DURATION_S
    for scenario in ("IDLE_GPU", "STALL", "OOM_CRASH"):
        r = run_scenario(scenario, "g4dn.xlarge")
        assert r.detected_at_s >= fault_at, scenario


def test_published_detection_latencies():
    # Fault lands at 160 min. An OOM is visible on the very next poll; the
    # other two are caught by the stall rule 10.5 min later -- when the GPU
    # goes idle, accuracy stops moving too, so R2 fires before R1's 15-minute
    # window elapses.
    assert run_scenario("IDLE_GPU", "g4dn.xlarge").detected_at_s == 10230.0
    assert run_scenario("STALL", "g4dn.xlarge").detected_at_s == 10230.0
    assert run_scenario("OOM_CRASH", "g4dn.xlarge").detected_at_s == 9600.0


def test_published_savings_percentages():
    assert run_scenario("IDLE_GPU", "g4dn.xlarge").saved_pct == pytest.approx(57.4, abs=0.1)
    assert run_scenario("STALL", "g4dn.xlarge").saved_pct == pytest.approx(57.4, abs=0.1)
    assert run_scenario("OOM_CRASH", "g4dn.xlarge").saved_pct == pytest.approx(60.0, abs=0.1)


def test_savings_percentage_is_independent_of_instance_price():
    # The engine saves the same *fraction* of wall clock whatever the box
    # costs; only the dollar figure scales.
    cheap = run_scenario("IDLE_GPU", "g4dn.xlarge")
    dear = run_scenario("IDLE_GPU", "p3.8xlarge")
    assert cheap.saved_pct == pytest.approx(dear.saved_pct)
    assert dear.saved_usd > cheap.saved_usd


def test_results_are_deterministic_across_runs():
    first = run_scenario("IDLE_GPU", "p3.2xlarge")
    second = run_scenario("IDLE_GPU", "p3.2xlarge")
    assert isinstance(first, Result)
    assert first.detected_at_s == second.detected_at_s
    assert first.saved_usd == second.saved_usd
