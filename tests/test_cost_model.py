"""Unit tests for the cost model.

These are the numbers every savings claim in the README rests on, so they are
asserted exactly rather than approximately where the arithmetic is exact.
"""
import pytest

from cost_model import (
    INSTANCE_COSTS,
    INSTANCE_DOWNGRADE_MAP,
    INSTANCE_UPGRADE_MAP,
    CostTracker,
    cost_per_second,
    hourly_cost,
)


class TestPricing:
    def test_known_instance_returns_its_published_rate(self):
        assert hourly_cost("g4dn.xlarge") == 0.526
        assert hourly_cost("p3.16xlarge") == 24.480

    def test_unknown_instance_falls_back_to_the_cheapest_gpu_rate(self):
        # Falling back to a low rate means the engine under-claims savings for
        # an unrecognised instance rather than over-claiming them.
        assert hourly_cost("does-not-exist") == 0.526
        assert hourly_cost("does-not-exist") == min(INSTANCE_COSTS.values())

    def test_per_second_rate_is_the_hourly_rate_over_3600(self):
        assert cost_per_second("g4dn.xlarge") == pytest.approx(0.526 / 3600)

    def test_every_upgrade_and_downgrade_target_is_a_priced_instance(self):
        # A recommendation pointing at an unpriced instance would silently
        # fall back to 0.526/hr and produce a nonsense cost delta.
        for source, target in {**INSTANCE_UPGRADE_MAP, **INSTANCE_DOWNGRADE_MAP}.items():
            assert source in INSTANCE_COSTS, f"{source} priced"
            assert target in INSTANCE_COSTS, f"{target} priced"

    def test_upgrade_targets_cost_more_and_downgrade_targets_cost_less(self):
        for source, target in INSTANCE_UPGRADE_MAP.items():
            assert hourly_cost(target) > hourly_cost(source), f"{source}->{target}"
        for source, target in INSTANCE_DOWNGRADE_MAP.items():
            assert hourly_cost(target) < hourly_cost(source), f"{source}->{target}"


class TestCostAccumulation:
    def test_tick_accumulates_elapsed_wall_time_at_the_instance_rate(self):
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        assert tracker.tick(3600.0) == pytest.approx(0.526)
        # A second hour on the same tracker doubles it.
        assert tracker.tick(7200.0) == pytest.approx(1.052)

    def test_tick_is_monotonic_and_cumulative_cost_matches_the_last_tick(self):
        tracker = CostTracker(instance_type="p3.2xlarge", start_time=0.0, _last_sample_time=0.0)
        previous = 0.0
        for t in (10.0, 20.0, 30.0):
            current = tracker.tick(t)
            assert current > previous
            previous = current
        assert tracker.cumulative_cost == pytest.approx(previous)

    def test_savings_from_pause_is_the_avoided_idle_spend(self):
        tracker = CostTracker(instance_type="p3.8xlarge", start_time=0.0, _last_sample_time=0.0)
        # 15 minutes idle on a $12.24/hr instance.
        assert tracker.savings_from_pause(900) == pytest.approx(12.24 / 4)


class TestConvergenceMetric:
    def test_cost_per_convergence_step_is_spend_over_accuracy_gained(self):
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        tracker.tick(3600.0)                       # $0.526 spent
        assert tracker.update_convergence(0.50) == pytest.approx(0.526 / 0.50)

    def test_a_stall_makes_the_metric_climb(self):
        # This is the signal the engine alerts on: during a stall the spend
        # keeps accruing while accuracy does not move, so the cost of each
        # unit of accuracy gained rises.
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        tracker.tick(3600.0)
        first = tracker.update_convergence(0.50)

        tracker.tick(7200.0)
        second = tracker.update_convergence(0.50)
        assert second > first
        assert second == pytest.approx(2 * first)

    def test_zero_progress_does_not_divide_by_zero(self):
        # Accuracy still at the 0.0 baseline: the guard must return the last
        # known value rather than raise ZeroDivisionError.
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        tracker.tick(3600.0)
        assert tracker.update_convergence(0.0) == 0.0
        assert tracker.update_convergence(0.00005) == 0.0

    def test_regressed_accuracy_does_not_produce_a_negative_cost(self):
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        tracker.tick(3600.0)
        tracker.update_convergence(0.50)
        assert tracker.update_convergence(0.20) >= 0.0

    def test_metric_starts_at_zero_before_any_progress_is_reported(self):
        tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        assert tracker.cost_per_convergence_step == 0.0
