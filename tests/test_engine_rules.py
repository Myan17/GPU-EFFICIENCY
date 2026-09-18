"""Rule-evaluation tests for the optimization engine.

The engine's three detection rules decide when a job is paused, which is the
only action that produces the savings the project claims. Each rule is tested
for the trigger, the non-trigger, the cooldown, and the terminal-status guards.

Slack and the simulator control API are stubbed; nothing here touches network.
"""
import importlib

import pytest


@pytest.fixture
def engine(monkeypatch):
    """A freshly imported engine with network calls and alerts stubbed out."""
    monkeypatch.setenv("IDLE_GPU_THRESHOLD_PERCENT", "10")
    monkeypatch.setenv("IDLE_GPU_DURATION_SECONDS", "900")
    monkeypatch.setenv("STALL_THRESHOLD_SECONDS", "600")
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "300")
    monkeypatch.setenv("INSTANCE_TYPE", "g4dn.xlarge")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "")

    # engine.py registers its gauges on import against the global Prometheus
    # registry, so reloading it would raise DuplicateTimeseries. Import first,
    # then drop what that import registered, then reload so each test gets a
    # clean engine bound to the env above.
    import prometheus_client

    import engine as engine_module

    for collector in list(prometheus_client.REGISTRY._collector_to_names):
        try:
            prometheus_client.REGISTRY.unregister(collector)
        except KeyError:
            pass

    importlib.reload(engine_module)

    calls = {"pause": 0, "idle_alert": [], "stall_alert": [], "oom_alert": []}

    monkeypatch.setattr(engine_module, "pause_job", lambda job_id: (calls.__setitem__("pause", calls["pause"] + 1), True)[1])
    monkeypatch.setattr(engine_module, "alert_idle_gpu", lambda **kw: calls["idle_alert"].append(kw))
    monkeypatch.setattr(engine_module, "alert_stall", lambda **kw: calls["stall_alert"].append(kw))
    monkeypatch.setattr(engine_module, "alert_oom_crash", lambda **kw: calls["oom_alert"].append(kw))

    engine_module._calls = calls
    return engine_module


def _at(engine, monkeypatch, timestamp):
    """Pin engine-visible wall-clock time."""
    monkeypatch.setattr(engine.time, "time", lambda: timestamp)


class TestIdleGpuRule:
    def test_util_above_threshold_never_starts_the_idle_timer(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": 85.0, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts is None
        assert engine._calls["pause"] == 0

    def test_brief_idle_starts_the_timer_but_does_not_act(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts == 1000.0

        # 10 minutes in — still short of the 15-minute threshold.
        _at(engine, monkeypatch, 1600.0)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._calls["pause"] == 0
        assert engine._calls["idle_alert"] == []

    def test_sustained_idle_past_the_threshold_pauses_the_job_once(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")

        _at(engine, monkeypatch, 1000.0 + 901)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._calls["pause"] == 1
        assert len(engine._calls["idle_alert"]) == 1

        # Still idle a second later: cooldown must suppress a repeat.
        _at(engine, monkeypatch, 1000.0 + 902)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._calls["pause"] == 1
        assert len(engine._calls["idle_alert"]) == 1

    def test_recovered_utilisation_resets_the_timer(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts is not None

        _at(engine, monkeypatch, 1300.0)
        engine.evaluate_idle_gpu({"gpu_util": 90.0, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts is None

        # A later idle spell must be timed from scratch, not from 1000.0.
        _at(engine, monkeypatch, 1400.0)
        engine.evaluate_idle_gpu({"gpu_util": 3.0, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts == 1400.0

    @pytest.mark.parametrize("status", [3, 4, 5], ids=["oom", "completed", "paused"])
    def test_terminal_statuses_are_not_treated_as_idle(self, engine, monkeypatch, status):
        # A paused or crashed job reports ~0% utilisation. Pausing it again, or
        # claiming savings for it, would be wrong.
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": 0.0, "job_status": status}, "job-1")
        assert engine._state.idle_start_ts is None
        assert engine._calls["pause"] == 0

    def test_missing_metric_is_not_treated_as_idle(self, engine, monkeypatch):
        # Prometheus being unreachable returns None. Unknown must not look idle.
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_idle_gpu({"gpu_util": None, "job_status": 0}, "job-1")
        assert engine._state.idle_start_ts is None
        assert engine._calls["pause"] == 0


class TestStallRule:
    def test_progress_within_threshold_does_not_trigger(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_stall({"stall_seconds": 120.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        assert engine._calls["stall_alert"] == []

    def test_stall_past_threshold_pauses_and_alerts(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_stall({"stall_seconds": 601.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        assert engine._calls["pause"] == 1
        assert len(engine._calls["stall_alert"]) == 1

    def test_cooldown_suppresses_a_repeat_stall_alert(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_stall({"stall_seconds": 601.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        _at(engine, monkeypatch, 1100.0)
        engine.evaluate_stall({"stall_seconds": 701.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        assert len(engine._calls["stall_alert"]) == 1

    def test_cooldown_expiry_allows_the_next_alert(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_stall({"stall_seconds": 601.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        _at(engine, monkeypatch, 1000.0 + 301)
        engine.evaluate_stall({"stall_seconds": 901.0, "val_accuracy": 0.4, "job_status": 0}, "job-1")
        assert len(engine._calls["stall_alert"]) == 2

    @pytest.mark.parametrize("status", [3, 4, 5], ids=["oom", "completed", "paused"])
    def test_terminal_statuses_are_not_stalls(self, engine, monkeypatch, status):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_stall({"stall_seconds": 9999.0, "val_accuracy": 0.4, "job_status": status}, "job-1")
        assert engine._calls["stall_alert"] == []


class TestOomRule:
    def test_oom_status_alerts_with_an_upgrade_recommendation(self, engine, monkeypatch):
        from cost_model import CostTracker

        _at(engine, monkeypatch, 1000.0)
        engine._state.cost_tracker = CostTracker(instance_type="g4dn.xlarge", start_time=0.0, _last_sample_time=0.0)
        engine.evaluate_oom({"job_status": 3}, "job-1")

        assert len(engine._calls["oom_alert"]) == 1
        # An OOM means the box was too small, so the recommendation is upward.
        assert engine._calls["oom_alert"][0]["upgrade_rec"] == "g4dn.2xlarge"

    def test_oom_does_not_pause_the_job(self, engine, monkeypatch):
        # The job already crashed; pausing it is meaningless.
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_oom({"job_status": 3}, "job-1")
        assert engine._calls["pause"] == 0

    @pytest.mark.parametrize("status", [0, 1, 2, 4, 5])
    def test_non_oom_statuses_do_not_alert(self, engine, monkeypatch, status):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_oom({"job_status": status}, "job-1")
        assert engine._calls["oom_alert"] == []

    def test_missing_status_does_not_alert(self, engine, monkeypatch):
        _at(engine, monkeypatch, 1000.0)
        engine.evaluate_oom({"job_status": None}, "job-1")
        assert engine._calls["oom_alert"] == []
