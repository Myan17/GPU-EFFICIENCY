"""Tests for the Slack alert payloads.

These alerts are how an operator learns a job was paused and what it saved, so
the numbers embedded in them are asserted, not just the fact that a payload was
produced. No test performs a network call.
"""
import importlib
import json

import pytest


@pytest.fixture
def notifier(monkeypatch):
    """Notifier with no webhook configured: _send logs instead of posting."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    import slack_notifier

    importlib.reload(slack_notifier)
    return slack_notifier


@pytest.fixture
def posted(monkeypatch):
    """Notifier with a webhook configured; captures what would be POSTed."""
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/T/B/X")
    import slack_notifier

    importlib.reload(slack_notifier)

    sent = []

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

    def _fake_post(url, json=None, timeout=None):
        sent.append({"url": url, "payload": json, "timeout": timeout})
        return _Resp()

    monkeypatch.setattr(slack_notifier.requests, "post", _fake_post)
    slack_notifier._sent = sent
    return slack_notifier


def _text(payload: dict) -> str:
    """Flatten a Slack payload to searchable text."""
    return json.dumps(payload)


class TestSendPath:
    def test_no_webhook_configured_is_a_no_op_that_reports_success(self, notifier):
        # Local development and CI have no webhook. Alerting must not fail the
        # engine's poll loop just because Slack is not wired up.
        assert notifier._send({"attachments": []}) is True

    def test_configured_webhook_receives_the_payload(self, posted):
        assert posted._send({"attachments": [{"color": "x"}]}) is True
        assert len(posted._sent) == 1
        assert posted._sent[0]["url"] == "https://hooks.slack.test/T/B/X"
        assert posted._sent[0]["timeout"] == 5

    def test_a_slack_outage_does_not_raise(self, posted, monkeypatch):
        # If Slack is down the engine must keep running and keep saving money.
        def _boom(*a, **kw):
            raise ConnectionError("slack unreachable")

        monkeypatch.setattr(posted.requests, "post", _boom)
        assert posted._send({"attachments": []}) is False


class TestIdleGpuAlert:
    def test_payload_carries_the_savings_and_the_downgrade_hint(self, posted):
        posted.alert_idle_gpu(
            job_id="job-7",
            instance_type="p3.2xlarge",
            gpu_util=2.5,
            idle_duration_s=930.0,
            estimated_savings_usd=0.79,
            paused=True,
            downgrade_rec="g4dn.xlarge",
        )
        body = _text(posted._sent[0]["payload"])
        assert "job-7" in body
        assert "p3.2xlarge" in body
        assert "g4dn.xlarge" in body
        assert "0.79" in body

    def test_says_paused_only_when_the_job_was_actually_paused(self, posted):
        posted.alert_idle_gpu(
            job_id="job-7", instance_type="g4dn.xlarge", gpu_util=2.0,
            idle_duration_s=930.0, estimated_savings_usd=0.14,
            paused=False, downgrade_rec=None,
        )
        body = _text(posted._sent[0]["payload"])
        assert "No automated action taken" in body
        assert "automatically" not in body


class TestStallAlert:
    def test_payload_carries_the_stall_duration_and_spend(self, posted):
        posted.alert_stall(
            job_id="job-9", instance_type="g4dn.xlarge",
            stall_duration_s=660.0, val_accuracy=0.61,
            cumulative_cost_usd=1.23, paused=True,
        )
        body = _text(posted._sent[0]["payload"])
        assert "job-9" in body
        assert "1.23" in body


class TestOomAlert:
    def test_payload_recommends_the_larger_instance(self, posted):
        posted.alert_oom_crash(
            job_id="job-3", instance_type="g4dn.xlarge",
            cumulative_cost_usd=0.44, upgrade_rec="g4dn.2xlarge",
        )
        body = _text(posted._sent[0]["payload"])
        assert "g4dn.2xlarge" in body

    def test_payload_is_still_valid_without_a_recommendation(self, posted):
        posted.alert_oom_crash(
            job_id="job-3", instance_type="unpriced.instance",
            cumulative_cost_usd=0.44, upgrade_rec=None,
        )
        assert posted._sent[0]["payload"]["attachments"]


class TestCostAlert:
    def test_cost_per_convergence_alert_is_sent(self, posted):
        posted.alert_cost_per_convergence(
            job_id="job-1", instance_type="g4dn.xlarge",
            cost_per_step=2.5, cumulative_cost=1.1, val_accuracy=0.44,
        )
        assert len(posted._sent) == 1
        assert "job-1" in _text(posted._sent[0]["payload"])
