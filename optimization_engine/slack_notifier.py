"""
Slack Notifier
==============
Sends rich alert messages to a Slack incoming webhook.
Falls back to console logging when SLACK_WEBHOOK_URL is not set (dev mode).
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger("slack_notifier")

_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Colour coding for alert types
_COLOURS = {
    "idle_gpu":    "#FFA500",   # orange
    "stall":       "#FF6B6B",   # red-orange
    "oom_crash":   "#FF0000",   # red
    "resolved":    "#36A64F",   # green
    "cost_alert":  "#9B59B6",   # purple
}

_ICONS = {
    "idle_gpu":   ":sleeping:",
    "stall":      ":warning:",
    "oom_crash":  ":boom:",
    "resolved":   ":white_check_mark:",
    "cost_alert": ":moneybag:",
}


def _send(payload: dict) -> bool:
    """POST payload to webhook. Returns True on success."""
    if not _WEBHOOK_URL:
        logger.info("[Slack DEV] %s", json.dumps(payload, indent=2))
        return True
    try:
        resp = requests.post(_WEBHOOK_URL, json=payload, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Slack send failed: %s", exc)
        return False


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def alert_idle_gpu(
    job_id: str,
    instance_type: str,
    gpu_util: float,
    idle_duration_s: float,
    estimated_savings_usd: float,
    paused: bool,
    downgrade_rec: str | None,
) -> bool:
    action_text = "Job has been automatically *paused* and checkpointed." if paused else "No automated action taken."
    rec_text    = f"Consider downgrading to `{downgrade_rec}` to reduce cost." if downgrade_rec else ""

    payload = {
        "attachments": [
            {
                "color": _COLOURS["idle_gpu"],
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{_ICONS['idle_gpu']} Idle GPU Detected — {job_id}"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Job ID:*\n`{job_id}`"},
                            {"type": "mrkdwn", "text": f"*Instance:*\n`{instance_type}`"},
                            {"type": "mrkdwn", "text": f"*GPU Utilization:*\n`{gpu_util:.1f}%` (threshold: 10%)"},
                            {"type": "mrkdwn", "text": f"*Idle Duration:*\n`{idle_duration_s/60:.1f} min`"},
                            {"type": "mrkdwn", "text": f"*Est. Savings (if paused):*\n`${estimated_savings_usd:.3f}`"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{_timestamp()}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Action:* {action_text}\n{rec_text}"},
                    },
                ],
            }
        ]
    }
    return _send(payload)


def alert_stall(
    job_id: str,
    instance_type: str,
    stall_duration_s: float,
    val_accuracy: float,
    cumulative_cost_usd: float,
    paused: bool,
) -> bool:
    action_text = "Job has been automatically *paused* and checkpointed." if paused else "No automated action taken."

    payload = {
        "attachments": [
            {
                "color": _COLOURS["stall"],
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{_ICONS['stall']} Training Stall Detected — {job_id}"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Job ID:*\n`{job_id}`"},
                            {"type": "mrkdwn", "text": f"*Instance:*\n`{instance_type}`"},
                            {"type": "mrkdwn", "text": f"*Stall Duration:*\n`{stall_duration_s/60:.1f} min`"},
                            {"type": "mrkdwn", "text": f"*Last Val Accuracy:*\n`{val_accuracy:.4f}`"},
                            {"type": "mrkdwn", "text": f"*Cost So Far:*\n`${cumulative_cost_usd:.4f}`"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{_timestamp()}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Action:* {action_text}"},
                    },
                ],
            }
        ]
    }
    return _send(payload)


def alert_oom_crash(
    job_id: str,
    instance_type: str,
    cumulative_cost_usd: float,
    upgrade_rec: str | None,
) -> bool:
    rec_text = f"Recommended upgrade: `{upgrade_rec}` (more GPU memory)." if upgrade_rec else "Review memory requirements."

    payload = {
        "attachments": [
            {
                "color": _COLOURS["oom_crash"],
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{_ICONS['oom_crash']} OOM Crash Detected — {job_id}"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Job ID:*\n`{job_id}`"},
                            {"type": "mrkdwn", "text": f"*Instance:*\n`{instance_type}`"},
                            {"type": "mrkdwn", "text": f"*Cost Wasted:*\n`${cumulative_cost_usd:.4f}`"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{_timestamp()}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Recommendation (requires human approval):* {rec_text}\nJob cannot be auto-resumed — manual restart required.",
                        },
                    },
                ],
            }
        ]
    }
    return _send(payload)


def alert_cost_per_convergence(
    job_id: str,
    instance_type: str,
    cost_per_step: float,
    val_accuracy: float,
    cumulative_cost: float,
) -> bool:
    payload = {
        "attachments": [
            {
                "color": _COLOURS["cost_alert"],
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"{_ICONS['cost_alert']} Cost-per-Convergence Update — {job_id}"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Job ID:*\n`{job_id}`"},
                            {"type": "mrkdwn", "text": f"*Instance:*\n`{instance_type}`"},
                            {"type": "mrkdwn", "text": f"*Cost / Convergence Step:*\n`${cost_per_step:.4f} per Δacc`"},
                            {"type": "mrkdwn", "text": f"*Current Val Accuracy:*\n`{val_accuracy:.4f}`"},
                            {"type": "mrkdwn", "text": f"*Cumulative Cost:*\n`${cumulative_cost:.4f}`"},
                            {"type": "mrkdwn", "text": f"*Time:*\n{_timestamp()}"},
                        ],
                    },
                ],
            }
        ]
    }
    return _send(payload)
