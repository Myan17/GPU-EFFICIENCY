"""
Convergence-Aware Optimization Engine
======================================
Polls Prometheus every POLL_INTERVAL_SECONDS, evaluates three detection rules,
computes the novel cost-per-convergence-step metric, and triggers automated
actions with Slack notifications.

Rules:
  R1 — IDLE_GPU:   gpu_util < threshold for >= IDLE_GPU_DURATION_SECONDS
  R2 — STALL:      time_since_last_progress > STALL_THRESHOLD_SECONDS
  R3 — OOM_CRASH:  job_status == 3 (OOM_CRASH)

Actions:
  - Pause job via simulator control API  (R1, R2)
  - Send Slack alert                     (R1, R2, R3)
  - Log cost-per-convergence-step        (every poll)

Engine also exposes its own Prometheus metrics on port 8001.
"""

import logging
import os
import time
from threading import Thread

import requests
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

from cost_model import CostTracker, hourly_cost
from slack_notifier import (
    alert_idle_gpu,
    alert_oom_crash,
    alert_stall,
    alert_cost_per_convergence,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("optimization_engine")

# ── Config ───────────────────────────────────────────────────────────────────
PROMETHEUS_URL     = os.environ.get("PROMETHEUS_URL",    "http://prometheus:9090")
SIMULATOR_URL      = os.environ.get("SIMULATOR_URL",     "http://workload_simulator:8000")
POLL_INTERVAL      = int(os.environ.get("POLL_INTERVAL_SECONDS",       "30"))
IDLE_THRESHOLD_PCT = float(os.environ.get("IDLE_GPU_THRESHOLD_PERCENT", "10"))
IDLE_DURATION_S    = int(os.environ.get("IDLE_GPU_DURATION_SECONDS",    "900"))
STALL_THRESHOLD_S  = int(os.environ.get("STALL_THRESHOLD_SECONDS",      "600"))
INSTANCE_TYPE      = os.environ.get("INSTANCE_TYPE",    "g4dn.xlarge")
COOLDOWN_S         = int(os.environ.get("ALERT_COOLDOWN_SECONDS",        "300"))

# Job status codes (mirror simulator.py JobStatus enum)
STATUS_RUNNING   = 0
STATUS_IDLE_GPU  = 1
STATUS_STALLED   = 2
STATUS_OOM_CRASH = 3
STATUS_COMPLETED = 4
STATUS_PAUSED    = 5

# ── Engine Prometheus Metrics ─────────────────────────────────────────────────
g_cost_per_step  = Gauge("optimization_cost_per_convergence_step", "$ per unit of val_accuracy gained", ["job_id"])
g_cumulative_cost = Gauge("optimization_cumulative_cost_usd",       "Cumulative simulated cost (USD)",   ["job_id"])
g_idle_duration  = Gauge("optimization_idle_duration_seconds",      "Current idle GPU duration (s)",     ["job_id"])
c_actions        = Counter("optimization_actions_total",            "Actions taken by engine",           ["action", "job_id"])

# ── State ─────────────────────────────────────────────────────────────────────
class EngineState:
    def __init__(self):
        self.cost_tracker:         CostTracker | None = None
        self.idle_start_ts:        float | None       = None   # when idle condition started
        self.last_alert:           dict[str, float]   = {}     # alert_type -> last_sent_ts
        self.job_paused_by_engine: bool               = False
        self.last_val_acc:         float              = 0.0
        self.job_id:               str                = "unknown"
        self.actions_log:          list[dict]         = []


_state = EngineState()


# ── Prometheus Query Helpers ──────────────────────────────────────────────────

def _prom_query(query: str) -> float | None:
    """Execute an instant PromQL query. Returns the scalar value or None."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("result", [])
        if results:
            return float(results[0]["value"][1])
        return None
    except Exception as exc:
        logger.warning("Prometheus query failed ('%s'): %s", query, exc)
        return None


def fetch_metrics() -> dict | None:
    """Fetch all relevant simulator metrics from Prometheus."""
    queries = {
        "gpu_util":         "ml_gpu_utilization_percent",
        "gpu_mem_mb":       "ml_gpu_memory_used_mb",
        "training_loss":    "ml_training_loss",
        "val_accuracy":     "ml_val_accuracy",
        "throughput":       "ml_throughput_samples_per_sec",
        "job_status":       "ml_job_status",
        "stall_seconds":    "ml_time_since_last_progress_seconds",
        "current_epoch":    "ml_current_epoch",
    }
    result = {}
    for key, query in queries.items():
        val = _prom_query(query)
        if val is None:
            logger.debug("Metric unavailable: %s", key)
            result[key] = None
        else:
            result[key] = val
    return result


# ── Simulator Control ─────────────────────────────────────────────────────────

def pause_job(job_id: str) -> bool:
    """Call simulator pause endpoint. Returns True on success."""
    try:
        resp = requests.post(f"{SIMULATOR_URL}/control/pause", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ok"):
            logger.info("Job %s paused. Checkpoint: %s", job_id, data.get("checkpoint"))
            c_actions.labels(action="pause", job_id=job_id).inc()
            _state.actions_log.append({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "action": "pause",
                "job_id": job_id,
            })
            return True
        logger.warning("Pause rejected: %s", data.get("reason"))
        return False
    except Exception as exc:
        logger.error("Failed to pause job: %s", exc)
        return False


# ── Cooldown Check ────────────────────────────────────────────────────────────

def _in_cooldown(alert_type: str) -> bool:
    last = _state.last_alert.get(alert_type, 0)
    return (time.time() - last) < COOLDOWN_S


def _mark_alerted(alert_type: str):
    _state.last_alert[alert_type] = time.time()


# ── Rule Evaluation ───────────────────────────────────────────────────────────

def evaluate_idle_gpu(metrics: dict, job_id: str):
    """R1: GPU utilization below threshold for sustained duration."""
    gpu_util    = metrics.get("gpu_util")
    job_status  = metrics.get("job_status")

    if gpu_util is None or job_status in (STATUS_OOM_CRASH, STATUS_COMPLETED, STATUS_PAUSED):
        _state.idle_start_ts = None
        g_idle_duration.labels(job_id=job_id).set(0)
        return

    if gpu_util < IDLE_THRESHOLD_PCT:
        if _state.idle_start_ts is None:
            _state.idle_start_ts = time.time()
            logger.info("R1: GPU util %.1f%% < %.1f%% — idle timer started", gpu_util, IDLE_THRESHOLD_PCT)

        idle_duration = time.time() - _state.idle_start_ts
        g_idle_duration.labels(job_id=job_id).set(idle_duration)

        logger.info("R1: Idle GPU — %.1f%% for %.1f min (threshold: %.1f min)",
                    gpu_util, idle_duration / 60, IDLE_DURATION_S / 60)

        if idle_duration >= IDLE_DURATION_S and not _in_cooldown("idle_gpu"):
            logger.warning("R1 TRIGGERED: GPU idle for %.1f min — taking action", idle_duration / 60)
            savings = _state.cost_tracker.savings_from_pause(idle_duration) if _state.cost_tracker else 0.0
            paused = False
            if not _state.job_paused_by_engine:
                paused = pause_job(job_id)
                _state.job_paused_by_engine = paused

            alert_idle_gpu(
                job_id=job_id,
                instance_type=INSTANCE_TYPE,
                gpu_util=gpu_util,
                idle_duration_s=idle_duration,
                estimated_savings_usd=savings,
                paused=paused,
                downgrade_rec=_state.cost_tracker.downgrade_recommendation() if _state.cost_tracker else None,
            )
            c_actions.labels(action="alert_idle_gpu", job_id=job_id).inc()
            _mark_alerted("idle_gpu")
    else:
        if _state.idle_start_ts is not None:
            logger.info("R1: GPU util recovered to %.1f%% — idle timer reset", gpu_util)
        _state.idle_start_ts = None
        _state.job_paused_by_engine = False
        g_idle_duration.labels(job_id=job_id).set(0)


def evaluate_stall(metrics: dict, job_id: str):
    """R2: No training progress for STALL_THRESHOLD_SECONDS."""
    stall_s    = metrics.get("stall_seconds")
    val_acc    = metrics.get("val_accuracy")
    job_status = metrics.get("job_status")

    if stall_s is None or job_status in (STATUS_OOM_CRASH, STATUS_COMPLETED, STATUS_PAUSED):
        return

    logger.info("R2: time_since_progress=%.1fs (threshold=%ds)", stall_s, STALL_THRESHOLD_S)

    if stall_s > STALL_THRESHOLD_S and not _in_cooldown("stall"):
        logger.warning("R2 TRIGGERED: Training stalled for %.1f min", stall_s / 60)
        paused = False
        if not _state.job_paused_by_engine:
            paused = pause_job(job_id)
            _state.job_paused_by_engine = paused

        alert_stall(
            job_id=job_id,
            instance_type=INSTANCE_TYPE,
            stall_duration_s=stall_s,
            val_accuracy=val_acc or 0.0,
            cumulative_cost_usd=_state.cost_tracker.cumulative_cost if _state.cost_tracker else 0.0,
            paused=paused,
        )
        c_actions.labels(action="alert_stall", job_id=job_id).inc()
        _mark_alerted("stall")


def evaluate_oom(metrics: dict, job_id: str):
    """R3: Job status is OOM_CRASH."""
    job_status = metrics.get("job_status")
    if job_status is None:
        return

    if int(job_status) == STATUS_OOM_CRASH and not _in_cooldown("oom_crash"):
        logger.error("R3 TRIGGERED: OOM crash detected on %s", job_id)
        alert_oom_crash(
            job_id=job_id,
            instance_type=INSTANCE_TYPE,
            cumulative_cost_usd=_state.cost_tracker.cumulative_cost if _state.cost_tracker else 0.0,
            upgrade_rec=_state.cost_tracker.upgrade_recommendation() if _state.cost_tracker else None,
        )
        c_actions.labels(action="alert_oom_crash", job_id=job_id).inc()
        _mark_alerted("oom_crash")


def update_cost_metrics(metrics: dict, job_id: str):
    """Compute cost-per-convergence-step and update Prometheus gauges."""
    if _state.cost_tracker is None:
        return

    now      = time.time()
    val_acc  = metrics.get("val_accuracy") or 0.0

    _state.cost_tracker.tick(now)
    cost_per_step = _state.cost_tracker.update_convergence(val_acc)
    cumulative    = _state.cost_tracker.cumulative_cost

    g_cost_per_step.labels(job_id=job_id).set(cost_per_step)
    g_cumulative_cost.labels(job_id=job_id).set(cumulative)

    logger.info(
        "Cost: cumulative=$%.4f  cost/convergence=$%.4f/Δacc  val_acc=%.4f",
        cumulative, cost_per_step, val_acc,
    )

    # Periodic cost-per-convergence Slack update (every ~10 min)
    if not _in_cooldown("cost_update") and val_acc > 0.01:
        alert_cost_per_convergence(
            job_id=job_id,
            instance_type=INSTANCE_TYPE,
            cost_per_step=cost_per_step,
            val_accuracy=val_acc,
            cumulative_cost=cumulative,
        )
        _mark_alerted("cost_update")


# ── Main Poll Loop ────────────────────────────────────────────────────────────

def get_job_id_from_prometheus() -> str:
    """Extract job_id label from a live metric."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "ml_job_status"},
            timeout=5,
        )
        results = resp.json().get("data", {}).get("result", [])
        if results:
            return results[0]["metric"].get("job_id", "unknown")
    except Exception:
        pass
    return "unknown"


def poll_loop():
    """Main engine loop — runs every POLL_INTERVAL seconds."""
    logger.info(
        "Optimization engine starting — poll_interval=%ds  idle_threshold=%.0f%%  "
        "idle_duration=%ds  stall_threshold=%ds",
        POLL_INTERVAL, IDLE_THRESHOLD_PCT, IDLE_DURATION_S, STALL_THRESHOLD_S,
    )

    # Wait for Prometheus to have data
    for attempt in range(30):
        if _prom_query("ml_gpu_utilization_percent") is not None:
            break
        logger.info("Waiting for simulator metrics in Prometheus... (%d/30)", attempt + 1)
        time.sleep(10)

    job_id = get_job_id_from_prometheus()
    _state.job_id = job_id
    _state.cost_tracker = CostTracker(instance_type=INSTANCE_TYPE)
    logger.info("Tracking job: %s on %s ($%.3f/hr)", job_id, INSTANCE_TYPE, hourly_cost(INSTANCE_TYPE))

    # Set cost_update cooldown to 10 min
    global COOLDOWN_S
    _state.last_alert["cost_update"] = time.time() - COOLDOWN_S  # allow first alert

    while True:
        try:
            metrics = fetch_metrics()
            if not any(v is not None for v in metrics.values()):
                logger.warning("No metrics available — is the simulator running?")
                time.sleep(POLL_INTERVAL)
                continue

            evaluate_idle_gpu(metrics, job_id)
            evaluate_stall(metrics, job_id)
            evaluate_oom(metrics, job_id)
            update_cost_metrics(metrics, job_id)

        except Exception as exc:
            logger.error("Poll loop error: %s", exc, exc_info=True)

        time.sleep(POLL_INTERVAL)


# ── FastAPI Status API ────────────────────────────────────────────────────────
app = FastAPI(title="Optimization Engine", version="1.0.0")


@app.on_event("startup")
def on_startup():
    Thread(target=poll_loop, daemon=True, name="poll-loop").start()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/status")
def status():
    tracker = _state.cost_tracker
    return {
        "engine":                    "optimization_engine",
        "job_id":                    _state.job_id,
        "instance_type":             INSTANCE_TYPE,
        "poll_interval_s":           POLL_INTERVAL,
        "idle_threshold_pct":        IDLE_THRESHOLD_PCT,
        "idle_duration_threshold_s": IDLE_DURATION_S,
        "stall_threshold_s":         STALL_THRESHOLD_S,
        "job_paused_by_engine":      _state.job_paused_by_engine,
        "cumulative_cost_usd":       round(tracker.cumulative_cost, 4) if tracker else 0,
        "cost_per_convergence_step": round(tracker.cost_per_convergence_step, 4) if tracker else 0,
        "recent_actions":            _state.actions_log[-10:],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
