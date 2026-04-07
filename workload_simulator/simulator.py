"""
Synthetic ML Training Job Simulator
====================================
Simulates a GPU training workload emitting realistic metrics to Prometheus
and logging convergence data to MLflow.

Scenarios:
  NORMAL     — healthy training run
  IDLE_GPU   — GPU utilization crashes after fault epoch (data loader bottleneck)
  STALL      — loss plateaus, time_since_last_progress grows
  OOM_CRASH  — job crashes with OOM status after fault epoch

Control API (port 8000):
  GET  /metrics              — Prometheus scrape endpoint
  GET  /status               — current job state (JSON)
  POST /control/pause        — checkpoint + suspend job
  POST /control/resume       — resume paused job
  POST /control/reset        — restart training from epoch 0
  POST /control/inject/{s}   — hot-swap scenario mid-run
"""

import math
import os
import random
import threading
import time
from enum import IntEnum

import mlflow
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

# ── Prometheus Metrics ──────────────────────────────────────────────────────
LABELS = ["job_id", "instance_type"]

g_gpu_util    = Gauge("ml_gpu_utilization_percent",      "GPU utilization %",          LABELS)
g_gpu_mem     = Gauge("ml_gpu_memory_used_mb",           "GPU memory used (MB)",       LABELS)
g_train_loss  = Gauge("ml_training_loss",                "Current training loss",      LABELS)
g_val_acc     = Gauge("ml_val_accuracy",                 "Validation accuracy [0-1]",  LABELS)
g_throughput  = Gauge("ml_throughput_samples_per_sec",   "Training throughput",        LABELS)
g_job_status  = Gauge("ml_job_status",                   "Job status code (see docs)", LABELS)
g_stall_time  = Gauge("ml_time_since_last_progress_seconds", "Seconds since last val-acc improvement", LABELS)
g_epoch       = Gauge("ml_current_epoch",                "Current training epoch",     LABELS)


class JobStatus(IntEnum):
    RUNNING   = 0
    IDLE_GPU  = 1
    STALLED   = 2
    OOM_CRASH = 3
    COMPLETED = 4
    PAUSED    = 5


# ── Shared State ────────────────────────────────────────────────────────────
class JobState:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock if hasattr(self, "lock") else _noop():
            self.job_id        = os.environ.get("JOB_ID", "job-001")
            self.scenario      = os.environ.get("SCENARIO", "NORMAL")
            self.instance_type = os.environ.get("INSTANCE_TYPE", "g4dn.xlarge")
            self.total_epochs  = int(os.environ.get("TOTAL_EPOCHS", "50"))
            self.epoch_dur     = float(os.environ.get("EPOCH_DURATION_SECONDS", "8"))
            self.fault_after   = int(os.environ.get("FAULT_INJECT_AFTER_EPOCH", "10"))

            self.current_epoch         = 0
            self.status                = JobStatus.RUNNING
            self.gpu_util              = 0.0
            self.gpu_mem_mb            = 0.0
            self.training_loss         = 0.0
            self.val_accuracy          = 0.0
            self.throughput            = 0.0
            self.time_since_progress   = 0.0
            self.best_val_acc          = 0.0
            self.last_progress_ts      = time.time()
            self.paused                = False
            self.checkpoint_written    = False
            self.mlflow_run_id         = None


import contextlib

@contextlib.contextmanager
def _noop():
    yield


state = JobState()

# ── Training Simulation ─────────────────────────────────────────────────────

def _noisy(value: float, noise_pct: float = 0.05) -> float:
    return value * (1 + random.uniform(-noise_pct, noise_pct))


def simulate_epoch(epoch: int, total: int, scenario: str, fault_after: int) -> dict:
    """Return simulated metric values for one epoch."""
    progress = epoch / max(total - 1, 1)          # 0 → 1 over training

    # ── Healthy baseline ────────────────────────────────────────
    base_loss    = 2.5 * math.exp(-3.5 * progress) + 0.05
    base_acc     = 0.95 * (1 - math.exp(-4 * progress))
    base_gpu     = _noisy(78.0, 0.08)             # ~78% with 8% noise
    base_mem     = _noisy(9_800, 0.03)            # ~9.8 GB in MB
    base_thruput = _noisy(1_400, 0.05)            # ~1400 samples/s

    fault_active = epoch >= fault_after

    if scenario == "IDLE_GPU" and fault_active:
        # GPU drops to near-zero; data loader starvation
        base_gpu    = _noisy(2.5, 0.3)
        base_mem    = _noisy(1_500, 0.05)
        base_thruput = _noisy(40, 0.2)
        # Training stalls too
        base_loss   = base_loss * _noisy(1.0, 0.01)
        base_acc    = base_acc  * _noisy(1.0, 0.005)

    elif scenario == "STALL" and fault_active:
        # Loss plateaus — progress frozen
        stall_progress = fault_after / max(total - 1, 1)
        base_loss  = 2.5 * math.exp(-3.5 * stall_progress) + 0.05 + _noisy(0.0, 0.002)
        base_acc   = 0.95 * (1 - math.exp(-4 * stall_progress)) + _noisy(0.0, 0.001)
        base_gpu   = _noisy(72.0, 0.06)  # GPU still running, just not converging

    elif scenario == "OOM_CRASH" and fault_active:
        # Return OOM sentinel — caller handles the crash
        return {"oom": True}

    return {
        "loss":       round(_noisy(base_loss, 0.02), 4),
        "val_acc":    round(min(_noisy(base_acc, 0.015), 0.999), 4),
        "gpu_util":   round(max(base_gpu, 0), 2),
        "gpu_mem_mb": round(max(base_mem, 0), 1),
        "throughput": round(max(base_thruput, 0), 1),
    }


def _write_checkpoint(epoch: int):
    """Simulate checkpointing — in production this writes to S3/disk."""
    path = f"/tmp/checkpoint_job_{state.job_id}_epoch_{epoch}.pt"
    with open(path, "w") as f:
        f.write(f"epoch={epoch} val_acc={state.val_accuracy:.4f}\n")
    print(f"[simulator] Checkpoint written → {path}", flush=True)
    return path


def _update_prometheus():
    labels = [state.job_id, state.instance_type]
    g_gpu_util.labels(*labels).set(state.gpu_util)
    g_gpu_mem.labels(*labels).set(state.gpu_mem_mb)
    g_train_loss.labels(*labels).set(state.training_loss)
    g_val_acc.labels(*labels).set(state.val_accuracy)
    g_throughput.labels(*labels).set(state.throughput)
    g_job_status.labels(*labels).set(int(state.status))
    g_stall_time.labels(*labels).set(state.time_since_progress)
    g_epoch.labels(*labels).set(state.current_epoch)


def training_loop():
    """Background thread running the simulated training job."""
    # Wait for MLflow to be ready (best-effort — training continues without it)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    # Short per-request timeout so hung workers don't block training startup
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")

    mlflow_available = False
    for attempt in range(10):
        try:
            mlflow.set_experiment("gpu-efficiency-phase1")
            mlflow_available = True
            break
        except Exception:
            print(f"[simulator] Waiting for MLflow... ({attempt+1}/10)", flush=True)
            time.sleep(3)

    if not mlflow_available:
        print("[simulator] MLflow unavailable — continuing without experiment tracking.", flush=True)

    def _run_training(active_run=None):
        for epoch in range(state.total_epochs):
            # ── Pause check ─────────────────────────────────────
            while state.paused:
                time.sleep(1)
                if state.status == JobStatus.COMPLETED:
                    return

            with state.lock:
                state.current_epoch = epoch
                scenario    = state.scenario
                fault_after = state.fault_after

            metrics = simulate_epoch(epoch, state.total_epochs, scenario, fault_after)

            # ── OOM crash ───────────────────────────────────────
            if metrics.get("oom"):
                with state.lock:
                    state.status    = JobStatus.OOM_CRASH
                    state.gpu_util  = 0.0
                    state.gpu_mem_mb = 0.0
                    state.throughput = 0.0
                _update_prometheus()
                print(f"[simulator] OOM_CRASH at epoch {epoch}", flush=True)
                if active_run:
                    mlflow.log_metric("job_status", int(JobStatus.OOM_CRASH), step=epoch)
                return

            # ── Update state ─────────────────────────────────────
            with state.lock:
                state.gpu_util      = metrics["gpu_util"]
                state.gpu_mem_mb    = metrics["gpu_mem_mb"]
                state.training_loss = metrics["loss"]
                state.val_accuracy  = metrics["val_acc"]
                state.throughput    = metrics["throughput"]

                if scenario == "IDLE_GPU" and epoch >= fault_after:
                    state.status = JobStatus.IDLE_GPU
                elif scenario == "STALL" and epoch >= fault_after:
                    state.status = JobStatus.STALLED
                else:
                    state.status = JobStatus.RUNNING

                if metrics["val_acc"] > state.best_val_acc + 0.001:
                    state.best_val_acc     = metrics["val_acc"]
                    state.last_progress_ts = time.time()
                state.time_since_progress = time.time() - state.last_progress_ts

            _update_prometheus()

            # ── MLflow logging (skipped if MLflow unavailable) ──
            if active_run:
                mlflow.log_metrics({
                    "training_loss":              metrics["loss"],
                    "val_accuracy":               metrics["val_acc"],
                    "gpu_utilization_percent":    metrics["gpu_util"],
                    "gpu_memory_used_mb":         metrics["gpu_mem_mb"],
                    "throughput_samples_per_sec": metrics["throughput"],
                    "time_since_last_progress":   state.time_since_progress,
                    "job_status":                 int(state.status),
                }, step=epoch)

            print(
                f"[simulator] epoch={epoch:03d} loss={metrics['loss']:.4f} "
                f"val_acc={metrics['val_acc']:.4f} gpu={metrics['gpu_util']:.1f}% "
                f"status={state.status.name}",
                flush=True,
            )

            time.sleep(state.epoch_dur)

        # ── Completed ─────────────────────────────────────────────
        with state.lock:
            state.status = JobStatus.COMPLETED
        _update_prometheus()
        print("[simulator] Training completed.", flush=True)

    if mlflow_available:
        try:
            with mlflow.start_run(run_name=f"{state.job_id}_{state.scenario}") as run:
                state.mlflow_run_id = run.info.run_id
                mlflow.log_params({
                    "job_id":        state.job_id,
                    "scenario":      state.scenario,
                    "instance_type": state.instance_type,
                    "total_epochs":  state.total_epochs,
                })
                _run_training(active_run=run)
        except Exception as exc:
            print(f"[simulator] MLflow run failed ({exc}) — restarting training without tracking.", flush=True)
            state.mlflow_run_id = None
            _run_training(active_run=None)
    else:
        _run_training(active_run=None)


# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Workload Simulator", version="1.0.0")
_bg_thread: threading.Thread | None = None


def _start_training_thread():
    global _bg_thread
    _bg_thread = threading.Thread(target=training_loop, daemon=True, name="training-loop")
    _bg_thread.start()


@app.on_event("startup")
def on_startup():
    _start_training_thread()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/status")
def status():
    with state.lock:
        return {
            "job_id":                     state.job_id,
            "scenario":                   state.scenario,
            "instance_type":              state.instance_type,
            "current_epoch":              state.current_epoch,
            "total_epochs":               state.total_epochs,
            "status":                     state.status.name,
            "gpu_utilization_percent":    round(state.gpu_util, 2),
            "gpu_memory_used_mb":         round(state.gpu_mem_mb, 1),
            "training_loss":              round(state.training_loss, 4),
            "val_accuracy":               round(state.val_accuracy, 4),
            "throughput_samples_per_sec": round(state.throughput, 1),
            "time_since_last_progress_s": round(state.time_since_progress, 1),
            "mlflow_run_id":              state.mlflow_run_id,
        }


@app.post("/control/pause")
def pause_job():
    with state.lock:
        if state.status in (JobStatus.COMPLETED, JobStatus.OOM_CRASH):
            return {"ok": False, "reason": f"Cannot pause — job is {state.status.name}"}
        checkpoint_path = _write_checkpoint(state.current_epoch)
        state.paused             = True
        state.checkpoint_written = True
        state.status             = JobStatus.PAUSED
    _update_prometheus()
    return {"ok": True, "action": "paused", "checkpoint": checkpoint_path}


@app.post("/control/resume")
def resume_job():
    with state.lock:
        if not state.paused:
            return {"ok": False, "reason": "Job is not paused"}
        state.paused  = False
        state.status  = JobStatus.RUNNING
    return {"ok": True, "action": "resumed"}


@app.post("/control/reset")
def reset_job():
    global _bg_thread
    with state.lock:
        state.paused = True   # stop current loop
    time.sleep(0.5)
    state.reset()
    _start_training_thread()
    return {"ok": True, "action": "reset"}


@app.post("/control/inject/{scenario}")
def inject_scenario(scenario: str):
    valid = {"NORMAL", "IDLE_GPU", "STALL", "OOM_CRASH"}
    if scenario not in valid:
        return {"ok": False, "reason": f"Unknown scenario. Valid: {valid}"}
    with state.lock:
        state.scenario    = scenario
        state.fault_after = state.current_epoch  # fault triggers immediately
    return {"ok": True, "action": f"scenario changed to {scenario}", "effective_from_epoch": state.current_epoch}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
