# Convergence-Aware GPU Efficiency — Phase 1 (Local Simulation)

[![CI](https://github.com/Myan17/GPU-EFFICIENCY/actions/workflows/ci.yml/badge.svg)](https://github.com/Myan17/GPU-EFFICIENCY/actions/workflows/ci.yml)

Detects idle GPUs, stalled training, and OOM crashes, then pauses the job before
it burns another hour of instance time. **54 tests · 69% coverage · measured
57–60% spend avoided on faulted jobs, 0 false positives on healthy ones.**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Docker Compose (local)                                     │
│                                                             │
│  ┌──────────────────┐    metrics     ┌──────────────────┐  │
│  │ workload_        │ ──────────────▶│   prometheus     │  │
│  │ simulator :8000  │                │   :9090          │  │
│  │                  │    epochs      ├──────────────────┤  │
│  │ · synthetic GPU  │ ──────────────▶│   mlflow         │  │
│  │   training loop  │                │   :5000          │  │
│  │ · control API    │                └──────────────────┘  │
│  │ · fault inject   │                        │             │
│  └──────────────────┘                 scrapes│             │
│                                              ▼             │
│  ┌──────────────────┐    pause/    ┌──────────────────┐   │
│  │ optimization_    │◀─────────────│ optimization_    │   │
│  │ simulator :8000  │   resume     │ engine :8001     │   │
│  └──────────────────┘              │                  │   │
│                                    │ · R1 idle GPU    │   │
│  ┌──────────────────┐              │ · R2 stall       │   │
│  │   grafana :3000  │◀─────────────│ · R3 OOM crash   │   │
│  │ (dashboards)     │   metrics    │ · cost/conv step │   │
│  └──────────────────┘              └────────┬─────────┘   │
│                                             │              │
└─────────────────────────────────────────────┼──────────────┘
                                              │ Slack alerts
                                              ▼
                                      ┌───────────────┐
                                      │  Slack channel│
                                      └───────────────┘
```

## Quick Start

### 1. Prerequisites
- Docker Desktop installed and running
- `make` (comes with Xcode Command Line Tools on Mac: `xcode-select --install`)

### 2. Slack Setup (Required for alerts)

**Step 1 — Create a Slack App**
1. Go to https://api.slack.com/apps → "Create New App" → "From scratch"
2. Name: `GPU Efficiency Bot`  |  Pick your workspace → "Create App"

**Step 2 — Enable Incoming Webhooks**
1. In your app's sidebar: "Incoming Webhooks" → toggle **On**
2. Click "Add New Webhook to Workspace"
3. Choose a channel (e.g. `#gpu-alerts`) → "Allow"
4. Copy the Webhook URL: `https://hooks.slack.com/services/T.../B.../...`

**Step 3 — Configure .env**
```bash
cp .env.example .env
# Edit .env and paste your webhook URL into SLACK_WEBHOOK_URL=
```

> **Dev mode:** If SLACK_WEBHOOK_URL is empty, alerts are printed to the engine's console log instead of being sent to Slack — useful for testing without Slack.

### 3. Start the Stack

```bash
cp .env.example .env   # configure your webhook URL
make up
```

Open your dashboards:
| Service    | URL                           | Credentials |
|------------|-------------------------------|-------------|
| Grafana    | http://localhost:3000         | admin/admin |
| Prometheus | http://localhost:9090         | —           |
| MLflow     | http://localhost:5000         | —           |
| Simulator  | http://localhost:8000/status  | —           |
| Engine     | http://localhost:8001/status  | —           |

## Fault Injection — Testing Each Detection Rule

Run these after `make up`. Each injects a fault mid-run and lets you watch the engine respond.

### Rule 1 — Idle GPU Detection
```bash
make inject-idle
# GPU util drops to ~2.5%
# After IDLE_GPU_DURATION_SECONDS (default 900s, override in .env), engine:
#   1. Pauses + checkpoints the job
#   2. Sends Slack alert with downgrade recommendation
```

To demo faster (2-minute idle window):
```bash
IDLE_GPU_DURATION_SECONDS=120 make up
make inject-idle
```

### Rule 2 — Stall Detection
```bash
make inject-stall
# Training loss/accuracy freeze
# After STALL_THRESHOLD_SECONDS (default 600s), engine pauses + alerts
```

### Rule 3 — OOM Crash
```bash
make inject-oom
# Job status → OOM_CRASH
# Engine immediately alerts with instance upgrade recommendation
# (No auto-resume — requires human action)
```

### Back to Normal
```bash
make inject-normal   # restore healthy metrics
make resume          # resume if paused
```

## Configuration Reference

| Variable                    | Default        | Description                                    |
|-----------------------------|----------------|------------------------------------------------|
| `SCENARIO`                  | `NORMAL`       | Initial scenario (NORMAL/IDLE_GPU/STALL/OOM_CRASH) |
| `TOTAL_EPOCHS`              | `50`           | Number of training epochs                      |
| `EPOCH_DURATION_SECONDS`    | `8`            | Real seconds per simulated epoch               |
| `FAULT_INJECT_AFTER_EPOCH`  | `10`           | Epoch at which fault activates                 |
| `INSTANCE_TYPE`             | `g4dn.xlarge`  | Affects cost calculations ($0.526/hr)          |
| `POLL_INTERVAL_SECONDS`     | `30`           | How often the engine queries Prometheus        |
| `IDLE_GPU_THRESHOLD_PERCENT`| `10`           | Below this % → idle condition starts           |
| `IDLE_GPU_DURATION_SECONDS` | `900`          | Seconds idle before action triggers (15 min)   |
| `STALL_THRESHOLD_SECONDS`   | `600`          | Seconds with no val_acc progress before action |
| `ALERT_COOLDOWN_SECONDS`    | `300`          | Min seconds between repeated alerts            |
| `SLACK_WEBHOOK_URL`         | (empty)        | Slack incoming webhook (empty = console only)  |

## Measured results

Run it yourself — no cloud account, no Docker:

```bash
pip install -r requirements-dev.txt
python benchmarks/savings_benchmark.py
```

The benchmark replays the simulator's own fault models through the engine's own
rule thresholds and prices both outcomes with the cost model. A 50-epoch job at
8 min/epoch is 6.7 h unmanaged; the fault lands at epoch 20 (160 min).

| Scenario | Detected at | Unmanaged | Managed | Saved | |
|---|---|---|---|---|---|
| NORMAL | never | $3.51 | $3.51 | $0.00 | **0%** |
| IDLE_GPU | 170.5 min | $3.51 | $1.49 | $2.01 | **57.4%** |
| STALL | 170.5 min | $3.51 | $1.49 | $2.01 | **57.4%** |
| OOM_CRASH | 160.0 min | $3.51 | $1.40 | $2.10 | **60.0%** |

*(g4dn.xlarge at $0.526/hr. The saved **percentage** is identical on p3.2xlarge
and p3.8xlarge — only the dollar figure scales, to $11.70 and $46.82.)*

Across the nine faulted runs in the matrix: **$184.37 saved, 35.0 GPU-hours
avoided.** Across the three healthy runs: **$0.00 — the engine paused nothing.**
No false positives is the result that matters most; a false pause would kill
real training.

Two honest notes on the numbers:

1. **IDLE_GPU is caught by the stall rule, not the idle rule.** When the GPU
   goes idle, accuracy stops moving as well, so R2's 10-minute window expires
   before R1's 15-minute window does. Both detect it; R2 gets there first.
2. **This measures detection latency priced at real AWS on-demand rates, on
   simulated jobs.** It is not a measurement of a real training run, and the
   README does not claim it is. Phase 2 (below) is where that would come from.

The figures above are pinned by `tests/test_savings_benchmark.py`, so changing a
threshold or a price fails CI until this table is updated.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

54 tests, gated at 65% coverage in CI:

| Suite | Tests | Covers |
|---|---|---|
| `test_engine_rules.py` | 23 | R1/R2/R3 trigger, non-trigger, cooldown, cooldown expiry, terminal-status guards, unreachable-Prometheus guard |
| `test_cost_model.py` | 13 | pricing table, per-second accrual, cost-per-convergence-step, stall behaviour, divide-by-zero guard, upgrade/downgrade map integrity |
| `test_slack_notifier.py` | 9 | payload contents, no-webhook no-op, Slack-outage tolerance |
| `test_savings_benchmark.py` | 9 | detection latencies, published savings, determinism, no false positives |

Nothing in the suite makes a network call or needs a running container.

## Novel Metric — Cost per Convergence Step

```
cost_per_convergence_step = cumulative_cost_usd / Δval_accuracy
```

- `cumulative_cost_usd`: elapsed real time × (instance $/hr ÷ 3600)
- `Δval_accuracy`: validation accuracy gained since training started
- Visible in Grafana → "Cost per Convergence Step ($/Δacc)" panel
- Also exposed via engine's `/metrics` as `optimization_cost_per_convergence_step`

## Useful Commands

```bash
make logs           # all service logs
make logs-engine    # optimization engine only
make logs-sim       # simulator only
make status         # JSON status from both services
make pause          # manually pause the training job
make resume         # resume a paused job
make reset          # restart training from epoch 0
make down           # stop all services
make clean          # stop + remove all volumes (fresh start)
```

## Phase 2 Preview (AWS)

Phase 2 will replace:
- Docker Compose → EKS (Kubernetes)
- Simulated GPU metrics → DCGM Exporter on real GPU nodes
- Local event loop → AWS EventBridge + Lambda
- Local checkpoint files → S3
- Console pause → `kubectl` job suspend
