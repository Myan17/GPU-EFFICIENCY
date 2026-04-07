.PHONY: up down logs restart status inject-normal inject-idle inject-stall inject-oom build clean

# ── Lifecycle ────────────────────────────────────────────────────
up:
	docker compose --env-file .env up -d --build
	@echo ""
	@echo "  Services:"
	@echo "    Grafana     → http://localhost:3000  (admin/admin)"
	@echo "    Prometheus  → http://localhost:9090"
	@echo "    MLflow      → http://localhost:5001"
	@echo "    Simulator   → http://localhost:8000/status"
	@echo "    Engine      → http://localhost:8001/status"
	@echo ""

down:
	docker compose down

restart:
	docker compose restart

build:
	docker compose build --no-cache

clean:
	docker compose down -v --remove-orphans

# ── Observability ────────────────────────────────────────────────
logs:
	docker compose logs -f

logs-sim:
	docker compose logs -f workload_simulator

logs-engine:
	docker compose logs -f optimization_engine

status:
	@curl -s http://localhost:8000/status | python3 -m json.tool
	@echo ""
	@curl -s http://localhost:8001/status | python3 -m json.tool

# ── Scenario Injection (call after `make up`) ────────────────────
inject-normal:
	@echo "Injecting NORMAL scenario..."
	curl -s -X POST http://localhost:8000/control/inject/NORMAL | python3 -m json.tool

inject-idle:
	@echo "Injecting IDLE_GPU scenario (GPU util will drop below 10%)..."
	curl -s -X POST http://localhost:8000/control/inject/IDLE_GPU | python3 -m json.tool

inject-stall:
	@echo "Injecting STALL scenario (training progress will freeze)..."
	curl -s -X POST http://localhost:8000/control/inject/STALL | python3 -m json.tool

inject-oom:
	@echo "Injecting OOM_CRASH scenario (job will crash with OOM)..."
	curl -s -X POST http://localhost:8000/control/inject/OOM_CRASH | python3 -m json.tool

# ── Job Control ──────────────────────────────────────────────────
pause:
	curl -s -X POST http://localhost:8000/control/pause | python3 -m json.tool

resume:
	curl -s -X POST http://localhost:8000/control/resume | python3 -m json.tool

reset:
	curl -s -X POST http://localhost:8000/control/reset | python3 -m json.tool
