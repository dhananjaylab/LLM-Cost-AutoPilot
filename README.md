# LLM-Cost-AutoPilot

`LLM-Cost-AutoPilot` is an intelligent routing layer that sits in front of multiple LLM providers. It analyzes incoming requests, estimates their complexity, routes them to the cheapest capable model that meets quality criteria, and continuously validates those routing decisions through offline verification and retraining workflows.

---

## 🏗️ Architecture Layout

The repository is structured as a monorepo featuring a shared core library, FastAPI web application, Celery background worker, and supporting infrastructure.

```text
├── apps/
│   ├── api/                      # FastAPI Web Application
│   │   └── llm_autopilot_api/
│   │       ├── main.py           # Lifespan handlers, CORS, OTel, Middleware
│   │       ├── dependencies.py   # Database session, configs & registry deps
│   │       └── routers/          # API Routers (e.g., health checking, router logic)
│   └── worker/                   # Celery worker process
│       └── llm_autopilot_worker/
│           ├── main.py           # Celery configuration, queues & beat scheduler
│           └── tasks/            # Verification & retraining background tasks
│
├── libs/
│   └── core/                     # Shared codebase & core capabilities
│       └── llm_autopilot_core/
│           ├── config.py         # Pydantic configuration loader
│           ├── schemas.py        # Shared Pydantic data schemas
│           ├── database.py       # SQL Alchemy database connections (Async)
│           ├── logging.py        # Centralized structured logging setup
│           ├── metrics.py        # OpenTelemetry & Prometheus metric collection
│           └── registry.py       # Model & provider registries
│
├── infra/                        # Infrastructure and deployment configuration
│   ├── Dockerfile.api            # Docker container builder for FastAPI
│   ├── Dockerfile.worker         # Docker container builder for Celery
│   ├── docker-compose.yml        # Setup Orchestration (9 containers: Postgres, Redis, Prometheus, Grafana, API, Worker, Flower, Ollama, etc.)
│   ├── prometheus/               # Prometheus configuration
│   ├── grafana/                  # Grafana dashboard & datasource provisioning
│   └── postgres/                 # PostgreSQL initialization scripts
│
└── configs/                      # Shared configuration files (routing definitions & models)
    ├── routing.yaml
    └── models.yaml
```

## ⚡ Quick Start

Follow these steps to unpack the repository, configure variables, install dependencies, and run services locally:

### 1. Navigate
```powershell
cd llm-cost-autopilot
```

### 2. Configure Environments
Create a local configuration file from the template:
```powershell
Copy-Item .env.example .env
```

### 3. Install Python Environment & Dependencies
This project uses **`uv`** for lightning-fast environment and package management.

#### A. Install `uv` (if not already installed)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Restart your terminal or refresh environment variables afterwards.
```

#### B. Sync Dependencies
Create the virtual environment and install all libraries (including dev dependencies like `pytest`, `ruff`, etc.):
```powershell
uv sync --dev
```
*This command automatically creates a virtual environment folder named `.venv` if one does not exist, and synchronizes packages to match `uv.lock`.*

#### C. Activate the Virtual Environment (Optional)
```powershell
.venv\Scripts\Activate.ps1
```

#### D. Install Pre-Commit Hooks
Set up git hook checks to auto-format and lint on every commit:
```powershell
uv run pre-commit install
```

#### E. Code Quality & Linting (Optional)

The project uses multiple tools for code quality checks. While these run automatically via pre-commit hooks on each commit, you can also run them manually:

**Run all linting and formatting checks:**
```powershell
uv run pre-commit run --all-files
```

**Run specific tools:**

* **Ruff** (Fast Python linter with auto-fix):
  ```powershell
  uv run ruff check --fix .
  ```

* **Black** (Code formatter):
  ```powershell
  uv run black .
  ```

* **Mypy** (Static type checker):
  ```powershell
  uv run mypy libs/core/llm_autopilot_core/providers/
  ```

* **Pyupgrade** (Remove redundant type casts and upgrades syntax):
  ```powershell
  uv run pyupgrade --py311-plus libs/core/llm_autopilot_core/providers/
  ```

> [!TIP]
> **Auto-fixing mypy errors:** Use `pyupgrade` or `ruff` to automatically fix redundant type casts and other code quality issues:
> ```powershell
> uv run ruff check --select UP --fix libs/
> ```

### 4. Boot Core Services (Local vs Docker)
If using local engines/databases (like Neon and Redis Cloud), ensure they are active.
Otherwise, if running local containers:
```powershell
docker compose up -d postgres redis
```

### 5. Run Database Migrations
Deploy schema configurations using Alembic (runs as a no-op until database models are defined in later phases):
```powershell
uv run alembic upgrade head
```

### 6. Start the Applications (Concurrently - Recommended)
Instead of starting each server and service manually in a separate window, you can run them all at once using one of the following methods:

#### Method 1: VS Code Tasks (Integrated Terminal)
If you are using VS Code, a workspace task runner has been configured.
1. Press `Ctrl + Shift + B` (or open the Command Palette `Ctrl + Shift + P` and search for **Tasks: Run Build Task**).
2. Select **`Start All Services`**.
3. VS Code will spin up all 7 applications and observability components in their own integrated terminal tabs.

#### Method 2: PowerShell Script (External Windows)
You can run a single script to open 7 separate PowerShell windows for each service:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-all.ps1
```

---

### 7. Start the Applications (Manually)

#### A. Core Application & Background Services
Run the following commands in separate PowerShell windows:
* **FastAPI API Gateway:**
  ```powershell
  uvicorn llm_autopilot_api.main:app --reload --port 8000
  ```
* **Celery Worker:**
  ```powershell
  celery -A llm_autopilot_worker.main worker -Q verification,retraining -P solo --loglevel DEBUG -E
  ```
* **Celery Beat Scheduler:**
  ```powershell
  celery -A llm_autopilot_worker.main beat --loglevel DEBUG
  ```
* **Celery Flower Dashboard:**
  ```powershell
  celery -A llm_autopilot_worker.main flower --port=5555
  ```

> [!NOTE]
> * **`-P solo`**: Required on Windows to prevent `PermissionError: [WinError 5] Access is denied` errors caused by Celery's default multiprocessing library attempting to create process forks.
> * **`-E`**: Enables task events publishing, allowing the worker status and jobs to show up inside the Flower dashboard.

## 📊 Monitoring & Services Dashboard URLs

Use the local observability stack as a small service mesh for the app: FastAPI emits metrics and traces, Prometheus scrapes metrics, Tempo stores traces, and Grafana becomes the single operator console for dashboards and trace exploration.

### 1. Start the Observability Services

Open separate PowerShell windows and start the stack in dependency order:

```powershell
# Terminal 1 -- Tempo (start first)
powershell -ExecutionPolicy Bypass -File scripts\obs-tempo.ps1

# Terminal 2 -- Prometheus
powershell -ExecutionPolicy Bypass -File scripts\obs-prometheus.ps1

# Terminal 3 -- Grafana (after MSI install)
powershell -ExecutionPolicy Bypass -File scripts\obs-grafana.ps1

```

### 2. Verify Each Service

Use `curl.exe` on Windows so the command behaves like standard curl instead of the PowerShell alias:

```powershell
# Tempo ready?
curl.exe http://localhost:3200/ready

# Prometheus targets?
curl.exe http://localhost:9090/api/v1/targets

# Grafana health?
curl.exe http://localhost:3000/api/health

# App metrics flowing?
curl.exe http://localhost:8000/metrics
```

### 3. Canonical Local Endpoints

* **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **API Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)
* **Prometheus UI**: [http://localhost:9090](http://localhost:9090)
* **Grafana UI**: [http://localhost:3000](http://localhost:3000) using `admin` / `autopilot`
* **Flower UI**: [http://localhost:5555](http://localhost:5555)
* **Tempo OTLP HTTP ingest**: [http://localhost:4318](http://localhost:4318)
* **Tempo health**: [http://localhost:3200/ready](http://localhost:3200/ready)
* **Tempo metrics**: [http://localhost:3200/metrics](http://localhost:3200/metrics)
* **RedisInsight**: [http://localhost:8001](http://localhost:8001)

> Note: `http://localhost:3200/` can return `404 Not Found` even when Tempo is running correctly. Use `/metrics` or `/ready` to verify the service.

## ⚙️ Configuration Tuning (`.env`)

Adjust routing heuristics, similarity threshold criteria, and verification parameters via your local `.env`:
* `CACHE_SIMILARITY_THRESHOLD`: Semantic caching proximity lookup (default `0.92`).
* `CLASSIFIER_CONFIDENCE_THRESHOLD`: Auto-escalate threshold when routing classification falls short.
* `VERIFICATION_SAMPLE_RATE_LOW_CONFIDENCE`: Fraction of low-confidence routes audited by default.
* `VERIFICATION_SAMPLE_RATE_HIGH_CONFIDENCE`: Fraction of high-confidence routes sampled for evaluation.
