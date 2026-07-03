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

### 1. Extract and Navigate
On Linux/macOS:
```bash
tar -xzf llm-cost-autopilot-phase0.tar.gz
cd llm-cost-autopilot
```
On Windows (PowerShell):
```powershell
tar -xzf llm-cost-autopilot-phase0.tar.gz
cd llm-cost-autopilot
```

### 2. Configure Environments
Create a local configuration file from the template:
On Linux/macOS:
```bash
cp .env.example .env
```
On Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

#### ☁️ Using Cloud Databases (NeonDB & Redis Cloud)
If running locally without Docker:
* **Postgres (NeonDB)**: Neon connections require SSL and the `asyncpg` driver. Modify the connection URL to prefix with `postgresql+asyncpg://`:
  ```ini
  DATABASE_URL=postgresql+asyncpg://<username>:<password>@<neon-host>/neondb?ssl=require
  ```
* **Redis Cloud**: Update your Redis connection details:
  ```ini
  REDIS_URL=redis://:<redis-password>@<redis-cloud-host>:<redis-cloud-port>
  CELERY_BROKER_URL=redis://:<redis-password>@<redis-cloud-host>:<redis-cloud-port>/0
  CELERY_RESULT_BACKEND=redis://:<redis-password>@<redis-cloud-host>:<redis-cloud-port>/1
  ```
  *(Note: If your Redis Cloud database doesn't support multiple database index partitions, you can remove the `/0` and `/1` suffixes).*

### 3. Install Python Environment & Dependencies
This project uses **`uv`** for lightning-fast environment and package management.

#### A. Install `uv` (if not already installed)
On Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# Remember to restart your terminal or refresh environment variables afterwards.
```
On Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### B. Sync Dependencies
Create the virtual environment and install all libraries (including dev dependencies like `pytest`, `ruff`, etc.):
```powershell
uv sync --dev
```
*This command automatically creates a virtual environment folder named `.venv` if one does not exist, and synchronizes packages to match `uv.lock`.*

#### C. Activate the Virtual Environment (Optional)
On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
On Linux/macOS:
```bash
source .venv/bin/activate
```

#### D. Install Pre-Commit Hooks
Set up git hook checks to auto-format and lint on every commit:
```bash
uv run pre-commit install
```

### 4. Boot Core Services (Local vs Docker)
If using local engines/databases (like Neon and Redis Cloud), ensure they are active.
Otherwise, if running local containers:
On Linux/macOS:
```bash
make up-core
```
On Windows (PowerShell):
```powershell
docker compose up -d postgres redis
```

### 5. Run Database Migrations
Deploy schema configurations using Alembic (runs as a no-op until database models are defined in later phases):
On Linux/macOS:
```bash
make migrate
```
On Windows (PowerShell):
```powershell
uv run alembic upgrade head
```

### 6. Start the Applications

#### A. Start the API Gateway
Run the FastAPI development server:
On Linux/macOS:
```bash
make dev-api
```
On Windows (PowerShell):
```powershell
$env:PYTHONPATH="libs/core;apps/api"
uv run uvicorn llm_autopilot_api.main:app --reload --host 0.0.0.0 --port 8000
```
The API is now running locally at [http://localhost:8000](http://localhost:8000) and documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

#### B. Start Celery worker & beat scheduler
To run background tasks locally:
```powershell
# In terminal 2:
$env:PYTHONPATH="libs/core;apps/api;apps/worker"
uv run celery -A llm_autopilot_worker.main worker -Q verification,retraining -c 2 --loglevel DEBUG

# In terminal 3 (Optional scheduler):
$env:PYTHONPATH="libs/core;apps/api;apps/worker"
uv run celery -A llm_autopilot_worker.main beat --loglevel DEBUG
```

## 🛠️ Development & Tooling Commands

The provided `Makefile` manages the local workflow with 20+ targets:

| Command | Action |
|:---|:---|
| `make install` | Installs production dependencies without development packages. |
| `make dev-install` | Sets up virtual environments using `uv` and installs pre-commit hooks. |
| `make lint` | Runs `ruff` to perform syntax checking, auto-formatting, and fixes. |
| `make format` | Performs auto-formatting using `black`. |
| `make typecheck` | Checks static typings with `mypy`. |
| `make check` | Runs both `lint` and `typecheck` commands. |
| `make test` | Runs the test suite (requires active Postgres & Redis). |
| `make test-unit` | Runs offline-only unit tests (no external services needed). |
| `make test-integration`| Runs integration tests directly. |
| `make test-cov` | Generates a coverage check with an HTML output report. |
| `make up` | Starts all 9 services in Docker Compose (API, worker, dashboards, Ollama, etc.). |
| `make up-core` | Launches Postgres & Redis only. |
| `make up-infra` | Launches Postgres, Redis, Prometheus, and Grafana (no application containers). |
| `make down` | Shuts down Docker Compose containers while maintaining database volume history. |
| `make down-v` | Shuts down containers and destroys volumes (fully resets databases). |
| `make dev-worker` | Runs the Celery background task worker process locally. |
| `make dev-beat` | Runs the Celery beat scheduler process locally (ensure only one scheduler is running). |
| `make dev-flower` | Launches the Celery Flower task-monitoring dashboard. |
| `make ollama-pull` | Pulls recommended LLM models inside the Ollama environment (Llama 3.1 & Mistral). |
| `make reset` | Tear down containers, remove database volumes, and wipe python caches. |

---

## 📊 Monitoring & Services Dashboard URLs

Once the full Docker stack is running (`make up`), you can access these web consoles:

* **FastAPI documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Grafana Dashboards**: [http://localhost:3000](http://localhost:3000) (Credentials: `admin` / `autopilot`)
* **Flower (Celery dashboard)**: [http://localhost:5555](http://localhost:5555)
* **Prometheus Metrics console**: [http://localhost:9090](http://localhost:9090)
* **RedisInsight (Redis dashboard)**: [http://localhost:8001](http://localhost:8001)

---

## ⚙️ Configuration Tuning (`.env`)

Adjust routing heuristics, similarity threshold criteria, and verification parameters via your local `.env`:
* `CACHE_SIMILARITY_THRESHOLD`: Semantic caching proximity lookup (default `0.92`).
* `CLASSIFIER_CONFIDENCE_THRESHOLD`: Auto-escalate threshold when routing classification falls short.
* `VERIFICATION_SAMPLE_RATE_LOW_CONFIDENCE`: Fraction of low-confidence routes audited by default.
* `VERIFICATION_SAMPLE_RATE_HIGH_CONFIDENCE`: Fraction of high-confidence routes sampled for evaluation.
