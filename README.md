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

---

## ⚡ Quick Start

Follow these steps to unpack the repository, configure variables, install dependencies, and run services locally:

### 1. Extract and Navigate
```bash
tar -xzf llm-cost-autopilot-phase0.tar.gz
cd llm-cost-autopilot
```

### 2. Configure Environments
Create a local configuration file from the template and enter your API credentials:
```bash
cp .env.example .env
# Open .env and add your respective OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### 3. Install Dependencies
This project uses `uv` for package management and setting up pre-commit hooks:
```bash
make dev-install
```

### 4. Boot Core Services
Start Postgres and Redis in the background:
```bash
make up-core
```

### 5. Run Database Migrations
Deploy schema configurations using Alembic (runs as a no-op until database models are defined in later phases):
```bash
make migrate
```

### 6. Start the API Gateway
Run the FastAPI development server:
```bash
make dev-api
```
The API is now running locally at [http://localhost:8000](http://localhost:8000) and documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

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
