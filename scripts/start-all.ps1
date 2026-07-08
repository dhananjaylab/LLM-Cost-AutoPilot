# Start all LLM Cost AutoPilot services in separate Windows PowerShell windows.

$services = @(
    @{ Name = "API"; Command = "uvicorn llm_autopilot_api.main:app --reload --port 8000" },
    @{ Name = "Celery Worker"; Command = "celery -A llm_autopilot_worker.main worker -Q verification,retraining -P solo --loglevel DEBUG -E" },
    @{ Name = "Celery Beat"; Command = "celery -A llm_autopilot_worker.main beat --loglevel DEBUG" },
    @{ Name = "Celery Flower"; Command = "celery -A llm_autopilot_worker.main flower --port=5555" },
    @{ Name = "Tempo"; Command = "powershell -ExecutionPolicy Bypass -File scripts\obs-tempo.ps1" },
    @{ Name = "Prometheus"; Command = "powershell -ExecutionPolicy Bypass -File scripts\obs-prometheus.ps1" },
    @{ Name = "Grafana"; Command = "powershell -ExecutionPolicy Bypass -File scripts\obs-grafana.ps1" }
)

foreach ($svc in $services) {
    Write-Host "Starting $($svc.Name)..." -ForegroundColor Cyan
    # Start in a new powershell window with a custom title
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = '$($svc.Name)'; $($svc.Command)"
}
