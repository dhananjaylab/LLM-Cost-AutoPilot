<#
.SYNOPSIS
    Launch the full observability stack (Prometheus + Tempo + Grafana)
    each in its own PowerShell window.

.USAGE
    .\scripts\obs-start.ps1
    .\scripts\obs-start.ps1 -DryRun   # just print what would run
#>
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest

$root    = [IO.Path]::GetFullPath("$PSScriptRoot\..")
$scripts = $PSScriptRoot

function Start-Obs {
    param([string]$Title, [string]$Script, [string]$Color = "White")
    $cmd = "Set-Location '$root'; Write-Host '=== $Title ===' -ForegroundColor $Color; & '$Script'; Read-Host 'Press Enter to close'"
    if ($DryRun) {
        Write-Host "[DRY RUN] Would open window: $Title → $Script" -ForegroundColor DarkGray
    } else {
        Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $cmd -WindowStyle Normal
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     LLM Cost AutoPilot — Observability Launcher          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host " Opening 3 windows — wait ~5 seconds for each to start." -ForegroundColor Gray
Write-Host ""

# Window 1 — Tempo (start first so Prometheus can scrape it)
Start-Obs -Title "Grafana Tempo  [:4318 OTLP | :3200 API]" `
          -Script "$scripts\obs-tempo.ps1" `
          -Color "Magenta"

Start-Sleep -Seconds 1

# Window 2 — Prometheus
Start-Obs -Title "Prometheus  [:9090]" `
          -Script "$scripts\obs-prometheus.ps1" `
          -Color "DarkYellow"

Start-Sleep -Seconds 1

# Window 3 — Grafana
Start-Obs -Title "Grafana  [:3000]" `
          -Script "$scripts\obs-grafana.ps1" `
          -Color "Yellow"

Write-Host ""
Write-Host " Services starting in separate windows:" -ForegroundColor Green
Write-Host ""
Write-Host "   Tempo      → OTLP HTTP : http://localhost:4318" -ForegroundColor Magenta
Write-Host "                Query API : http://localhost:3200  (root may return 404)"
Write-Host "                Health    : http://localhost:3200/metrics  or  /ready"
Write-Host ""
Write-Host "   Prometheus → UI        : http://localhost:9090" -ForegroundColor DarkYellow
Write-Host ""
Write-Host "   Grafana    → UI        : http://localhost:3000  (admin / autopilot)" -ForegroundColor Yellow
Write-Host ""
Write-Host " Then start the API:" -ForegroundColor Green
Write-Host "   uv run uvicorn llm_autopilot_api.main:app --reload --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host " Metrics  → http://localhost:8000/metrics"
Write-Host " API docs → http://localhost:8000/docs"
Write-Host ""
