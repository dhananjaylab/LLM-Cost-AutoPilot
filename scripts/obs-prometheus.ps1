<#
.SYNOPSIS
    Start Prometheus locally, scraping the LLM Cost AutoPilot API.
    PowerShell-safe: avoids the -- unary operator problem.

.USAGE
    .\scripts\obs-prometheus.ps1
    .\scripts\obs-prometheus.ps1 -Port 9091
#>
param(
    [string]$BinPath    = "",
    [string]$ConfigFile = "",
    [string]$DataPath   = "",
    [string]$Port       = "9090",
    [string]$Retention  = "7d"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Defaults resolved after $PSScriptRoot is available
if (-not $BinPath)    { $BinPath    = "$PSScriptRoot\..\.obs\bin\prometheus.exe" }
if (-not $ConfigFile) { $ConfigFile = "$PSScriptRoot\..\infra\prometheus\prometheus.yml" }
if (-not $DataPath)   { $DataPath   = "$PSScriptRoot\..\.obs\prometheus\data" }

# ── Resolve absolute paths (needed when launched from any working dir) ─────────
$BinPath    = [IO.Path]::GetFullPath($BinPath)
$ConfigFile = [IO.Path]::GetFullPath($ConfigFile)
$DataPath   = [IO.Path]::GetFullPath($DataPath)

# ── Fallback: look for prometheus.exe on PATH ──────────────────────────────────
if (-not (Test-Path $BinPath)) {
    $found = Get-Command prometheus.exe -ErrorAction SilentlyContinue
    if ($found) {
        $BinPath = $found.Source
    } else {
        Write-Error @"
prometheus.exe not found.
Put it in .obs\bin\  OR  add it to your PATH.
Download: https://prometheus.io/download/
"@
        exit 1
    }
}

if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config not found: $ConfigFile"
    exit 1
}

# Ensure data directory exists
New-Item -ItemType Directory -Force -Path $DataPath | Out-Null

Write-Host ""
Write-Host "=== Prometheus ===" -ForegroundColor Cyan
Write-Host "  Binary : $BinPath"
Write-Host "  Config : $ConfigFile"
Write-Host "  Data   : $DataPath"
Write-Host "  UI     : http://localhost:$Port"
Write-Host ""

# ── Launch — quote every argument individually so PS doesn't mangle -- ────────
# Note: Prometheus writes logs to stderr; redirect 2>&1 so PS doesn't
#       treat them as errors (they're just INFO lines, not real errors).
& $BinPath `
    "--config.file=$ConfigFile" `
    "--storage.tsdb.path=$DataPath" `
    "--storage.tsdb.retention.time=$Retention" `
    "--web.listen-address=:$Port" `
    "--web.enable-lifecycle" 2>&1
