<#
.SYNOPSIS
    Start Grafana Tempo locally.
    Fixes the PowerShell -- flag problem AND the single-vs-double dash Tempo issue.

.USAGE
    .\scripts\obs-tempo.ps1
#>
param(
    [string]$BinPath    = "",
    [string]$ConfigFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Defaults resolved after $PSScriptRoot is available
if (-not $BinPath)    { $BinPath    = "$PSScriptRoot\..\.obs\bin\tempo.exe" }
if (-not $ConfigFile) { $ConfigFile = "$PSScriptRoot\..\infra\tempo\tempo.yml" }

# ── Resolve absolute paths ─────────────────────────────────────────────────────
$BinPath    = [IO.Path]::GetFullPath($BinPath)
$ConfigFile = [IO.Path]::GetFullPath($ConfigFile)

# ── Fallback: look for tempo.exe on PATH ───────────────────────────────────────
if (-not (Test-Path $BinPath)) {
    $found = Get-Command tempo.exe -ErrorAction SilentlyContinue
    if ($found) {
        $BinPath = $found.Source
    } else {
        Write-Error @"
tempo.exe not found.
Put it in .obs\bin\  OR  add it to your PATH.
Download: https://github.com/grafana/tempo/releases
"@
        exit 1
    }
}

if (-not (Test-Path $ConfigFile)) {
    Write-Error "Config not found: $ConfigFile"
    exit 1
}

# Ensure storage paths from tempo.yml exist
$walPath    = "C:\Temp\tempo\wal"
$blocksPath = "C:\Temp\tempo\blocks"
$genPath    = "C:\Temp\tempo\generator\wal"
foreach ($p in @($walPath, $blocksPath, $genPath)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
}

Write-Host ""
Write-Host "=== Grafana Tempo ===" -ForegroundColor Magenta
Write-Host "  Binary     : $BinPath"
Write-Host "  Config     : $ConfigFile"
Write-Host "  OTLP HTTP  : http://localhost:4318  (app sends traces here)"
Write-Host "  OTLP gRPC  : localhost:4317"
Write-Host "  Query API  : http://localhost:3200  (base URL; root may return 404)"
Write-Host "  Health     : http://localhost:3200/metrics  or  http://localhost:3200/ready"
Write-Host ""

# Tempo 2.x uses single-dash -config.file (not --config.file).
# Quoting the argument is the fix — PowerShell passes it verbatim to the process.
# 2>&1 merges stderr into stdout so PS doesn't flag log lines as errors.
Write-Host "  Using flag : -config.file=$ConfigFile" -ForegroundColor DarkGray
Write-Host ""

& $BinPath "-config.file=$ConfigFile" 2>&1
