<#
.SYNOPSIS
    Start Grafana 13.x locally on Windows.
    Grafana 13 changed: grafana-server.exe -> grafana.exe with `server` subcommand.

.USAGE
    .\scripts\obs-grafana.ps1
    .\scripts\obs-grafana.ps1 -GrafanaHome "D:\grafana"
#>
param(
    # Where Grafana is installed (MSI default for 13.x)
    [string]$GrafanaHome = "C:\Program Files\GrafanaLabs\grafana",
    [string]$Port        = "3000"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -- Locate grafana.exe (Grafana 13+ uses grafana.exe, not grafana-server.exe) --
$serverExe = Join-Path $GrafanaHome "bin\grafana.exe"

if (-not (Test-Path $serverExe)) {
    # Fallback: try old name grafana-server.exe (Grafana <= 10)
    $legacy = Join-Path $GrafanaHome "bin\grafana-server.exe"
    if (Test-Path $legacy) {
        $serverExe = $legacy
    } else {
        # Try PATH for either name
        foreach ($name in @("grafana.exe", "grafana-server.exe")) {
            $found = Get-Command $name -ErrorAction SilentlyContinue
            if ($found) {
                $serverExe   = $found.Source
                $GrafanaHome = Split-Path (Split-Path $serverExe -Parent) -Parent
                break
            }
        }
        if (-not (Test-Path $serverExe)) {
            Write-Error @"
grafana.exe not found at: $serverExe
Install Grafana from: https://grafana.com/grafana/download?platform=windows
Or set -GrafanaHome to your install path.
"@
            exit 1
        }
    }
}

# -- Check if port is already in use -------------------------------------------
$portActive = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($portActive) {
    $pidUsingPort = $portActive.OwningProcess[0]
    Write-Host "WARNING: Port $Port is already in use by process ID $pidUsingPort." -ForegroundColor Yellow
    Write-Host "If this is the Grafana Windows service, you can stop it from an Administrator PowerShell prompt using:" -ForegroundColor Yellow
    Write-Host "  Stop-Service -Name Grafana -Force" -ForegroundColor Cyan
    Write-Error "Cannot start Grafana: Port $Port is in use."
    exit 1
}

# -- Build absolute workspace paths -------------------------------------------
$projectRoot            = [IO.Path]::GetFullPath("$PSScriptRoot\..")
$sourceProvisioningDir  = Join-Path $projectRoot "infra\grafana\provisioning"
$sourceDatasourcesDir   = Join-Path $sourceProvisioningDir "datasources"
$dashboardsDir          = Join-Path $projectRoot "infra\grafana\dashboards"
$homeDashboard          = Join-Path $dashboardsDir "cost_overview.json"
$runtimeRoot            = Join-Path ([IO.Path]::GetTempPath()) "llm-cost-autopilot-grafana"
$patchedRoot            = Join-Path $runtimeRoot "grafana"
$patchedProvisioningDir = Join-Path $patchedRoot "provisioning"
$patchedDashboardsDir   = Join-Path $patchedProvisioningDir "dashboards"
$patchedDatasourcesDir  = Join-Path $patchedProvisioningDir "datasources"
$patchedPluginsDir      = Join-Path $patchedProvisioningDir "plugins"
$patchedAlertingDir     = Join-Path $patchedProvisioningDir "alerting"
$patchedIni             = Join-Path $patchedRoot "grafana-local.ini"
$dataDir                = Join-Path $patchedRoot "data"
$logsDir                = Join-Path $patchedRoot "logs"

foreach ($dir in @(
    $patchedRoot,
    $patchedProvisioningDir,
    $patchedDashboardsDir,
    $patchedDatasourcesDir,
    $patchedPluginsDir,
    $patchedAlertingDir,
    $dataDir,
    $logsDir,
    $dashboardsDir
)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if (Test-Path $sourceDatasourcesDir) {
    Copy-Item -Path (Join-Path $sourceDatasourcesDir "*") -Destination $patchedDatasourcesDir -Recurse -Force
}

$dashboardProvisioningContent = @"
apiVersion: 1

providers:
  - name: LLM Cost Autopilot
    orgId: 1
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    allowUiUpdates: true
    options:
      path: $dashboardsDir
      foldersFromFilesStructure: true
"@
Set-Content -Path (Join-Path $patchedDashboardsDir "dashboards.yml") -Value $dashboardProvisioningContent -Encoding UTF8

$iniContent = @"
[server]
http_port = $Port
domain = localhost

[security]
admin_user = admin
admin_password = autopilot
disable_gravatar = true

[users]
allow_sign_up = false

[analytics]
reporting_enabled = false
check_for_updates = false

[paths]
provisioning = $patchedProvisioningDir
data = $dataDir
logs = $logsDir

[dashboards]
default_home_dashboard_path = $homeDashboard

[log]
mode = console
level = info

[feature_toggles]
serviceGraph = true
nodeGraph = true
traceToMetrics = true
tempoSearch = true
tempoBackendSearch = true
tempoServiceGraph = true
"@
Set-Content -Path $patchedIni -Value $iniContent -Encoding UTF8

Write-Host ""
Write-Host "=== Grafana ===" -ForegroundColor Yellow
Write-Host "  Binary       : $serverExe"
Write-Host "  Home         : $GrafanaHome"
Write-Host "  Provisioning : $patchedProvisioningDir"
Write-Host "  Dashboards   : $dashboardsDir"
Write-Host "  Patched INI  : $patchedIni"
Write-Host "  UI           : http://localhost:$Port  (admin / autopilot)"
Write-Host ""

# -- Launch -- Grafana 13 syntax: grafana server --config=... --homepath=... --
# Note: Grafana writes logs to stderr. We temporarily set ErrorActionPreference
#       to Continue so PowerShell doesn't treat stderr logs as terminating errors.
$OrgAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $serverExe server "--config=$patchedIni" "--homepath=$GrafanaHome"
} finally {
    $ErrorActionPreference = $OrgAction
}
