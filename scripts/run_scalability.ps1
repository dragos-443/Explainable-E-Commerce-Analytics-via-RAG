[CmdletBinding()]
param(
    [ValidateSet('generate', 'benchmark', 'all')]
    [string]$Mode = 'all',
    [int[]]$Factors = @(1, 5, 10, 25, 50, 100),
    [ValidateRange(1, 20)]
    [int]$Repetitions = 3,
    [ValidateRange(1, 5)]
    [int]$WarmupRuns = 1,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$uiWasRunning = $false
Push-Location $projectRoot

try {
    $runningServices = docker compose ps --status running --services
    $required = @('app', 'namenode', 'datanode-1', 'datanode-2', 'spark-master', 'spark-worker-1', 'spark-worker-2')
    $missing = @($required | Where-Object { $_ -notin $runningServices })
    if ($LASTEXITCODE -ne 0 -or $missing.Count -gt 0) {
        throw "Servizi mancanti: $($missing -join ', '). Avviare prima scripts/start.ps1."
    }
    if ($Factors.Count -eq 0 -or @($Factors | Where-Object { $_ -le 0 }).Count -gt 0) {
        throw 'I fattori devono essere interi positivi.'
    }

    docker compose exec -T app bash -lc "pgrep -f '[s]treamlit run' >/dev/null"
    $uiWasRunning = $LASTEXITCODE -eq 0
    if ($uiWasRunning) {
        docker compose exec -T app bash -lc "pkill -TERM -f '[s]treamlit run' || true"
    }

    $arguments = @(
        'compose', 'exec', '-T', 'app', '/opt/spark/bin/spark-submit',
        '--master', 'spark://spark-master:7077',
        '--conf', 'spark.driver.host=app',
        '--conf', 'spark.driver.bindAddress=0.0.0.0',
        '--conf', 'spark.driver.port=39000',
        '--conf', 'spark.blockManager.port=39001',
        '--conf', 'spark.cores.max=4',
        '--conf', 'spark.executor.cores=2',
        '--conf', 'spark.executor.memory=1g',
        '--conf', 'spark.sql.shuffle.partitions=16',
        '/workspace/src/ecommerce_rag/scalability/phase9.py',
        '--environment', 'local',
        '--mode', $Mode,
        '--repetitions', "$Repetitions",
        '--warmup-runs', "$WarmupRuns",
        '--factors'
    )
    $arguments += @($Factors | ForEach-Object { "$_" })
    if ($Force) { $arguments += '--force' }

    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Esperimento di scalabilità non riuscito.'
    }
    Write-Host "Risultati disponibili in: $projectRoot\reports\scalability\phase9" -ForegroundColor Green
}
finally {
    if ($uiWasRunning) {
        & "$PSScriptRoot\start_ui.ps1"
    }
    Pop-Location
}
