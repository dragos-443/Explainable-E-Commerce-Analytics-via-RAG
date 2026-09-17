[CmdletBinding()]
param(
    [string]$QueryId = 'analytics-query',
    [string]$ProductCategory,
    [string]$CustomerState,
    [string]$StartMonth,
    [string]$EndMonth,
    [string]$OutputRoot = '/workspace/reports/analytics/phase3'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $projectRoot

try {
    $runningServices = docker compose ps --status running --services
    if ($LASTEXITCODE -ne 0 -or 'app' -notin $runningServices) {
        throw 'Avviare prima l''ambiente con scripts/start.ps1.'
    }

    $queryArguments = @(
        '--environment', 'local',
        '--query-id', $QueryId,
        '--output', "$OutputRoot/$QueryId.json"
    )
    if ($ProductCategory) { $queryArguments += @('--product-category', $ProductCategory) }
    if ($CustomerState) { $queryArguments += @('--customer-state', $CustomerState) }
    if ($StartMonth) { $queryArguments += @('--start-month', $StartMonth) }
    if ($EndMonth) { $queryArguments += @('--end-month', $EndMonth) }

    $sparkArguments = @(
        'compose', 'exec', '-T', 'app',
        '/opt/spark/bin/spark-submit',
        '--master', 'spark://spark-master:7077',
        '--conf', 'spark.driver.host=app',
        '--conf', 'spark.driver.bindAddress=0.0.0.0',
        '--conf', 'spark.driver.port=39000',
        '--conf', 'spark.blockManager.port=39001',
        '/workspace/src/ecommerce_rag/analytics/run_query.py'
    ) + $queryArguments

    & docker @sparkArguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Query analytics della Fase 3 non riuscita.'
    }

    Write-Host "Risultato disponibile in: $projectRoot\reports\analytics\phase3\$QueryId.json" -ForegroundColor Green
}
finally {
    Pop-Location
}
