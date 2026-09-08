[CmdletBinding()]
param(
    [ValidateSet(
        'logistics_march_2018',
        'office_furniture_product_issues',
        'overall_complaint_analysis',
        'order_volume_insufficient',
        'bed_bath_table_quality',
        'electronics_defects',
        'small_appliances_wrong_or_missing',
        'office_furniture_service_refund'
    )]
    [string]$CaseId
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $runningServices = docker compose ps --status running --services
    if ($LASTEXITCODE -ne 0 -or 'app' -notin $runningServices) {
        throw 'Avviare prima l''ambiente con scripts/start.ps1.'
    }
    $arguments = @(
        'compose', 'exec', '-T', 'app', '/opt/spark/bin/spark-submit',
        '--master', 'spark://spark-master:7077',
        '--conf', 'spark.driver.host=app',
        '--conf', 'spark.driver.bindAddress=0.0.0.0',
        '--conf', 'spark.driver.port=39000',
        '--conf', 'spark.blockManager.port=39001',
        '/workspace/src/ecommerce_rag/app/run_demos.py',
        '--environment', 'local'
    )
    if ($CaseId) { $arguments += @('--case-id', $CaseId) }

    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Esecuzione dei casi demo non riuscita.'
    }
    Write-Host "Risultati disponibili in: $projectRoot\reports\demo\phase6" -ForegroundColor Green
}
finally {
    Pop-Location
}
