[CmdletBinding()]
param(
    [string]$SummaryName = 'index_summary'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $projectRoot

try {
    $runningServices = docker compose ps --status running --services
    if ($LASTEXITCODE -ne 0 -or 'app' -notin $runningServices) {
        throw 'Avviare prima l''ambiente con scripts/start.ps1.'
    }
    docker compose exec -T app /opt/spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        --conf spark.driver.host=app `
        --conf spark.driver.bindAddress=0.0.0.0 `
        --conf spark.driver.port=39000 `
        --conf spark.blockManager.port=39001 `
        /workspace/src/ecommerce_rag/rag/index.py `
        --environment local `
        --summary-output "/workspace/reports/rag/phase4/$SummaryName.json"
    if ($LASTEXITCODE -ne 0) {
        throw 'Indicizzazione Chroma non riuscita.'
    }
}
finally {
    Pop-Location
}
