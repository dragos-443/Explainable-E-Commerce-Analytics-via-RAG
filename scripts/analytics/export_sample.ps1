[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('orders_enriched', 'reviews_enriched', 'review_order_links')]
    [string]$Dataset,

    [ValidateRange(1, 10000)]
    [int]$Rows = 100,

    [int]$Seed = 42
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
        /workspace/src/ecommerce_rag/preprocessing/export_sample.py `
        --dataset $Dataset `
        --rows $Rows `
        --seed $Seed `
        --environment local `
        --output-root /workspace/data/exports
    if ($LASTEXITCODE -ne 0) {
        throw "Esportazione del campione non riuscita: $Dataset"
    }

    $outputPath = Join-Path $projectRoot "data\exports\$Dataset\sample.csv"
    Write-Host "Campione disponibile in: $outputPath" -ForegroundColor Green
}
finally {
    Pop-Location
}
