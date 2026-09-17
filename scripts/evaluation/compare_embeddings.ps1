param(
    [ValidateSet("index", "evaluate", "score", "all", "delete")]
    [string]$Mode = "all",
    [ValidateRange(1, 1000)]
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

docker compose exec app python3 -m ecommerce_rag.evaluation.embedding_comparison `
    $Mode `
    --environment local `
    --batch-size $BatchSize

if ($LASTEXITCODE -ne 0) {
    throw "Embedding comparison failed with exit code $LASTEXITCODE"
}
