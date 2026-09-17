[CmdletBinding()]
param(
    [ValidateSet("collect", "annotate", "score", "all")]
    [string]$Mode = "all",
    [ValidateRange(1, 10)]
    [int]$Repetitions = 3
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

docker compose exec app python3 -m ecommerce_rag.evaluation.service_refund_improvement `
    $Mode `
    --environment local `
    --top-k 5 `
    --repetitions $Repetitions

if ($LASTEXITCODE -ne 0) {
    throw "Service/refund retrieval experiment failed with exit code $LASTEXITCODE"
}
