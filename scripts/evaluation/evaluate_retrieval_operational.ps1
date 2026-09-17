[CmdletBinding()]
param(
    [ValidateSet("collect", "annotate", "score", "all")]
    [string]$Mode = "all",
    [ValidateRange(1, 10)]
    [int]$Repetitions = 1,
    [ValidateSet("auto", "openai", "codex")]
    [string]$AnnotationSource = "openai"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

docker compose exec app python3 -m ecommerce_rag.evaluation.retrieval_operational_benchmark `
    $Mode `
    --environment local `
    --top-k 5 `
    --repetitions $Repetitions `
    --annotation-source $AnnotationSource

if ($LASTEXITCODE -ne 0) {
    throw "Realistic 50-query retrieval evaluation failed with exit code $LASTEXITCODE"
}
