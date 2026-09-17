[CmdletBinding()]
param(
    [ValidateSet("collect", "annotate", "score", "all")]
    [string]$Mode = "all",
    [ValidateRange(1, 10)]
    [int]$Repetitions = 3,
    [ValidateSet("auto", "openai", "codex")]
    [string]$AnnotationSource = "auto"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $projectRoot

docker compose exec app python3 -m ecommerce_rag.evaluation.retrieval_final_holdout `
    $Mode `
    --environment local `
    --top-k 5 `
    --repetitions $Repetitions `
    --annotation-source $AnnotationSource

if ($LASTEXITCODE -ne 0) {
    throw "Final 100-query retrieval evaluation failed with exit code $LASTEXITCODE"
}
