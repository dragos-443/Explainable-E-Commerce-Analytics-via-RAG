[CmdletBinding()]
param(
    [ValidateSet('collect', 'score')]
    [string]$Mode = 'score',
    [ValidateRange(1, 20)]
    [int]$TopK = 5
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    docker compose exec -T app python3 -m ecommerce_rag.evaluation.retrieval `
        $Mode --environment local --top-k $TopK
    if ($LASTEXITCODE -ne 0) {
        throw 'Valutazione del retrieval non riuscita.'
    }
}
finally {
    Pop-Location
}
