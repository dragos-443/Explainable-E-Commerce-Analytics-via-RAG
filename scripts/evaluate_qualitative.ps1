[CmdletBinding()]
param(
    [ValidateSet('reference', 'latest')]
    [string]$InputSource = 'reference'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    docker compose exec -T app python3 -m ecommerce_rag.evaluation.qualitative `
        --input-source $InputSource
    if ($LASTEXITCODE -ne 0) {
        throw 'Valutazione qualitativa non riuscita.'
    }
}
finally {
    Pop-Location
}
