[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Question,
    [string]$QueryId = 'retrieval-query',
    [ValidateRange(1, 50)]
    [int]$TopK = 5,
    [string]$ProductCategory,
    [string]$CustomerState,
    [string]$StartMonth,
    [string]$EndMonth,
    [ValidateRange(1, 5)]
    [int]$MaxReviewScore,
    [string]$Theme,
    [switch]$Translate
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
        'compose', 'exec', '-T', 'app', 'python3',
        '/workspace/src/ecommerce_rag/rag/run_retrieval.py',
        $Question,
        '--environment', 'local',
        '--query-id', $QueryId,
        '--top-k', [string]$TopK
    )
    if ($ProductCategory) { $arguments += @('--product-category', $ProductCategory) }
    if ($CustomerState) { $arguments += @('--customer-state', $CustomerState) }
    if ($StartMonth) { $arguments += @('--start-month', $StartMonth) }
    if ($EndMonth) { $arguments += @('--end-month', $EndMonth) }
    if ($PSBoundParameters.ContainsKey('MaxReviewScore')) {
        $arguments += @('--max-review-score', [string]$MaxReviewScore)
    }
    if ($Theme) { $arguments += @('--theme', $Theme) }
    if ($Translate) { $arguments += '--translate' }

    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Retrieval delle recensioni non riuscito.'
    }
    Write-Host "Risultato disponibile in: $projectRoot\reports\rag\phase4\$QueryId.json" -ForegroundColor Green
}
finally {
    Pop-Location
}
