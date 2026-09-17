[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Question,
    [string]$QueryId = 'grounded-explanation',
    [ValidateSet(
        'explain_rating_drop',
        'explain_negative_review_increase',
        'analyze_low_rating_complaints',
        'explain_delivery_impact_on_rating'
    )]
    [string]$Intent,
    [string]$Metric,
    [string]$ProductCategory,
    [string]$CustomerState,
    [string]$StartMonth,
    [string]$EndMonth,
    [ValidateRange(1, 5)]
    [int]$ThemesLimit = 3,
    [ValidateRange(1, 10)]
    [int]$EvidencePerTheme = 2,
    [switch]$NoTranslate
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
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
        '/workspace/src/ecommerce_rag/rag/run_explanation.py',
        $Question,
        '--environment', 'local',
        '--query-id', $QueryId,
        '--themes-limit', [string]$ThemesLimit,
        '--evidence-per-theme', [string]$EvidencePerTheme
    )
    if ($Intent) { $arguments += @('--intent', $Intent) }
    if ($Metric) { $arguments += @('--metric', $Metric) }
    if ($ProductCategory) { $arguments += @('--product-category', $ProductCategory) }
    if ($CustomerState) { $arguments += @('--customer-state', $CustomerState) }
    if ($StartMonth) { $arguments += @('--start-month', $StartMonth) }
    if ($EndMonth) { $arguments += @('--end-month', $EndMonth) }
    if ($NoTranslate) { $arguments += '--no-translate' }

    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Generazione della spiegazione grounded non riuscita.'
    }
    Write-Host "Risultato disponibile in: $projectRoot\reports\integration\phase5\$QueryId.json" -ForegroundColor Green
}
finally {
    Pop-Location
}
