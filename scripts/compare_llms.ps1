[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$uiWasRunning = $false
Push-Location $projectRoot

try {
    $runningServices = docker compose ps --status running --services
    if ($LASTEXITCODE -ne 0 -or 'app' -notin $runningServices -or 'ollama' -notin $runningServices) {
        throw 'Avviare prima l''ambiente con scripts/start.ps1.'
    }
    docker compose exec -T app bash -lc 'test -n "$OPENAI_API_KEY"'
    if ($LASTEXITCODE -ne 0) {
        throw 'OPENAI_API_KEY non è disponibile nel container app.'
    }
    docker compose exec -T app bash -lc "pgrep -f '[s]treamlit run' >/dev/null"
    $uiWasRunning = $LASTEXITCODE -eq 0
    if ($uiWasRunning) {
        docker compose exec -T app bash -lc "pkill -TERM -f '[s]treamlit run' || true"
    }
    & docker compose exec -T app /opt/spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        --conf spark.driver.host=app `
        --conf spark.driver.bindAddress=0.0.0.0 `
        --conf spark.driver.port=39000 `
        --conf spark.blockManager.port=39001 `
        /workspace/src/ecommerce_rag/evaluation/llm_comparison.py `
        --environment local
    if ($LASTEXITCODE -ne 0) {
        throw 'Confronto LLM non riuscito.'
    }
    Write-Host "Risultati disponibili in: $projectRoot\reports\evaluation\phase7-final-llm-comparison" -ForegroundColor Green
}
finally {
    if ($uiWasRunning) {
        & "$PSScriptRoot\start_ui.ps1"
    }
    Pop-Location
}
