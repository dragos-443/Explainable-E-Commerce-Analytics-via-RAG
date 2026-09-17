$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $projectRoot

try {
    docker compose exec -T app python3 -m ecommerce_rag.evaluation.reporting
    if ($LASTEXITCODE -ne 0) {
        throw 'Creazione del riepilogo di valutazione non riuscita.'
    }
}
finally {
    Pop-Location
}
