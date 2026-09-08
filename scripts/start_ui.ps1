[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $mapping = docker compose port app 8501 | Select-Object -First 1
    if (-not $mapping) {
        throw 'Porta Streamlit non pubblicata. Esegui prima scripts/start.ps1 per ricreare il container.'
    }
    $publishedPort = [int](($mapping -split ':')[-1])
    $healthUri = "http://localhost:$publishedPort/_stcore/health"
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri $healthUri -TimeoutSec 3
        if ($health.StatusCode -eq 200) {
            Write-Host "Streamlit è già disponibile su http://localhost:$publishedPort" -ForegroundColor Green
            return
        }
    }
    catch {
        # Il server non è ancora attivo: viene avviato nel container app.
    }

    docker compose exec -d app streamlit run `
        /workspace/src/ecommerce_rag/app/streamlit_app.py `
        --server.address 0.0.0.0 `
        --server.port 8501 `
        --server.headless true `
        --browser.gatherUsageStats false
    if ($LASTEXITCODE -ne 0) {
        throw 'Avvio di Streamlit non riuscito.'
    }

    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $health = Invoke-WebRequest -UseBasicParsing -Uri $healthUri -TimeoutSec 3
            if ($health.StatusCode -eq 200) {
                Write-Host "Streamlit disponibile su http://localhost:$publishedPort" -ForegroundColor Green
                return
            }
        }
        catch {
            # Attende il prossimo tentativo.
        }
    }
    throw 'Streamlit non è diventato disponibile entro 30 secondi.'
}
finally {
    Pop-Location
}
