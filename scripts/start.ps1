[CmdletBinding()]
param(
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

function Test-DockerEngine {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        docker info *> $null
        return $LASTEXITCODE -eq 0
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

try {
    if (-not (Test-Path -LiteralPath '.env')) {
        Copy-Item -LiteralPath '.env.example' -Destination '.env'
        Write-Host 'Creato .env locale da .env.example.'
    }

    if (-not (Test-DockerEngine)) {
        $dockerDesktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (-not (Test-Path -LiteralPath $dockerDesktop)) {
            throw 'Docker Desktop non è in esecuzione e non è stato trovato.'
        }

        Write-Host 'Avvio Docker Desktop...'
        Start-Process -FilePath $dockerDesktop
        $ready = $false
        for ($attempt = 0; $attempt -lt 60; $attempt++) {
            Start-Sleep -Seconds 2
            if (Test-DockerEngine) {
                $ready = $true
                break
            }
        }
        if (-not $ready) {
            throw 'Docker Desktop non è diventato disponibile entro 120 secondi.'
        }
    }

    & "$PSScriptRoot\check_environment.ps1"

    $arguments = @('compose', 'up', '-d', '--wait', '--wait-timeout', '240')
    if (-not $NoBuild) {
        $arguments += '--build'
    }
    & docker $arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Avvio dei servizi Docker non riuscito.'
    }

    docker compose exec -T namenode hdfs dfs -mkdir -p /data/raw /data/processed /data/curated /data/scaled /data/outputs
    if ($LASTEXITCODE -ne 0) {
        throw 'Creazione della struttura HDFS non riuscita.'
    }

    $llmProvider = docker compose exec -T app python3 -c "from ecommerce_rag.common.config import load_config; print(load_config()['llm']['provider'])"
    if ($LASTEXITCODE -ne 0) {
        throw 'Lettura della configurazione LLM non riuscita.'
    }
    if ($llmProvider.Trim() -eq 'ollama') {
        $llmModel = docker compose exec -T app python3 -c "from ecommerce_rag.common.config import load_config; print(load_config()['llm']['model'])"
        if ($LASTEXITCODE -ne 0 -or -not $llmModel.Trim()) {
            throw 'Lettura del modello Ollama non riuscita.'
        }
        Write-Host "Verifica del modello Ollama $($llmModel.Trim())..."
        docker compose exec -T ollama ollama pull $llmModel.Trim()
        if ($LASTEXITCODE -ne 0) {
            throw 'Download o verifica del modello Ollama non riusciti.'
        }
    }

    & "$PSScriptRoot\status.ps1"
    Write-Host 'Ambiente pronto. I volumi persistono anche dopo stop.ps1.' -ForegroundColor Green
}
finally {
    Pop-Location
}
