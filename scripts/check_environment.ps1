[CmdletBinding()]
param()

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
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker CLI non trovato. Installare Docker Desktop.'
    }

    docker compose version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose v2 non è disponibile.'
    }

    if (-not (Test-DockerEngine)) {
        throw 'Il motore Docker non è in esecuzione. Avviare Docker Desktop.'
    }

    docker compose config --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'compose.yml non è valido.'
    }

    Write-Host 'Ambiente locale: OK' -ForegroundColor Green
}
finally {
    Pop-Location
}
