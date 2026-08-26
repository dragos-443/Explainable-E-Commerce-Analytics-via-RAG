[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    docker compose down
    if ($LASTEXITCODE -ne 0) {
        throw 'Arresto dei servizi non riuscito.'
    }
    Write-Host 'Servizi arrestati; i volumi HDFS e Chroma sono stati conservati.' -ForegroundColor Green
}
finally {
    Pop-Location
}
