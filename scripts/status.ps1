[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

function Get-PublishedPort([string]$Service, [int]$ContainerPort) {
    $mapping = docker compose port $Service $ContainerPort | Select-Object -First 1
    if (-not $mapping) {
        throw "Porta non pubblicata per ${Service}:${ContainerPort}."
    }
    return [int](($mapping -split ':')[-1])
}

try {
    docker compose ps
    if ($LASTEXITCODE -ne 0) {
        throw 'Impossibile leggere lo stato dei servizi.'
    }

    Write-Host "`nHDFS"
    docker compose exec -T namenode hdfs dfsadmin -report

    Write-Host "`nSpark"
    try {
        $sparkPort = Get-PublishedPort 'spark-master' 8080
        $spark = Invoke-RestMethod -Uri "http://localhost:$sparkPort/json/" -TimeoutSec 5
        Write-Host "Master: $($spark.url) | worker attivi: $($spark.aliveworkers)"
    }
    catch {
        Write-Warning 'UI/API del Master Spark non ancora raggiungibile.'
    }

    Write-Host "`nChroma"
    try {
        $chromaPort = Get-PublishedPort 'chroma' 8000
        $heartbeat = Invoke-RestMethod -Uri "http://localhost:$chromaPort/api/v1/heartbeat" -TimeoutSec 5
        Write-Host "Heartbeat: $($heartbeat.'nanosecond heartbeat')"
    }
    catch {
        Write-Warning 'Heartbeat Chroma non ancora raggiungibile.'
    }
}
finally {
    Pop-Location
}
