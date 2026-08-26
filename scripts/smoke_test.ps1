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
    $report = docker compose exec -T namenode hdfs dfsadmin -report
    if ($LASTEXITCODE -ne 0 -or ($report -join "`n") -notmatch 'Live datanodes \(2\)') {
        throw 'Lo smoke test richiede esattamente due DataNode HDFS attivi.'
    }

    $replication = docker compose exec -T namenode hdfs getconf -confKey dfs.replication
    if ($LASTEXITCODE -ne 0 -or ($replication -join '').Trim() -ne '1') {
        throw "Replication factor HDFS inatteso: $replication"
    }

    $sparkPort = Get-PublishedPort 'spark-master' 8080
    $spark = Invoke-RestMethod -Uri "http://localhost:$sparkPort/json/" -TimeoutSec 10
    if ([int]$spark.aliveworkers -ne 2) {
        throw "Worker Spark attivi: $($spark.aliveworkers), attesi: 2."
    }

    $rawUri = (docker compose exec -T app python3 -m ecommerce_rag.common.resolve_config --get storage.raw_uri | Out-String).Trim()
    $outputsUri = (docker compose exec -T app python3 -m ecommerce_rag.common.resolve_config --get storage.outputs_uri | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $rawUri -or -not $outputsUri) {
        throw 'Impossibile risolvere gli URI dal profilo local.yml.'
    }
    $inputUri = "$rawUri/smoke"
    $outputUri = "$outputsUri/smoke/reviews_parquet"

    docker compose exec -T namenode hdfs dfs -rm -r -f /data/raw/smoke
    docker compose exec -T namenode hdfs dfs -mkdir -p /data/raw/smoke /data/outputs
    docker compose cp tests/integration/fixtures/smoke_reviews.txt namenode:/tmp/smoke_reviews.txt
    for ($index = 0; $index -lt 8; $index++) {
        docker compose exec -T namenode hdfs dfs -put -f /tmp/smoke_reviews.txt "/data/raw/smoke/reviews-$index.txt"
        if ($LASTEXITCODE -ne 0) {
            throw 'Caricamento dei file di prova su HDFS non riuscito.'
        }
    }

    docker compose exec -T app /opt/spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        --conf spark.driver.host=app `
        --conf spark.driver.bindAddress=0.0.0.0 `
        --conf spark.driver.port=39000 `
        --conf spark.blockManager.port=39001 `
        /workspace/src/ecommerce_rag/smoke/spark_hdfs.py `
        --input-uri $inputUri `
        --output-uri $outputUri
    if ($LASTEXITCODE -ne 0) {
        throw 'Smoke test Spark/HDFS non riuscito.'
    }

    docker compose exec -T namenode hdfs dfs -ls /data/outputs/smoke/reviews_parquet
    $fsck = docker compose exec -T namenode hdfs fsck /data -files -blocks -locations
    $fsck | Out-Host
    if ($LASTEXITCODE -ne 0 -or ($fsck -join "`n") -notmatch 'Average block replication:\s*1\.0') {
        throw 'Verifica dei blocchi HDFS con replication factor 1 non riuscita.'
    }

    $distributionReport = docker compose exec -T namenode hdfs dfsadmin -report
    $distributionText = $distributionReport -join "`n"
    $matches = [regex]::Matches(
        $distributionText,
        'Hostname:\s+([^\r\n]+).*?Num of Blocks:\s+(\d+)',
        [System.Text.RegularExpressions.RegexOptions]::Singleline
    )
    if ($matches.Count -ne 2 -or ($matches | Where-Object { [int]$_.Groups[2].Value -eq 0 })) {
        throw 'I blocchi di prova non risultano distribuiti su entrambi i DataNode.'
    }

    docker compose exec -T app python3 /workspace/src/ecommerce_rag/smoke/chroma_smoke.py
    if ($LASTEXITCODE -ne 0) {
        throw 'Smoke test Chroma non riuscito.'
    }

    Write-Host 'Smoke test completo: HDFS -> Spark -> Parquet e Chroma sono operativi.' -ForegroundColor Green
}
finally {
    Pop-Location
}
