[CmdletBinding()]
param(
    [string]$SourceDirectory = 'data/raw/olist'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $sourcePath = (Resolve-Path -LiteralPath $SourceDirectory).Path
    $expectedFiles = @(
        'olist_customers_dataset.csv',
        'olist_geolocation_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_order_payments_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_orders_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv',
        'product_category_name_translation.csv'
    )

    $runningServices = docker compose ps --status running --services
    if ($LASTEXITCODE -ne 0 -or 'app' -notin $runningServices -or 'namenode' -notin $runningServices) {
        throw 'Avviare prima l''ambiente con scripts/start.ps1.'
    }

    $rawUri = (docker compose exec -T app python3 -m ecommerce_rag.common.resolve_config --get storage.raw_uri | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $rawUri) {
        throw 'Impossibile risolvere storage.raw_uri dal profilo locale.'
    }
    $destinationDirectory = "$rawUri/olist"
    docker compose exec -T namenode hdfs dfs -mkdir -p $destinationDirectory
    if ($LASTEXITCODE -ne 0) {
        throw 'Impossibile creare la directory raw Olist in HDFS.'
    }

    foreach ($fileName in $expectedFiles) {
        $localFile = Join-Path $sourcePath $fileName
        if (-not (Test-Path -LiteralPath $localFile -PathType Leaf)) {
            throw "File richiesto non trovato: $localFile"
        }

        $localHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $localFile).Hash.ToLowerInvariant()
        $destination = "$destinationDirectory/$fileName"

        docker compose exec -T namenode hdfs dfs -test -e $destination
        $exists = $LASTEXITCODE -eq 0
        if ($exists) {
            $remoteHashOutput = docker compose exec -T namenode sh -c "hdfs dfs -cat '$destination' | sha256sum"
            if ($LASTEXITCODE -ne 0) {
                throw "Impossibile verificare il file HDFS esistente: $destination"
            }
            $remoteHash = (($remoteHashOutput | Out-String).Trim() -split '\s+')[0].ToLowerInvariant()
            if ($remoteHash -ne $localHash) {
                throw "Il file raw HDFS differisce dal file locale e non verra sovrascritto: $fileName"
            }
            Write-Host "Gia presente e verificato: $fileName ($localHash)"
            continue
        }

        $temporaryPath = "/tmp/$fileName"
        docker compose cp $localFile "namenode:$temporaryPath"
        if ($LASTEXITCODE -ne 0) {
            throw "Copia nel container non riuscita: $fileName"
        }
        docker compose exec -T namenode hdfs dfs -put $temporaryPath $destination
        if ($LASTEXITCODE -ne 0) {
            throw "Caricamento HDFS non riuscito: $fileName"
        }

        $uploadedHashOutput = docker compose exec -T namenode sh -c "hdfs dfs -cat '$destination' | sha256sum"
        $uploadedHash = (($uploadedHashOutput | Out-String).Trim() -split '\s+')[0].ToLowerInvariant()
        if ($LASTEXITCODE -ne 0 -or $uploadedHash -ne $localHash) {
            throw "Verifica SHA-256 dopo il caricamento non riuscita: $fileName"
        }
        Write-Host "Caricato e verificato: $fileName ($localHash)"
    }

    Write-Host 'Dataset raw Olist disponibile in HDFS e verificato senza modifiche.' -ForegroundColor Green
}
finally {
    Pop-Location
}
