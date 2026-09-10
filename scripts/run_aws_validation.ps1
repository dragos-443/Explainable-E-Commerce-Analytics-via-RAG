[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Launch', 'Wait', 'Collect', 'All')]
    [string]$Mode = 'All',
    [string]$Profile = 'ecommerce-rag-lab',
    [string]$Region = 'us-east-1',
    [string]$Bucket = 'ecommerce-rag-analytics-5138901',
    [string]$InstanceType = 'm5.xlarge',
    [int[]]$Factors = @(1, 10, 100)
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportRoot = Join-Path $projectRoot 'reports\aws\phase10'
$statePath = Join-Path $reportRoot 'run_state.json'

function Invoke-Aws {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $output = & aws @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI command failed: aws $($Arguments -join ' ')"
    }
    return $output
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Content
    )
    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function Assert-Prerequisites {
    if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
        throw 'AWS CLI non trovato.'
    }
    Invoke-Aws sts get-caller-identity --profile $Profile --query Account --output text | Out-Null
    $configuredRegion = & aws configure get region --profile $Profile
    if ($configuredRegion -ne $Region) {
        throw "Il profilo $Profile usa la regione '$configuredRegion', attesa '$Region'."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'data\raw\olist'))) {
        throw 'Dataset raw Olist locale non trovato.'
    }
}

function New-SourceArchive {
    param([string]$Destination)
    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("phase10-" + [guid]::NewGuid().ToString('N'))
    $stage = Join-Path $temporaryRoot 'stage'
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    try {
        Copy-Item -LiteralPath (Join-Path $projectRoot 'src\ecommerce_rag') -Destination $stage -Recurse
        Get-ChildItem -LiteralPath $stage -Directory -Recurse -Filter '__pycache__' |
            Remove-Item -Recurse -Force
        Compress-Archive -Path (Join-Path $stage 'ecommerce_rag') -DestinationPath $Destination -Force
    }
    finally {
        $resolvedTemporaryRoot = [System.IO.Path]::GetFullPath($temporaryRoot)
        $resolvedSystemTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if ($resolvedTemporaryRoot.StartsWith($resolvedSystemTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force
        }
    }
}

function Prepare-Phase10 {
    New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
    $bucketNames = @(Invoke-Aws s3api list-buckets --profile $Profile --query 'Buckets[].Name' --output text) -join "`t"
    if (($bucketNames -split '\s+') -notcontains $Bucket) {
        Write-Host "Creo il bucket dedicato $Bucket..." -ForegroundColor Cyan
        Invoke-Aws s3api create-bucket --bucket $Bucket --region $Region --profile $Profile | Out-Null
    }
    Invoke-Aws s3api put-public-access-block --bucket $Bucket --profile $Profile `
        --public-access-block-configuration `
        'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true' | Out-Null
    $encryptionPath = Join-Path ([System.IO.Path]::GetTempPath()) ("phase10-encryption-" + [guid]::NewGuid().ToString('N') + '.json')
    try {
        $encryptionJson = @{
            Rules = @(
                @{
                    ApplyServerSideEncryptionByDefault = @{ SSEAlgorithm = 'AES256' }
                    BucketKeyEnabled = $false
                }
            )
        } | ConvertTo-Json -Depth 6
        Write-Utf8NoBom -Path $encryptionPath -Content $encryptionJson
        Invoke-Aws s3api put-bucket-encryption --bucket $Bucket --profile $Profile `
            --server-side-encryption-configuration "file://$encryptionPath" | Out-Null
    }
    finally {
        if (Test-Path -LiteralPath $encryptionPath) {
            Remove-Item -LiteralPath $encryptionPath -Force
        }
    }

    Write-Host 'Genero il manifest locale di riferimento...' -ForegroundColor Cyan
    & docker compose exec -T app /opt/spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        --conf spark.driver.host=app `
        --conf spark.driver.bindAddress=0.0.0.0 `
        --conf spark.driver.port=39000 `
        --conf spark.blockManager.port=39001 `
        /workspace/src/ecommerce_rag/cloud/validation.py manifest `
        --environment local `
        --output /workspace/reports/aws/phase10/local_manifest.json
    if ($LASTEXITCODE -ne 0) {
        throw 'Creazione del manifest locale non riuscita.'
    }

    $archive = Join-Path $reportRoot 'ecommerce_rag.zip'
    New-SourceArchive -Destination $archive
    Write-Host 'Carico codice, configurazione e dataset raw su S3...' -ForegroundColor Cyan
    Invoke-Aws s3 cp $archive "s3://$Bucket/artifacts/ecommerce_rag.zip" --profile $Profile --only-show-errors | Out-Null
    $artifacts = @{
        'config\environments\aws.yml' = 'aws.yml'
        'src\ecommerce_rag\preprocessing\pipeline.py' = 'pipeline.py'
        'src\ecommerce_rag\rag\prepare.py' = 'prepare.py'
        'src\ecommerce_rag\scalability\phase9.py' = 'phase9.py'
        'src\ecommerce_rag\cloud\validation.py' = 'cloud_validation.py'
        'scripts\aws\phase10_step.sh' = 'phase10_step.sh'
        'scripts\aws\bootstrap_phase10.sh' = 'bootstrap_phase10.sh'
    }
    foreach ($entry in $artifacts.GetEnumerator()) {
        Invoke-Aws s3 cp (Join-Path $projectRoot $entry.Key) `
            "s3://$Bucket/artifacts/$($entry.Value)" --profile $Profile --only-show-errors | Out-Null
    }
    Invoke-Aws s3 sync (Join-Path $projectRoot 'src\ecommerce_rag') `
        "s3://$Bucket/artifacts/source/ecommerce_rag" --profile $Profile `
        --exclude '*' --include '*.py' --only-show-errors | Out-Null
    Invoke-Aws s3 sync (Join-Path $projectRoot 'data\raw\olist') `
        "s3://$Bucket/ecommerce-rag/raw/olist" --profile $Profile --only-show-errors | Out-Null
    Invoke-Aws s3 cp (Join-Path $reportRoot 'local_manifest.json') `
        "s3://$Bucket/reference/local_manifest.json" --profile $Profile --only-show-errors | Out-Null
    Write-Host 'Preparazione S3 completata.' -ForegroundColor Green
}

function Launch-Phase10 {
    New-Item -ItemType Directory -Path $reportRoot -Force | Out-Null
    $subnet = Invoke-Aws ec2 describe-subnets --region $Region --profile $Profile `
        --filters 'Name=default-for-az,Values=true' `
        --query 'sort_by(Subnets,&AvailabilityZone)[0].SubnetId' --output text
    $runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("phase10-emr-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
    try {
        $instanceGroups = @(
            @{ Name = 'Primary'; InstanceGroupType = 'MASTER'; InstanceType = $InstanceType; InstanceCount = 1 },
            @{ Name = 'Core'; InstanceGroupType = 'CORE'; InstanceType = $InstanceType; InstanceCount = 2 }
        )
        $bootstrapActions = @(
            @{ Name = 'Install Phase 10 Python dependencies'; Path = "s3://$Bucket/artifacts/bootstrap_phase10.sh"; Args = @() }
        )
        $actions = @('pipeline', 'prepare_rag', 'manifest', 'scalability')
        $steps = foreach ($action in $actions) {
            $factorArguments = if ($action -eq 'scalability') { ' ' + ($Factors -join ' ') } else { '' }
            $command = "aws s3 cp s3://$Bucket/artifacts/phase10_step.sh /tmp/phase10_step.sh --only-show-errors && chmod +x /tmp/phase10_step.sh && /tmp/phase10_step.sh $action $Bucket $runId$factorArguments"
            @{
                Name = "phase10-$action"
                ActionOnFailure = 'TERMINATE_CLUSTER'
                Type = 'CUSTOM_JAR'
                Jar = 'command-runner.jar'
                Args = @('bash', '-c', $command)
            }
        }
        $instanceGroupsPath = Join-Path $temporaryRoot 'instance-groups.json'
        $bootstrapPath = Join-Path $temporaryRoot 'bootstrap-actions.json'
        $stepsPath = Join-Path $temporaryRoot 'steps.json'
        Write-Utf8NoBom -Path $instanceGroupsPath -Content (ConvertTo-Json -InputObject $instanceGroups -Depth 8)
        Write-Utf8NoBom -Path $bootstrapPath -Content (ConvertTo-Json -InputObject $bootstrapActions -Depth 8)
        Write-Utf8NoBom -Path $stepsPath -Content (ConvertTo-Json -InputObject $steps -Depth 8)

        Write-Host "Avvio EMR ${InstanceType}: 1 primary + 2 core..." -ForegroundColor Cyan
        $clusterId = Invoke-Aws emr create-cluster `
            --name "ecommerce-rag-phase10-$runId" `
            --release-label emr-7.9.0 `
            --applications Name=Spark `
            --service-role EMR_DefaultRole `
            --ec2-attributes "InstanceProfile=EMR_EC2_DefaultRole,SubnetId=$subnet" `
            --instance-groups "file://$instanceGroupsPath" `
            --bootstrap-actions "file://$bootstrapPath" `
            --steps "file://$stepsPath" `
            --log-uri "s3://$Bucket/logs/$runId/" `
            --auto-terminate `
            --no-termination-protected `
            --visible-to-all-users `
            --tags Project=ecommerce-rag Phase=10 `
            --region $Region --profile $Profile `
            --query ClusterId --output text
        $state = @{
            schema_version = '1.0'
            cluster_id = $clusterId
            run_id = $runId
            bucket = $Bucket
            region = $Region
            profile = $Profile
            release_label = 'emr-7.9.0'
            spark_version = '3.5.5'
            instance_type = $InstanceType
            primary_nodes = 1
            core_nodes = 2
            factors = $Factors
            launched_at_utc = (Get-Date).ToUniversalTime().ToString('o')
        }
        $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
        Write-Host "Cluster avviato: $clusterId" -ForegroundColor Green
    }
    finally {
        $resolvedTemporaryRoot = [System.IO.Path]::GetFullPath($temporaryRoot)
        $resolvedSystemTemp = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath())
        if ($resolvedTemporaryRoot.StartsWith($resolvedSystemTemp, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedTemporaryRoot -Recurse -Force
        }
    }
}

function Wait-Phase10 {
    if (-not (Test-Path -LiteralPath $statePath)) {
        throw "Stato della run non trovato: $statePath"
    }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $lastSummary = ''
    while ($true) {
        $clusterState = Invoke-Aws emr describe-cluster --cluster-id $state.cluster_id `
            --region $state.region --profile $state.profile --query 'Cluster.Status.State' --output text
        $stepSummary = Invoke-Aws emr list-steps --cluster-id $state.cluster_id `
            --region $state.region --profile $state.profile `
            --query 'reverse(Steps)[].join(`: `,[Name,Status.State])' --output text
        $summary = "$clusterState | $stepSummary"
        if ($summary -ne $lastSummary) {
            Write-Host $summary
            $lastSummary = $summary
        }
        if ($clusterState -in @('TERMINATED', 'TERMINATED_WITH_ERRORS')) {
            if ($clusterState -eq 'TERMINATED_WITH_ERRORS') {
                throw "Il cluster $($state.cluster_id) è terminato con errori."
            }
            break
        }
        Start-Sleep -Seconds 20
    }
}

function Collect-Phase10 {
    if (-not (Test-Path -LiteralPath $statePath)) {
        throw "Stato della run non trovato: $statePath"
    }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $cloudRoot = Join-Path $reportRoot 'cloud'
    New-Item -ItemType Directory -Path $cloudRoot -Force | Out-Null
    Invoke-Aws s3 sync "s3://$($state.bucket)/results/$($state.run_id)" $cloudRoot `
        --profile $state.profile --only-show-errors | Out-Null
    Invoke-Aws emr describe-cluster --cluster-id $state.cluster_id --region $state.region `
        --profile $state.profile --output json |
        Set-Content -LiteralPath (Join-Path $reportRoot 'cluster.json') -Encoding utf8
    Invoke-Aws emr list-steps --cluster-id $state.cluster_id --region $state.region `
        --profile $state.profile --output json |
        Set-Content -LiteralPath (Join-Path $reportRoot 'steps.json') -Encoding utf8

    & docker compose exec -T app python3 -m ecommerce_rag.cloud.validation compare `
        --local /workspace/reports/aws/phase10/local_manifest.json `
        --cloud /workspace/reports/aws/phase10/cloud/manifest/cloud_manifest.json `
        --output /workspace/reports/aws/phase10/compatibility.json
    if ($LASTEXITCODE -ne 0) {
        throw 'Gli output locali e AWS non risultano compatibili.'
    }
    & docker compose exec -T app python3 -m ecommerce_rag.cloud.reporting `
        --local /workspace/reports/scalability/phase9/summary.json `
        --cloud /workspace/reports/aws/phase10/cloud/scalability/summary.json `
        --output-root /workspace/reports/aws/phase10
    if ($LASTEXITCODE -ne 0) {
        throw 'Creazione del confronto locale/AWS non riuscita.'
    }
    Write-Host "Risultati raccolti in $reportRoot" -ForegroundColor Green
}

Push-Location $projectRoot
try {
    Assert-Prerequisites
    switch ($Mode) {
        'Prepare' { Prepare-Phase10 }
        'Launch' { Launch-Phase10 }
        'Wait' { Wait-Phase10 }
        'Collect' { Collect-Phase10 }
        'All' {
            Prepare-Phase10
            Launch-Phase10
            Wait-Phase10
            Collect-Phase10
        }
    }
}
finally {
    Pop-Location
}
