[CmdletBinding()]
param(
    [ValidateSet('collect', 'score')]
    [string]$Mode = 'score',
    [ValidateSet('reference', 'latest')]
    [string]$SampleSource = 'reference'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $arguments = @(
        'compose', 'exec', '-T', 'app', '/opt/spark/bin/spark-submit',
        '--master', 'spark://spark-master:7077',
        '--conf', 'spark.driver.host=app',
        '--conf', 'spark.driver.bindAddress=0.0.0.0',
        '--conf', 'spark.driver.port=39000',
        '--conf', 'spark.blockManager.port=39001',
        '/workspace/src/ecommerce_rag/evaluation/theme_classification.py',
        $Mode, '--environment', 'local', '--sample-source', $SampleSource
    )
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Valutazione dei complaint theme non riuscita.'
    }
}
finally {
    Pop-Location
}
