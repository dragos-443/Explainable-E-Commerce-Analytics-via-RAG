[CmdletBinding()]
param(
    [ValidateSet('collect', 'score', 'score-development')]
    [string]$Mode = 'score'
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
        '/workspace/src/ecommerce_rag/evaluation/theme_improvement.py',
        $Mode, '--environment', 'local'
    )
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Valutazione del miglioramento dei complaint theme non riuscita.'
    }
}
finally {
    Pop-Location
}
