[CmdletBinding()]
param(
    [ValidateSet('smoke', 'phase1')]
    [string]$Pipeline = 'smoke'
)

$ErrorActionPreference = 'Stop'

switch ($Pipeline) {
    'smoke' { & "$PSScriptRoot\smoke_test.ps1" }
    'phase1' {
        $projectRoot = Split-Path -Parent $PSScriptRoot
        Push-Location $projectRoot
        try {
            & "$PSScriptRoot\load_dataset.ps1"
            if ($LASTEXITCODE -ne 0) {
                throw 'Caricamento del dataset raw non riuscito.'
            }

            docker compose exec -T app /opt/spark/bin/spark-submit `
                --master spark://spark-master:7077 `
                --conf spark.driver.host=app `
                --conf spark.driver.bindAddress=0.0.0.0 `
                --conf spark.driver.port=39000 `
                --conf spark.blockManager.port=39001 `
                /workspace/src/ecommerce_rag/preprocessing/pipeline.py `
                --environment local
            if ($LASTEXITCODE -ne 0) {
                throw 'Pipeline PySpark della Fase 1 non riuscita.'
            }
        }
        finally {
            Pop-Location
        }
    }
}
