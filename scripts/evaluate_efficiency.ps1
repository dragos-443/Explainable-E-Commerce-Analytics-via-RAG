$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $chromaBytes = (& docker compose exec -T chroma sh -c 'du -sb /chroma/chroma | cut -f1' | Out-String).Trim()
    $translationCacheBytes = (& docker compose exec -T app sh -c 'stat -c %s /tmp/app-home/translation-cache/translations.sqlite3' | Out-String).Trim()
    if ($chromaBytes -notmatch '^\d+$' -or $translationCacheBytes -notmatch '^\d+$') {
        throw 'Impossibile misurare lo spazio di Chroma o della cache traduzioni.'
    }
    $arguments = @(
        'compose', 'exec', '-T', 'app', '/opt/spark/bin/spark-submit',
        '--master', 'spark://spark-master:7077',
        '--conf', 'spark.driver.host=app',
        '--conf', 'spark.driver.bindAddress=0.0.0.0',
        '--conf', 'spark.driver.port=39000',
        '--conf', 'spark.blockManager.port=39001',
        '/workspace/src/ecommerce_rag/evaluation/efficiency.py',
        '--environment', 'local',
        '--sample-size', '512',
        '--repetitions', '3',
        '--chroma-bytes', $chromaBytes,
        '--translation-cache-bytes', $translationCacheBytes
    )
    & docker @arguments
    if ($LASTEXITCODE -ne 0) {
        throw 'Benchmark di efficienza non riuscito.'
    }
}
finally {
    Pop-Location
}
