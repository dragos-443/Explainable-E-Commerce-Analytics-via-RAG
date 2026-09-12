[CmdletBinding()]
param(
    [string]$InputPath = 'report.md',
    [string]$OutputDirectory = 'reports/final'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$source = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $InputPath))
$outputRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputDirectory))
$htmlPath = Join-Path $outputRoot 'report.html'
$pdfPath = Join-Path $outputRoot 'report.pdf'

if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Report Markdown non trovato: $source"
}
if (-not $source.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    -not $outputRoot.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Input e output devono trovarsi nella directory del progetto.'
}

$edgeCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'),
    (Join-Path $env:ProgramFiles 'Microsoft\Edge\Application\msedge.exe')
)
$edge = $edgeCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $edge) {
    throw 'Microsoft Edge non trovato. Installarlo o adattare lo script a un browser Chromium.'
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$composeFile = Join-Path $projectRoot 'compose.yml'
$composeConfig = docker compose -f $composeFile config --format json | ConvertFrom-Json
if ($LASTEXITCODE -ne 0 -or -not $composeConfig.name -or -not $composeConfig.services.app) {
    throw 'Impossibile risolvere il servizio app dalla configurazione Compose.'
}
$imageName = if ($composeConfig.services.app.image) {
    $composeConfig.services.app.image
} else {
    "$($composeConfig.name)-app"
}
docker image inspect $imageName *> $null
if ($LASTEXITCODE -ne 0) {
    docker compose -f $composeFile build app
    if ($LASTEXITCODE -ne 0) { throw "Build dell'immagine applicativa non riuscita." }
}
docker image inspect $imageName *> $null
if ($LASTEXITCODE -ne 0) { throw 'Immagine applicativa Docker non disponibile.' }

$baseUri = ([System.Uri]($projectRoot.TrimEnd('\') + '\')).AbsoluteUri
$containerInput = '/workspace/' + $source.Substring($projectRoot.Length).TrimStart('\').Replace('\', '/')
$containerHtml = '/workspace/' + $htmlPath.Substring($projectRoot.Length).TrimStart('\').Replace('\', '/')

docker run --rm `
    --volume "${projectRoot}:/workspace" `
    --entrypoint python3 `
    $imageName `
    /workspace/scripts/render_report.py `
    --input $containerInput `
    --output $containerHtml `
    --base-uri $baseUri
if ($LASTEXITCODE -ne 0) { throw 'Conversione Markdown-HTML non riuscita.' }

$htmlUri = ([System.Uri]$htmlPath).AbsoluteUri
& $edge `
    --headless `
    --disable-gpu `
    --allow-file-access-from-files `
    --no-pdf-header-footer `
    "--print-to-pdf=$pdfPath" `
    $htmlUri | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Esportazione PDF non riuscita.' }

if (-not (Test-Path -LiteralPath $pdfPath -PathType Leaf)) {
    throw 'Microsoft Edge non ha prodotto il PDF atteso.'
}
$signature = [System.IO.File]::ReadAllBytes($pdfPath)[0..4]
if ([System.Text.Encoding]::ASCII.GetString($signature) -ne '%PDF-') {
    throw 'Il file generato non possiede una firma PDF valida.'
}

$pdf = Get-Item -LiteralPath $pdfPath
Write-Host "Report HTML: $htmlPath" -ForegroundColor Green
Write-Host "Report PDF:  $pdfPath ($($pdf.Length) byte)" -ForegroundColor Green
