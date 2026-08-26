[CmdletBinding()]
param(
    [ValidateSet('smoke')]
    [string]$Pipeline = 'smoke'
)

$ErrorActionPreference = 'Stop'

switch ($Pipeline) {
    'smoke' { & "$PSScriptRoot\smoke_test.ps1" }
}
