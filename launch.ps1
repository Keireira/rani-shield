param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 7876
)

$ErrorActionPreference = 'Stop'
# Use the same foreground lifecycle, environment and port checks as make run.
& (Join-Path $PSScriptRoot 'manage.ps1') -Command run -Port $Port
