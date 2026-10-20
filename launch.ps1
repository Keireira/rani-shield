param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 7876
)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Не найдено подготовленное окружение .venv. Сначала завершите установку FaceShield.'
}

$env:FACESHIELD_PORT = [string]$Port
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:HF_HOME = Join-Path $PSScriptRoot 'weights\cache\huggingface'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonPath -u (Join-Path $PSScriptRoot 'app.py')
    $appExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $appExitCode



