$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$managerPath = Join-Path $projectRoot 'manage.ps1'
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython -PathType Leaf)) { throw 'Foreground checks require the prepared local .venv.' }

function Assert-True($Value, [string]$Message) { if (-not $Value) { throw $Message } }

$tokens = $null; $parseErrors = $null
$managerAst = [Management.Automation.Language.Parser]::ParseFile($managerPath, [ref]$tokens, [ref]$parseErrors)
Assert-True ($parseErrors.Count -eq 0) 'The manager must parse before testing foreground behavior.'
$startAst = $managerAst.Find({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Start-Server' }, $true)
Assert-True ($null -ne $startAst) 'The manager has no Start-Server function.'
$detachedCommands = @($startAst.FindAll({ param($node) $node -is [Management.Automation.Language.CommandAst] -and $node.GetCommandName() -in @('Start-Process', 'Start-Job', 'Start-ThreadJob') }, $true))
Assert-True ($detachedCommands.Count -eq 0) 'Start-Server must not wrap the native call in a background job or detached process.'
$redirects = @($startAst.FindAll({ param($node) $node -is [Management.Automation.Language.RedirectionAst] }, $true))
Assert-True ($redirects.Count -eq 0) 'Start-Server must inherit native stdout/stderr without redirection.'

$fixture = Join-Path ([IO.Path]::GetTempPath()) ('Rani Shield foreground test ' + [guid]::NewGuid().ToString('N'))
$savedEnvironment = @{}
foreach ($name in @('FACESHIELD_PORT', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'RANI_FOREGROUND_TEST_EXIT')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
New-Item -ItemType Directory -Path $fixture | Out-Null
try {
    # This fixture never imports the application or opens a socket.
    $fixtureSource = @'
import json
import os
from pathlib import Path
import sys
import time

root = Path(__file__).parent
print("fixture stdout before completion", flush=True)
print("fixture stderr before completion", file=sys.stderr, flush=True)
time.sleep(0.2)
(root / "completed.json").write_text(json.dumps({
    "arguments": sys.argv[1:],
    "unbuffered": sys.stdout.write_through,
    "environment": {name: os.environ.get(name) for name in
                    ("FACESHIELD_PORT", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
}), encoding="utf-8")
print("fixture completed", flush=True)
sys.exit(int(os.environ.get("RANI_FOREGROUND_TEST_EXIT", "0")))
'@
    [IO.File]::WriteAllText((Join-Path $fixture 'gui.py'), $fixtureSource, [Text.UTF8Encoding]::new($false))
    . $managerPath -LibraryOnly -ProjectRoot $fixture -Port 17876
    $script:Python = $localPython
    $script:FixtureOwned = $false
    $script:FixturePortBusy = $false
    $script:RuntimeChecks = 0
    function Assert-LocalRuntime { $script:RuntimeChecks++ }
    function Read-ServerRecord {
        if ($script:FixtureOwned) { return [pscustomobject]@{pid=12345; port=17876} }
        return $null
    }
    function Get-OwnedServer($Record) {
        if ($script:FixtureOwned) { return [pscustomobject]@{Id=12345} }
        return $null
    }
    function Get-PortListener([int]$ServerPort) {
        if ($script:FixturePortBusy) { return [pscustomobject]@{OwningProcess=54321} }
        return $null
    }
    function Start-Process { throw 'Foreground run must not detach through Start-Process.' }
    function Stop-Process { throw 'Foreground run must not stop any existing process.' }
    function Invoke-Checked { throw 'Foreground run must preserve native output and interrupt handling directly.' }

    $env:FACESHIELD_PORT = 'before-run'
    $env:OMP_NUM_THREADS = '7'
    [Environment]::SetEnvironmentVariable('MKL_NUM_THREADS', $null, 'Process')
    $marker = Join-Path $fixture 'completed.json'
    function Assert-RestoredEnvironment {
        Assert-True ($env:FACESHIELD_PORT -eq 'before-run') 'FACESHIELD_PORT was not restored.'
        Assert-True ($env:OMP_NUM_THREADS -eq '7') 'OMP_NUM_THREADS was not restored.'
        Assert-True ($null -eq [Environment]::GetEnvironmentVariable('MKL_NUM_THREADS', 'Process')) 'An originally absent MKL_NUM_THREADS was not restored.'
        Assert-True (-not (Test-Path -LiteralPath (Join-Path $fixture '.cache'))) 'Foreground run must not write PID records or redirected log files.'
    }

    foreach ($exitCode in @(0, 130, -1073741510, 7)) {
        if (Test-Path -LiteralPath $marker) { Remove-Item -LiteralPath $marker }
        $env:RANI_FOREGROUND_TEST_EXIT = [string]$exitCode
        $outputLines = [Collections.Generic.List[string]]::new()
        $earlyLines = [Collections.Generic.List[string]]::new()
        $caught = $null
        $watch = [Diagnostics.Stopwatch]::StartNew()
        # Windows PowerShell 5.1 wraps captured native stderr in ErrorRecord.
        # Continue permits inspection of that stream without changing run itself.
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            Start-Server 2>&1 | ForEach-Object {
                $line = [string]$_
                $outputLines.Add($line)
                if (-not (Test-Path -LiteralPath $marker)) { $earlyLines.Add($line) }
            }
        }
        catch { $caught = $_ }
        finally {
            $ErrorActionPreference = $previousPreference
            $watch.Stop()
        }
        Assert-True (Test-Path -LiteralPath $marker) "Run returned before the fixture completed (exit $exitCode)."
        Assert-True ($watch.ElapsedMilliseconds -ge 150) 'Run did not wait for the native foreground process.'
        Assert-True ($outputLines.Contains('fixture stdout before completion')) 'Native stdout did not reach the invoking terminal stream.'
        Assert-True (($outputLines -join "`n") -match 'fixture stderr before completion') 'Native stderr did not reach the invoking terminal stream.'
        Assert-True ($earlyLines.Contains('fixture stdout before completion')) 'Native stdout was buffered until process completion.'
        Assert-True ($outputLines.Contains('fixture completed')) 'The final native output was lost.'
        if ($exitCode -eq 7) {
            Assert-True ($null -ne $caught -and $caught.Exception.Message -match '7') 'An unexpected native failure must report its exit code.'
        }
        else { Assert-True ($null -eq $caught) "Normal completion or Ctrl+C exit $exitCode must not be reported as a launch failure: $caught" }
        $completed = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
        Assert-True (($completed.arguments -join '|') -eq '--host|127.0.0.1|--port|17876|--no-browser') 'Foreground CLI arguments or paths containing spaces were not preserved.'
        Assert-True $completed.unbuffered 'The foreground Python interpreter must receive -u for immediate logs.'
        Assert-True ($completed.environment.FACESHIELD_PORT -eq '17876') 'The foreground child received the wrong port environment.'
        Assert-True ($completed.environment.OMP_NUM_THREADS -eq '4' -and $completed.environment.MKL_NUM_THREADS -eq '4') 'The foreground child received the wrong thread settings.'
        Assert-RestoredEnvironment
    }
    Assert-True ($script:RuntimeChecks -eq 4) 'Foreground runs must retain the local-runtime preflight.'

    foreach ($busyKind in @('port', 'owned')) {
        Remove-Item -LiteralPath $marker
        $script:FixturePortBusy = $busyKind -eq 'port'
        $script:FixtureOwned = $busyKind -eq 'owned'
        $caught = $null
        try { Start-Server | Out-Null } catch { $caught = $_ }
        Assert-True ($null -ne $caught) "An existing $busyKind server must be refused instead of reused or stopped."
        Assert-True (-not (Test-Path -LiteralPath $marker)) "The child was started despite an existing $busyKind server."
        Assert-RestoredEnvironment
        # Recreate only the harmless marker so the next case can remove it.
        [IO.File]::WriteAllText($marker, '{}')
    }
    Write-Host 'PASS: foreground native wait; streaming stdout/stderr; CLI arguments with spaced paths; exit/interrupt handling; environment restoration; busy port and existing background refusal; no PID/log files or process changes.'
}
finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedFixture.StartsWith($temporaryRoot, [StringComparison]::OrdinalIgnoreCase) -and (Split-Path -Leaf $resolvedFixture).StartsWith('Rani Shield foreground test ')) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
