[CmdletBinding()]
param(
    [ValidateSet('help', 'setup', 'models', 'import-models', 'configure', 'doctor', 'run', 'stop', 'status', 'test', 'smoke', 'smoke-extra', 'benchmark', 'benchmark-help')]
    [string]$Command = 'help',
    [ValidateRange(1024, 65535)][int]$Port = 7876,
    [string]$Checkpoints,
    [ValidateSet('all', 'nullswap', 'diffswap', 'infoswap', 'uniface')][string]$ModelSet = 'all',
    [string]$Cloaks,
    [string]$Python311,
    [ValidateRange(0, 10000)][int]$Sources = 0,
    [string]$OutputDir = 'results\benchmarks\latest',
    [string]$Manifest = 'benchmark_manifest.json',
    [string]$ProjectRoot,
    [switch]$DryRun,
    [switch]$LibraryOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $ProjectRoot) { $ProjectRoot = $PSScriptRoot }
$script:Root = [IO.Path]::GetFullPath($ProjectRoot)
$script:Python = Join-Path $script:Root '.venv\Scripts\python.exe'
$script:BasePython = Join-Path $script:Root '.runtime\python\python.exe'
$script:Node = Join-Path $script:Root '.runtime\node\node.exe'
$script:Checks = Join-Path $script:Root 'scripts\manage_checks.py'
$script:StateDir = Join-Path $script:Root '.cache\manage'
$script:PidFile = Join-Path $script:StateDir 'server.json'

function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $script:Root $Value))
}

function Quote-NativeArgument([string]$Value) {
    # Windows CommandLineToArgvW quoting, including trailing backslashes.
    return '"' + ([regex]::Replace([regex]::Replace($Value, '(\\*)"', '$1$1\"'), '(\\+)$', '$1$1')) + '"'
}

function Invoke-Checked([string]$File, [string[]]$Arguments) {
    if ($DryRun) {
        Write-Host ('[dry-run] ' + (Quote-NativeArgument $File) + ' ' + (($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join ' '))
        return
    }
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $File" }
}

function Require-Python {
    if (-not $DryRun -and -not (Test-Path -LiteralPath $script:Python -PathType Leaf)) {
        throw 'The local .venv is missing. Run manage.cmd setup first.'
    }
}

function Set-ProjectEnvironment {
    $contract = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'scripts\runtime_environment.json') -Raw | ConvertFrom-Json
    $previous = @{}
    $values = @{}
    foreach ($property in $contract.cache_paths.PSObject.Properties) {
        $path = Resolve-ProjectPath $property.Value
        $ancestor = $path
        while ($ancestor -and $ancestor -ine $script:Root) {
            $existing = Get-Item -LiteralPath $ancestor -Force -ErrorAction SilentlyContinue
            if ($existing -and ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                throw "Runtime cache cannot traverse links or junctions: $ancestor"
            }
            $ancestor = Split-Path -Parent $ancestor
        }
        if (-not $ancestor) { throw "Runtime cache escaped the project: $path" }
        $item = New-Item -ItemType Directory -Path $path -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Runtime cache must be a local directory: $path" }
        $values[$property.Name] = $path
    }
    foreach ($property in $contract.settings.PSObject.Properties) { $values[$property.Name] = [string]$property.Value }
    foreach ($name in $contract.unset) { $values[$name] = $null }
    foreach ($name in $values.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $values[$name], 'Process')
    }
    return $previous
}

function Restore-ProjectEnvironment($Previous) {
    foreach ($name in $Previous.Keys) { [Environment]::SetEnvironmentVariable($name, $Previous[$name], 'Process') }
}

function Assert-LocalRuntime {
    if ($DryRun) { return }
    $runtime = (& $script:Python -I $script:Checks runtime --root $script:Root | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $runtime.ready) {
        throw ('Python still depends on files outside this project. Run manage.cmd setup. ' + ($runtime.errors -join '; '))
    }
}

function Install-LocalRuntime {
    $bootstrap = Join-Path $script:Root 'scripts\bootstrap_runtime.py'
    $venvConfig = Join-Path $script:Root '.venv\pyvenv.cfg'
    if (Test-Path -LiteralPath $venvConfig -PathType Leaf) {
        $configuredVersion = [regex]::Match((Get-Content -LiteralPath $venvConfig -Raw), '(?m)^version\s*=\s*(\d+\.\d+)')
        if ($configuredVersion.Success -and $configuredVersion.Groups[1].Value -ne '3.11') {
            throw 'Existing .venv uses a different Python version. Rebuild it manually before setup; incompatible installed packages will not be overwritten.'
        }
    }
    if (-not (Test-Path -LiteralPath $script:BasePython -PathType Leaf)) {
        if ($Python311) { $sourcePython = $Python311 }
        elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
            $sourcePython = & py.exe -3.11 -c 'import sys; print(sys.executable)'
            if ($LASTEXITCODE -ne 0) { throw 'Could not find Python 3.11 with py.exe.' }
        }
        else { throw 'Install CPython 3.11 once or pass -Python311 "C:\path\python.exe". Setup copies its runtime into .runtime\python.' }
        $sourceRoot = & $sourcePython -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"; print(sys.base_prefix)'
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Python 3.11 source installation.' }
        Invoke-Checked $sourcePython @($bootstrap, '--root', $script:Root, '--source', $sourceRoot)
    }
    # Refresh redirectors/configuration in-place and preserve installed packages.
    Invoke-Checked $script:BasePython @('-m', 'venv', (Join-Path $script:Root '.venv'))
    Invoke-Checked $script:Python @($bootstrap, '--root', $script:Root, '--refresh-scripts')
    Assert-LocalRuntime
    if (-not (Test-Path -LiteralPath $script:Node -PathType Leaf)) {
        $nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
        if ($nodeCommand) {
            $nodeLicense = @('LICENSE', 'LICENSE.txt') | ForEach-Object { Join-Path (Split-Path -Parent $nodeCommand.Source) $_ } | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
            if ($nodeLicense) { Invoke-Checked $script:Python @($bootstrap, '--root', $script:Root, '--node-source', $nodeCommand.Source, '--node-license', $nodeLicense) }
            else { Write-Host 'Optional frontend checks need .runtime\node\node.exe plus LICENSE. See README for Node setup.' }
        }
    }
}

function Install-Environment {
    if ($DryRun) {
        Write-Host '[dry-run] Copy CPython 3.11 into .runtime\python if absent; refresh .venv locally, preserving installed packages; verify package versions first.'
        Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch==2.7.1+cu128', 'torchvision==0.22.1+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', (Join-Path $script:Root 'requirements-windows.txt'))
        return
    }
    Install-LocalRuntime
    $report = (& $script:Python $script:Checks environment --root $script:Root | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the local environment.' }
    if (-not $report.python_ready) { throw 'Existing .venv is not Python 3.11. Select a separate project directory or rebuild it manually; setup will not delete it.' }
    if ($report.ready) { Write-Host 'Environment is already ready; nothing to install.'; return }
    if (-not $report.torch_ready) {
        Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch==2.7.1+cu128', 'torchvision==0.22.1+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
    }
    if (-not $report.requirements_ready) {
        Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', (Join-Path $script:Root 'requirements-windows.txt'))
    }
    Invoke-Checked $script:Python @('-m', 'pip', 'check')
    $report = (& $script:Python $script:Checks environment --root $script:Root | ConvertFrom-Json)
    if ($LASTEXITCODE -ne 0 -or -not $report.ready) { throw ('Environment verification failed: ' + ($report.errors -join '; ')) }
    Write-Host 'Environment ready. Next: manage.cmd models, configure, doctor.'
}

function Install-Models {
    Require-Python
    Assert-LocalRuntime
    Invoke-Checked $script:Python @((Join-Path $script:Root 'download_models.py'))
    Invoke-Checked $script:Python @((Join-Path $script:Root 'download_phantomseal.py'), '--root', $script:Root)
    Write-Host 'Public downloads reuse existing files. Full PhantomSeal checkpoints and mix_full bank: manage.cmd configure.'
}

function Read-ServerRecord {
    if (-not (Test-Path -LiteralPath $script:PidFile -PathType Leaf)) { return $null }
    try { return Get-Content -LiteralPath $script:PidFile -Raw | ConvertFrom-Json }
    catch { throw "Invalid managed server record: $script:PidFile. No process was changed." }
}

function Get-ProcessExecutable($ProcessInfo, $CimInfo = $null) {
    $executable = $null
    try { $executable = $ProcessInfo.Path } catch { }
    # A 32-bit GNU Make may launch 32-bit PowerShell, whose Process.Path is null
    # for 64-bit Python. Win32_Process exposes the executable across architectures.
    if (-not $executable -and $CimInfo) { $executable = $CimInfo.ExecutablePath }
    return $executable
}

function Test-ServerIdentity($Record, $ProcessInfo, $CommandLine, $CimInfo = $null) {
    if (-not $Record -or -not $ProcessInfo) { return $false }
    try {
        $escapedScript = [regex]::Escape([string]$Record.script)
        $scriptAlternatives = '"' + $escapedScript + '"'
        if ($Record.script -notmatch '\s') { $scriptAlternatives += '|' + $escapedScript }
        $scriptPattern = '(?:^|\s)(?:' + $scriptAlternatives + ')(?=\s|$)'
        $escapedPort = [regex]::Escape([string]$Record.port)
        $portPattern = '(?:^|\s)(?:"--port"|--port)\s+(?:"' + $escapedPort + '"|' + $escapedPort + ')(?=\s|$)'
        $executable = Get-ProcessExecutable $ProcessInfo $CimInfo
        return $Record.root -ieq $script:Root -and
            [int]$Record.pid -eq [int]$ProcessInfo.Id -and
            [string]$Record.start_ticks -eq [string]$ProcessInfo.StartTime.ToUniversalTime().Ticks -and
            $Record.executable -and $executable -and $Record.executable -ieq $executable -and
            $Record.script -ieq (Join-Path $script:Root 'gui.py') -and
            [int]$Record.port -ge 1024 -and [int]$Record.port -le 65535 -and
            $CommandLine -match $scriptPattern -and
            $CommandLine -match $portPattern
    }
    catch { return $false }
}

function Get-OwnedServer($Record) {
    if (-not $Record) { return $null }
    try {
        $processInfo = Get-Process -Id ([int]$Record.pid) -ErrorAction Stop
        $cim = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$Record.pid) -ErrorAction Stop
        if (Test-ServerIdentity $Record $processInfo $cim.CommandLine $cim) { return $processInfo }
    }
    catch { return $null }
    return $null
}

function New-ServerRecord($ProcessInfo, [int]$ServerPort, $LauncherRecord = $null, $CimInfo = $null) {
    $executable = Get-ProcessExecutable $ProcessInfo $CimInfo
    if (-not $executable) { throw "Could not determine executable for PID $($ProcessInfo.Id)." }
    $record = [ordered]@{
        root=$script:Root; pid=$ProcessInfo.Id
        start_ticks=[string]$ProcessInfo.StartTime.ToUniversalTime().Ticks
        executable=$executable; script=(Join-Path $script:Root 'gui.py'); port=$ServerPort
    }
    if ($LauncherRecord -and $LauncherRecord.pid -ne $ProcessInfo.Id) { $record.launcher = $LauncherRecord }
    return [pscustomobject]$record
}

function Test-LaunchedServer($LauncherRecord, $ProcessInfo, $CimInfo) {
    try {
        if ($ProcessInfo.Id -eq $LauncherRecord.pid) { return Test-ServerIdentity $LauncherRecord $ProcessInfo $CimInfo.CommandLine $CimInfo }
        # Windows venv python.exe is a redirector; its base-Python child owns the socket.
        if ([int]$CimInfo.ParentProcessId -ne [int]$LauncherRecord.pid -or
            $ProcessInfo.StartTime.ToUniversalTime().Ticks -lt [long]$LauncherRecord.start_ticks -or
            (Split-Path -Leaf (Get-ProcessExecutable $ProcessInfo $CimInfo)) -ine 'python.exe') { return $false }
        $candidate = New-ServerRecord $ProcessInfo ([int]$LauncherRecord.port) $LauncherRecord $CimInfo
        return Test-ServerIdentity $candidate $ProcessInfo $CimInfo.CommandLine $CimInfo
    }
    catch { return $false }
}

function Test-Health([int]$ServerPort) {
    try { return (Invoke-RestMethod -Uri "http://127.0.0.1:$ServerPort/health" -TimeoutSec 2).ok -eq $true }
    catch { return $false }
}

function Get-PortListener([int]$ServerPort) {
    return @(Get-NetTCPConnection -State Listen -LocalPort $ServerPort -ErrorAction SilentlyContinue)
}

function Test-OwnedHealth($Record, $OwnedProcess) {
    $listeners = Get-PortListener ([int]$Record.port)
    return [bool]($listeners | Where-Object { $_.OwningProcess -eq $OwnedProcess.Id }) -and (Test-Health ([int]$Record.port))
}

function Show-Status {
    if ($DryRun) { Write-Host "[dry-run] Inspect owned PID identity and http://127.0.0.1:$Port/health; no process changes."; return }
    $record = Read-ServerRecord
    $owned = Get-OwnedServer $record
    if ($owned) {
        $healthy = Test-OwnedHealth $record $owned
        Write-Host "Managed server PID $($owned.Id): healthy=$healthy; http://127.0.0.1:$($record.port)"
        Write-Host "Logs: $script:StateDir"
    }
    elseif (Get-PortListener $Port) { Write-Host "Port $Port is occupied. Foreground servers stop with Ctrl+C in their terminal; this process will not be adopted or stopped." }
    else { Write-Host "No legacy background server is running; port $Port has no listener." }
}

function Start-Server {
    Require-Python
    Assert-LocalRuntime
    $gui = Join-Path $script:Root 'gui.py'
    $arguments = @('-u', $gui, '--host', '127.0.0.1', '--port', [string]$Port, '--no-browser')
    if ($DryRun) {
        Write-Host "[dry-run] Refuse an occupied port or an existing legacy background server; run in this terminal on $Port until exit. Stream logs here; Ctrl+C stops the server."
        Invoke-Checked $script:Python $arguments
        return
    }
    $record = Read-ServerRecord
    $owned = Get-OwnedServer $record
    if ($owned) {
        throw "A legacy background server is still running on $($record.port) (PID $($owned.Id)). Stop it once with manage.cmd stop, then run again to use this terminal."
    }
    if (Get-PortListener $Port) { throw "Port $Port is occupied. Use -Port with a free port; no process was changed." }
    $oldEnvironment = @{}
    foreach ($name in @('FACESHIELD_PORT', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:FACESHIELD_PORT = [string]$Port; $env:OMP_NUM_THREADS = '4'; $env:MKL_NUM_THREADS = '4'
        Write-Host "Running in this terminal: http://127.0.0.1:$Port. Press Ctrl+C to stop."
        # Inherit the console and its control events. Do not detach or redirect:
        # Python/Uvicorn owns the lifetime and emits logs directly to this terminal.
        & $script:Python @arguments
        $serverExitCode = $LASTEXITCODE
        if ($serverExitCode -notin @(0, 130, -1073741510)) {
            throw "Server exited with code $serverExitCode. See the terminal output above."
        }
    }
    finally { foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') } }
}

function Stop-Server {
    if ($DryRun) { Write-Host '[dry-run] Stop only PID matching this root, executable, gui.py command line and exact creation time; never kill by port/name.'; return }
    $record = Read-ServerRecord
    $owned = Get-OwnedServer $record
    if (-not $owned) { Write-Host 'No verified managed process to stop. No process was changed.'; return }
    Stop-Process -InputObject $owned -Force
    $owned.WaitForExit(5000) | Out-Null
    Remove-Item -LiteralPath $script:PidFile
    Write-Host "Stopped managed server PID $($record.pid)."
}

function Invoke-Tests {
    Require-Python
    $oldCuda = $env:CUDA_VISIBLE_DEVICES; $oldRoot = $env:FACESHIELD_ROOT
    try {
        $env:CUDA_VISIBLE_DEVICES = '-1'; $env:FACESHIELD_ROOT = $script:Root
        Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $script:Root 'tests\test_manage.ps1'))
        Invoke-Checked 'powershell.exe' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $script:Root 'tests\test_foreground_run.ps1'))
        Invoke-Checked $script:Python @('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
        if (Test-Path -LiteralPath $script:Node -PathType Leaf) { Invoke-Checked $script:Node @((Join-Path $script:Root 'tests\frontend-check.cjs')) }
        else { Write-Host 'Local .runtime\node\node.exe not found; optional frontend regression skipped.' }
    }
    finally { $env:CUDA_VISIBLE_DEVICES = $oldCuda; $env:FACESHIELD_ROOT = $oldRoot }
}

function Invoke-Benchmark {
    Require-Python
    Assert-LocalRuntime
    $arguments = @((Join-Path $script:Root 'benchmark_suite.py'))
    if ($Command -eq 'benchmark-help') { $arguments += '--help' }
    else {
        Write-Host 'Benchmark: 6 sources x 3 targets x PNG/JPEG95 = 36 pairs, 108 swaps across 3 methods. Uses the GPU; allow about 10 minutes. Completed cases are reused.'
        $arguments += @('--manifest', (Resolve-ProjectPath $Manifest), '--out', (Resolve-ProjectPath $OutputDir), '--resume', '--port', [string]$Port)
        if ($Sources -gt 0) { $arguments += @('--max-sources', [string]$Sources) }
    }
    Invoke-Checked $script:Python $arguments
}

if ($LibraryOnly) { return }
$previousEnvironment = @{}
try {
    if (-not $DryRun -and $Command -notin @('help', 'status', 'stop')) { $previousEnvironment = Set-ProjectEnvironment }
    Push-Location -LiteralPath $script:Root
    try {
        switch ($Command) {
            help { Write-Host @'
Rani Shield Windows commands (make is optional)
  manage.cmd setup             Copy local Python runtime; prepare pinned CUDA 12.8 packages
  manage.cmd models            Download public base weights; reuse present files
  manage.cmd import-models -Checkpoints "PATH\TO\checkpoints" [-ModelSet all]
  manage.cmd configure -Checkpoints "weights" -Cloaks "weights\phantomseal\cloaks\mix_full"
  manage.cmd doctor            Check dependencies, CUDA, 4 branches and full 214-face bank
  manage.cmd run [-Port 7876]   Run in this terminal with live logs; Ctrl+C stops
  manage.cmd status            Inspect port / legacy background server health
  manage.cmd stop              Stop a verified background server from older versions
  manage.cmd test              Run CPU contract tests; optional Node frontend tests
  manage.cmd smoke             Check real 1280px GPU chain and swaps without a server
  manage.cmd smoke-extra       Check NullSwap1280 and all four swappers PNG/JPEG95; no server
  manage.cmd benchmark         Run/resume 36 pairs / 108 swaps (GPU, about 10 minutes)
  manage.cmd benchmark -Sources 1 -OutputDir "results\benchmarks\quick"
  manage.cmd benchmark-help    Show benchmark options
Add -DryRun to print a plan without installing, writing, starting or stopping.
'@ }
            setup { Install-Environment }
            models { Install-Models }
            import-models {
                Require-Python
                if (-not $Checkpoints) { throw 'Supply -Checkpoints pointing at an extracted checkpoint directory or archive.' }
                Invoke-Checked $script:Python @((Join-Path $script:Root 'scripts\import_swap_assets.py'), '--checkpoints', (Resolve-ProjectPath $Checkpoints), '--models', $ModelSet)
            }
            configure {
                Require-Python
                if (-not $Checkpoints -or -not $Cloaks) { throw 'Supply -Checkpoints and -Cloaks, pointing at already extracted directories. See help.' }
                Invoke-Checked $script:Python @($script:Checks, 'configure', '--root', $script:Root, '--checkpoints', (Resolve-ProjectPath $Checkpoints), '--cloaks', (Resolve-ProjectPath $Cloaks))
            }
            doctor { Require-Python; Invoke-Checked $script:Python @($script:Checks, 'doctor', '--root', $script:Root) }
            run { Start-Server }
            stop { Stop-Server }
            status { Show-Status }
            test { Invoke-Tests }
            smoke { Require-Python; Assert-LocalRuntime; Invoke-Checked $script:Python @((Join-Path $script:Root 'scripts\smoke_check.py')) }
            smoke-extra { Require-Python; Assert-LocalRuntime; Invoke-Checked $script:Python @((Join-Path $script:Root 'scripts\smoke_check.py'), '--mode', 'nullswap', '--backends', 'simswap', 'diffswap', 'infoswap', 'uniface') }
            { $_ -in @('benchmark', 'benchmark-help') } { Invoke-Benchmark }
        }
    }
    finally { Pop-Location }
}
catch { Write-Error $_; exit 1 }
finally { Restore-ProjectEnvironment $previousEnvironment }
