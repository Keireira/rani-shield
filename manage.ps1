[CmdletBinding()]
param(
    [ValidateSet('help', 'setup', 'models', 'configure', 'doctor', 'run', 'stop', 'status', 'test', 'benchmark', 'benchmark-help')]
    [string]$Command = 'help',
    [ValidateRange(1024, 65535)][int]$Port = 7876,
    [string]$Checkpoints,
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

function Install-Environment {
    if ($DryRun) {
        Write-Host '[dry-run] Create Python 3.11 .venv only if absent; verify installed package versions first.'
        Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch==2.7.1+cu128', 'torchvision==0.22.1+cu128', '--index-url', 'https://download.pytorch.org/whl/cu128')
        Invoke-Checked $script:Python @('-m', 'pip', 'install', '-r', (Join-Path $script:Root 'requirements-windows.txt'))
        return
    }
    if (-not (Test-Path -LiteralPath $script:Python -PathType Leaf)) {
        if ($Python311) { Invoke-Checked $Python311 @('-m', 'venv', (Join-Path $script:Root '.venv')) }
        elseif (Get-Command py.exe -ErrorAction SilentlyContinue) { Invoke-Checked 'py.exe' @('-3.11', '-m', 'venv', (Join-Path $script:Root '.venv')) }
        else { throw 'Install Python 3.11 with the Windows py launcher, or pass -Python311 "C:\path\python.exe".' }
    }
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
    Invoke-Checked $script:Python @((Join-Path $script:Root 'download_models.py'))
    Invoke-Checked $script:Python @((Join-Path $script:Root 'download_phantomseal.py'), '--root', $script:Root)
    $adapter = Join-Path $script:Root 'weights\ip-adapter\ip-adapter_sd15.bin'
    if (-not (Test-Path -LiteralPath $adapter -PathType Leaf) -or (Get-Item -LiteralPath $adapter).Length -eq 0) {
        $url = 'https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin'
        if ($DryRun) { Write-Host "[dry-run] Download missing IP-Adapter: $url -> $adapter" }
        else {
            New-Item -ItemType Directory -Path (Split-Path -Parent $adapter) -Force | Out-Null
            $partial = $adapter + '.part'
            Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing
            Move-Item -LiteralPath $partial -Destination $adapter -Force
        }
    }
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
    elseif (Get-PortListener $Port) { Write-Host "Port $Port is occupied by an unmanaged process. It will not be adopted or stopped." }
    else { Write-Host 'Managed server is not running.' }
}

function Start-Server {
    Require-Python
    $gui = Join-Path $script:Root 'gui.py'
    $arguments = @('-u', $gui, '--host', '127.0.0.1', '--port', [string]$Port, '--no-browser')
    if ($DryRun) {
        Write-Host "[dry-run] Check owned PID, refuse an occupied port, then start hidden on $Port; write own PID/start time and logs."
        Invoke-Checked $script:Python $arguments
        return
    }
    $record = Read-ServerRecord
    $owned = Get-OwnedServer $record
    if ($owned) {
        if ([int]$record.port -ne $Port) { throw "A managed server is already running on $($record.port). Stop it before changing ports." }
        if (Test-OwnedHealth $record $owned) { Write-Host "Already running: http://127.0.0.1:$Port (PID $($owned.Id))"; return }
        throw "Managed server exists but is not healthy. Check $script:StateDir or use stop."
    }
    if (Get-PortListener $Port) { throw "Port $Port is occupied. Use -Port with a free port; no process was changed." }
    New-Item -ItemType Directory -Path $script:StateDir -Force | Out-Null
    $oldEnvironment = @{}
    foreach ($name in @('FACESHIELD_PORT', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'HF_HOME')) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
    try {
        $env:FACESHIELD_PORT = [string]$Port; $env:OMP_NUM_THREADS = '4'; $env:MKL_NUM_THREADS = '4'
        $env:HF_HOME = Join-Path $script:Root 'weights\cache\huggingface'
        $nativeArguments = ($arguments | ForEach-Object { Quote-NativeArgument $_ }) -join ' '
        $started = Start-Process -FilePath $script:Python -ArgumentList $nativeArguments -WorkingDirectory $script:Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $script:StateDir 'server.stdout.log') -RedirectStandardError (Join-Path $script:StateDir 'server.stderr.log') -PassThru
        $launcherCim = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$started.Id) -ErrorAction Stop
        $launcherRecord = New-ServerRecord $started $Port $null $launcherCim
        Write-Verbose ("Launched process identity: " + ($launcherRecord | ConvertTo-Json -Compress))
    }
    finally { foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') } }
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    do {
        $started.Refresh()
        $launcherAlive = -not $started.HasExited
        $listeners = Get-PortListener $Port
        Write-Verbose "Waiting for port $Port; launcher PID $($launcherRecord.pid) alive=$launcherAlive."
        foreach ($listener in $listeners) {
            try {
                $server = Get-Process -Id $listener.OwningProcess -ErrorAction Stop
                $cim = Get-CimInstance Win32_Process -Filter ('ProcessId=' + [int]$listener.OwningProcess) -ErrorAction Stop
                $matchesLaunch = Test-LaunchedServer $launcherRecord $server $cim
                Write-Verbose "Listener PID $($server.Id), parent PID $($cim.ParentProcessId), start ticks $($server.StartTime.ToUniversalTime().Ticks), matches launch=$matchesLaunch."
                if ($launcherAlive -and $matchesLaunch) {
                    $newRecord = New-ServerRecord $server $Port $launcherRecord $cim
                    $newRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $script:PidFile -Encoding UTF8
                    if (Test-Health $Port) { Write-Host "Ready: http://127.0.0.1:$Port (PID $($server.Id))"; return }
                }
            }
            catch { Write-Verbose "Listener identity is not available yet: $_" }
        }
        if (-not $launcherAlive) { throw "Server launcher exited. Read $script:StateDir\server.stderr.log" }
        Start-Sleep -Milliseconds 400
    } while ([DateTime]::UtcNow -lt $deadline)
    throw "Server did not become healthy within 45 seconds; it was left running. Use status, inspect logs, or stop."
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
        Invoke-Checked $script:Python @('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v')
        if (Get-Command node.exe -ErrorAction SilentlyContinue) { Invoke-Checked 'node.exe' @((Join-Path $script:Root 'tests\frontend-check.cjs')) }
        else { Write-Host 'Node.js not found; optional frontend regression skipped.' }
    }
    finally { $env:CUDA_VISIBLE_DEVICES = $oldCuda; $env:FACESHIELD_ROOT = $oldRoot }
}

function Invoke-Benchmark {
    Require-Python
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
try {
    Push-Location -LiteralPath $script:Root
    try {
        switch ($Command) {
            help { Write-Host @'
FaceShield Windows commands (make is optional)
  manage.cmd setup             Create/reuse Python 3.11 + pinned CUDA 12.8 environment
  manage.cmd models            Download public base weights; reuse present files
  manage.cmd configure -Checkpoints "D:\models\checkpoints" -Cloaks "D:\data\vggface2_cloak\mix_full"
  manage.cmd doctor            Check dependencies, CUDA, 4 branches and full 214-face bank
  manage.cmd run [-Port 7876]   Start hidden; reuse this tool's healthy server
  manage.cmd status            Show managed PID/HTTP health
  manage.cmd stop              Stop only this tool's verified process
  manage.cmd test              Run CPU contract tests; optional Node frontend tests
  manage.cmd benchmark         Run/resume 36 pairs / 108 swaps (GPU, about 10 minutes)
  manage.cmd benchmark -Sources 1 -OutputDir "results\benchmarks\quick"
  manage.cmd benchmark-help    Show benchmark options
Add -DryRun to print a plan without installing, writing, starting or stopping.
'@ }
            setup { Install-Environment }
            models { Install-Models }
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
            { $_ -in @('benchmark', 'benchmark-help') } { Invoke-Benchmark }
        }
    }
    finally { Pop-Location }
}
catch { Write-Error $_; exit 1 }
