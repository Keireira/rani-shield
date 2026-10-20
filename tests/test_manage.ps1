$ErrorActionPreference = 'Stop'
$stageRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $stageRoot 'manage.ps1'
$tokens = $null; $parseErrors = $null
[System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$parseErrors) | Out-Null
if ($parseErrors.Count) { throw ($parseErrors.Message -join '; ') }
function Assert-True($Value, [string]$Message) { if (-not $Value) { throw $Message } }
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('FaceShield manage test ' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $fixture | Out-Null
try {
    . $scriptPath -LibraryOnly -ProjectRoot $fixture
    Assert-True ((Quote-NativeArgument 'C:\folder with spaces\') -eq '"C:\folder with spaces\\"') 'Trailing slash/space argument quoting failed.'
    Assert-True ((Quote-NativeArgument 'has"quote') -eq '"has\"quote"') 'Embedded quote argument escaping failed.'
    Assert-True ((Quote-NativeArgument '') -eq '""') 'Empty argument quoting failed.'
    $created = [DateTime]::SpecifyKind([DateTime]'2026-01-02T03:04:05', [DateTimeKind]::Utc)
    $fakeProcess = [pscustomobject]@{Id=12345; StartTime=$created; Path=(Join-Path $fixture '.venv\Scripts\python.exe')}
    $record = [pscustomobject]@{root=$fixture; pid=12345; start_ticks=[string]$created.Ticks; executable=$fakeProcess.Path; script=(Join-Path $fixture 'gui.py'); port=7875}
    $line = '"' + $fakeProcess.Path + '" -u "' + $record.script + '" --host 127.0.0.1 --port 7875 --no-browser'
    Assert-True (Test-ServerIdentity $record $fakeProcess $line) 'A matching managed identity should pass.'
    $fullyQuotedLine = '"' + $fakeProcess.Path + '" "-u" "' + $record.script + '" "--host" "127.0.0.1" "--port" "7875" "--no-browser"'
    Assert-True (Test-ServerIdentity $record $fakeProcess $fullyQuotedLine) 'Fully quoted Start-Process command-line tokens must pass.'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess ($fullyQuotedLine.Replace('"7875"', '"78750"')))) 'A longer quoted port must not match a prefix.'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess ($line.Replace('--port 7875', '--port 78750')))) 'A longer unquoted port must not match a prefix.'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess ($fullyQuotedLine.Replace('gui.py"', 'gui.py.bak"')))) 'Script path must match a complete argument, not a filename prefix.'
    $cim = [pscustomobject]@{ParentProcessId=1; CommandLine=$line}
    Assert-True (Test-LaunchedServer $record $fakeProcess $cim) 'A direct launcher listener should pass.'
    $child = [pscustomobject]@{Id=54321; StartTime=$created.AddMilliseconds(50); Path='C:\Python311\python.exe'}
    $childCim = [pscustomobject]@{ParentProcessId=12345; CommandLine=$line}
    Assert-True (Test-LaunchedServer $record $child $childCim) 'The venv launcher child must be accepted as actual listener.'
    $childCim.CommandLine = $fullyQuotedLine
    Assert-True (Test-LaunchedServer $record $child $childCim) 'The venv child must accept fully quoted real launcher tokens.'
    $crossArchitectureChild = [pscustomobject]@{Id=54321; StartTime=$created.AddMilliseconds(50); Path=$null}
    $crossArchitectureCim = [pscustomobject]@{ParentProcessId=12345; CommandLine=$fullyQuotedLine; ExecutablePath='C:\Python311\python.exe'}
    Assert-True (Test-LaunchedServer $record $crossArchitectureChild $crossArchitectureCim) '32-bit PowerShell must identify 64-bit Python through CIM when Process.Path is null.'
    $crossRecord = New-ServerRecord $crossArchitectureChild 7875 $record $crossArchitectureCim
    Assert-True ($crossRecord.executable -eq 'C:\Python311\python.exe') 'Record must retain actual executable when Process.Path is unavailable.'
    Assert-True (Test-ServerIdentity $crossRecord $crossArchitectureChild $fullyQuotedLine $crossArchitectureCim) 'Cross-architecture record must remain verifiable for stop.'
    $actualRecord = New-ServerRecord $child 7875 $record
    Assert-True ($actualRecord.pid -eq 54321 -and $actualRecord.launcher.pid -eq 12345) 'The record must store actual server PID and separate launcher.'
    Assert-True (Test-ServerIdentity $actualRecord $child $line) 'Recorded base-Python child should remain verifiable for stop.'
    $childCim.ParentProcessId = 999
    Assert-True (-not (Test-LaunchedServer $record $child $childCim)) 'A foreign listener cannot be adopted.'
    $childCim.ParentProcessId = 12345
    $child.StartTime = $created.AddMilliseconds(-50)
    Assert-True (-not (Test-LaunchedServer $record $child $childCim)) 'An older process cannot be the new launcher child.'
    $record.start_ticks = [string]($created.Ticks + 1)
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess $line)) 'Reused PID must be rejected by creation time.'
    $record.start_ticks = [string]$created.Ticks
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess ($line.Replace('--port 7875', '--port 7876')))) 'A different command-line port must be rejected.'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess 'python other-project\gui.py --port 7875')) 'A foreign script must be rejected.'
    $record.root = $fixture + '-other'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess $line)) 'A foreign root must be rejected.'
    $record.root = $fixture
    $fakeProcess.Path = 'C:\foreign\python.exe'
    Assert-True (-not (Test-ServerIdentity $record $fakeProcess $line)) 'A different executable must be rejected.'
    Assert-True (-not (Test-ServerIdentity $null $fakeProcess $line)) 'An absent record cannot own a process.'
    function Get-PortListener { return [pscustomobject]@{OwningProcess=999} }
    function Test-Health { return $true }
    Assert-True (-not (Test-OwnedHealth $record $fakeProcess)) 'Healthy HTTP from a foreign listener must not count as owned.'
    function Get-PortListener { return [pscustomobject]@{OwningProcess=12345} }
    Assert-True (Test-OwnedHealth $record $fakeProcess) 'Owned listener plus successful HTTP health should pass.'
    foreach ($action in @('setup','models','doctor','run','stop','status','test','smoke','smoke-extra','benchmark','benchmark-help')) {
        $plan = (& $scriptPath -Command $action -ProjectRoot $fixture -DryRun 6>&1 | Out-String -Width 32767)
        Assert-True ($plan -match '\[dry-run\]') "$action has no dry-run plan."
    }
    $plan = (& $scriptPath configure -ProjectRoot $fixture -Checkpoints 'D:\Models with spaces\checkpoints' -Cloaks 'D:\Data with spaces\mix_full' -DryRun 6>&1 | Out-String -Width 32767)
    Assert-True ($plan.Contains('"D:\Models with spaces\checkpoints"')) 'Configure must retain a spaced path as one argument.'
    $plan = (& $scriptPath import-models -ProjectRoot $fixture -Checkpoints 'D:\Models with spaces\checkpoints.zip' -ModelSet diffswap -DryRun 6>&1 | Out-String -Width 32767)
    Assert-True ($plan.Contains('"D:\Models with spaces\checkpoints.zip"') -and $plan.Contains('"--models" "diffswap"')) 'Optional model import must forward archive path and model selection.'
    $plan = (& $scriptPath benchmark -ProjectRoot $fixture -Port 7876 -Sources 1 -OutputDir 'results\trial with spaces' -DryRun 6>&1 | Out-String -Width 32767)
    Assert-True ($plan.Contains('"--resume"') -and $plan.Contains('"--max-sources" "1"') -and $plan.Contains('"--port" "7876"') -and $plan.Contains('trial with spaces')) 'Benchmark plan must forward resume/source/output/port arguments.'
    $rejected = $false
    try { & $scriptPath run -ProjectRoot $fixture -Port 1 -DryRun } catch { $rejected = $true }
    Assert-True $rejected 'Invalid port must be rejected before execution.'
    Assert-True (@(Get-ChildItem -LiteralPath $fixture -Force).Count -eq 0) 'Dry-run must not create files.'
    New-Item -ItemType Directory -Path (Join-Path $fixture '.venv') | Out-Null
    $versionConfig = Join-Path $fixture '.venv\pyvenv.cfg'
    Set-Content -LiteralPath $versionConfig -Value 'version = 3.12.9'
    $versionRejected = $false
    try { Install-LocalRuntime } catch { $versionRejected = $_.Exception.Message -match 'different Python version' }
    Assert-True $versionRejected 'Setup must reject an incompatible existing venv before replacing its interpreter.'
    Assert-True ((Get-Content -LiteralPath $versionConfig -Raw).Trim() -eq 'version = 3.12.9') 'Rejected setup must preserve the existing venv configuration.'
    Assert-True (-not (Test-Path -LiteralPath (Join-Path $fixture '.runtime'))) 'Rejected setup must not provision a new runtime.'
    Write-Host 'PASS: PowerShell parsing; spaced/quoted arguments; PID/root/executable/command-line ownership; actual venv child listener identity; HTTP listener ownership; all dry-run actions; benchmark forwarding; port validation; no dry-run writes; incompatible Python setup preserves existing environment.'
}
finally {
    # Delete only the exact fresh temporary fixture created by this test.
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    $temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ($resolvedFixture.StartsWith($temporaryRoot, [StringComparison]::OrdinalIgnoreCase) -and (Split-Path -Leaf $resolvedFixture).StartsWith('FaceShield manage test ')) {
        Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
    }
}
