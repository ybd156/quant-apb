# APB Training Kit - portable Python runtime bootstrap
#
# Goal: run on a machine with NO Python installed. Everything lives under
#       runtime\ inside the kit folder. No admin rights, no registry, no PATH.
#
# Steps:
#   1. download Python embeddable package (~11MB, official no-install build)
#   2. extract to runtime\python
#   3. enable "import site" in python312._pth (embed builds disable it, pip needs it)
#   4. download get-pip.py and install pip
#   5. pip install numpy / pandas into runtime\python\Lib\site-packages
#
# NOTE: this file is intentionally ASCII-only. Windows PowerShell 5.1 reads .ps1
#       as ANSI unless the file has a UTF-8 BOM, which mangles non-ASCII text
#       and produces bogus "missing }" parse errors.
#
# Usage (called automatically by train.bat):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\bootstrap.ps1

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root    = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Runtime = Join-Path $Root 'runtime'
$PyDir   = Join-Path $Runtime 'python'
$PyExe   = Join-Path $PyDir 'python.exe'

$PyVer = '3.12.7'

# Native commands (python.exe) write to stderr even for benign "no module"
# messages. With $ErrorActionPreference='Stop' that becomes a terminating
# exception. Always call python through this wrapper.
function Invoke-Py {
    param([string[]]$PyArgs, [switch]$Quiet)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        if ($Quiet) {
            & $PyExe @PyArgs 2>$null 1>$null
        } else {
            & $PyExe @PyArgs 2>&1 | Select-Object -Last 3
        }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Step($m) { Write-Host "  [..] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn2($m){ Write-Host "  [!!] $m" -ForegroundColor Yellow }

function Get-Remote($urls, $out, $what, $minBytes = 1000) {
    # $urls may be a single string or an array (multi-mirror fallback).
    # python.org is often 1-2 MB/s from China; huaweicloud hits 3.5+ MB/s,
    # so the China mirrors are tried first.
    if ($urls -is [string]) { $urls = @($urls) }
    Step "Downloading $what ..."
    $tmp = "$out.part"
    $done = $false
    foreach ($u in $urls) {
        if ($done) { break }
        $host_ = ([Uri]$u).Host
        for ($i = 1; $i -le 2; $i++) {
            if (Test-Path $tmp) { Remove-Item $tmp -Force -ErrorAction SilentlyContinue }
            $sw = [Diagnostics.Stopwatch]::StartNew()
            try {
                Invoke-WebRequest -Uri $u -OutFile $tmp -UseBasicParsing -TimeoutSec 180
                $sz = (Get-Item $tmp).Length
                if ($sz -gt $minBytes) {
                    $sw.Stop()
                    $mb = [math]::Round($sz / 1MB, 1)
                    $sp = [math]::Round($sz / 1MB / [math]::Max($sw.Elapsed.TotalSeconds, 0.1), 1)
                    Ok "$what from $host_ ($mb MB, $sp MB/s)"
                    $done = $true
                    break
                }
                Warn2 "$host_ returned only $sz bytes"
            } catch {
                $sw.Stop()
                Warn2 ("{0} attempt {1} failed: {2}" -f $host_, $i, $_.Exception.Message)
                Start-Sleep -Seconds $i
            }
        }
    }
    if (-not $done) { throw "Failed to download $what from all mirrors. Check your network." }
    Move-Item -Force $tmp $out
}

# ---------------------------------------------------------------- already ready?
if (Test-Path $PyExe) {
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $v = & $PyExe -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>$null
    & $PyExe -c "import numpy,pandas" 2>$null 1>$null
    $rc0 = $LASTEXITCODE
    $ErrorActionPreference = $prevEap
    if ($rc0 -eq 0) {
        Ok "Portable runtime ready (Python $v, numpy/pandas present)"
        Write-Output $PyExe
        exit 0
    }
    Warn2 "Runtime present but dependencies missing, will install numpy/pandas"
}

New-Item -ItemType Directory -Force -Path $Runtime | Out-Null

# ---------------------------------------------------------------- 1. python
if (-not (Test-Path $PyExe)) {
    if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
    New-Item -ItemType Directory -Force -Path $PyDir | Out-Null
    $zip = Join-Path $Runtime "python-$PyVer-embed-amd64.zip"
    if (-not (Test-Path $zip)) {
        $f = "python-$PyVer-embed-amd64.zip"
        Get-Remote @(
            "https://mirrors.huaweicloud.com/python/$PyVer/$f",
            "https://registry.npmmirror.com/-/binary/python/$PyVer/$f",
            "https://mirrors.aliyun.com/python-release/windows/$f",
            "https://www.python.org/ftp/python/$PyVer/$f"
        ) $zip "Python $PyVer embeddable" 5000000
    }
    Step "Extracting to runtime\python ..."
    Expand-Archive -Path $zip -DestinationPath $PyDir -Force
    Ok "Extracted"
}

# ---------------------------------------------------------------- 2. enable site
$pth = Get-ChildItem -Path $PyDir -Filter 'python*._pth' | Select-Object -First 1
if ($pth) {
    $lines = Get-Content $pth.FullName
    $need = $false
    foreach ($l in $lines) { if ($l -match '^\s*#\s*import\s+site') { $need = $true } }
    if ($need) {
        $new = foreach ($l in $lines) {
            if ($l -match '^\s*#\s*import\s+site') { 'import site' } else { $l }
        }
        Set-Content -Path $pth.FullName -Value $new -Encoding ASCII
        Ok "Enabled 'import site' (embed builds disable it; pip needs it)"
    }
}

# ---------------------------------------------------------------- 3. pip
$pipRc = Invoke-Py -PyArgs @('-m','pip','--version') -Quiet
if ($pipRc -ne 0) {
    $getpip = Join-Path $Runtime 'get-pip.py'
    if (-not (Test-Path $getpip)) {
        Get-Remote @(
            'https://mirrors.aliyun.com/pypi/get-pip.py',
            'https://bootstrap.pypa.io/get-pip.py'
        ) $getpip 'pip installer' 100000
    }
    Step "Installing pip ..."
    $rc = Invoke-Py -PyArgs @($getpip,'--no-warn-script-location','--disable-pip-version-check')
    if ($rc -ne 0) { throw "pip installation failed" }
    Ok "pip installed"
}

# ---------------------------------------------------------------- 4. numpy / pandas
$depRc = Invoke-Py -PyArgs @('-c','import numpy,pandas') -Quiet
if ($depRc -ne 0) {
    # Bundled wheels first: installing from local files takes a few seconds,
    # while downloading numpy+pandas over a slow link can take many minutes.
    $wheelDir = Join-Path $Root 'wheels'
    $hasWheels = (Test-Path $wheelDir) -and
                 ((Get-ChildItem -Path $wheelDir -Filter '*.whl' -ErrorAction SilentlyContinue).Count -gt 0)

    $okDep = $false
    if ($hasWheels) {
        $n = (Get-ChildItem -Path $wheelDir -Filter '*.whl').Count
        Step "Installing numpy / pandas from bundled wheels ($n files, no download)"
        $a = @('-m','pip','install','--quiet','--disable-pip-version-check',
               '--no-warn-script-location','--no-index',
               '--find-links', $wheelDir, 'numpy', 'pandas')
        $rc = Invoke-Py -PyArgs $a
        if ($rc -eq 0) {
            $chk = Invoke-Py -PyArgs @('-c','import numpy,pandas') -Quiet
            if ($chk -eq 0) { $okDep = $true }
        }
        if (-not $okDep) {
            Warn2 "Bundled wheels did not work, falling back to online install"
        }
    }

    if (-not $okDep) {
        Step "Installing numpy / pandas from the internet (first run only) ..."
        $mirrors = @(
            'https://pypi.tuna.tsinghua.edu.cn/simple',
            'https://mirrors.aliyun.com/pypi/simple/',
            ''
        )
        foreach ($m in $mirrors) {
            $a = @('-m','pip','install','--quiet','--disable-pip-version-check',
                   '--no-warn-script-location','numpy','pandas')
            if ($m) {
                $a += @('-i', $m)
                $a += @('--trusted-host', ([Uri]$m).Host)
                Step "Trying mirror: $m"
            } else {
                Step "Trying official pypi.org"
            }
            $rc = Invoke-Py -PyArgs $a
            if ($rc -eq 0) {
                $chk = Invoke-Py -PyArgs @('-c','import numpy,pandas') -Quiet
                if ($chk -eq 0) { $okDep = $true; break }
            }
        }
    }
    if (-not $okDep) {
        throw "numpy/pandas install failed. Try manually: runtime\python\python.exe -m pip install numpy pandas"
    }
    Ok "numpy / pandas installed"
}

# ---------------------------------------------------------------- done
$prevEap2 = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
$v = & $PyExe -c "import sys,numpy,pandas;print('%d.%d.%d'%sys.version_info[:3], numpy.__version__, pandas.__version__)" 2>$null
$ErrorActionPreference = $prevEap2
Ok "Portable runtime ready: Python $v"
Write-Output $PyExe
