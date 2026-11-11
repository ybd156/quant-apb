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

# Proxy handling: do NOT pre-judge. Try as configured; only bypass on failure.
#
# Why not probe first: measured on a real machine, a proxy at 127.0.0.1:10808
# answered an Invoke-WebRequest through it, but git and curl both failed
# through the same endpoint (protocol mismatch: it speaks SOCKS, not HTTP).
# Any pre-flight test can be right for one client and wrong for another, so
# the only trustworthy signal is "did the real operation succeed".
#
# Machines without a proxy are unaffected: the fallback simply never fires.
function Clear-ProxyEnv {
    $names = @('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')
    $had = @()
    foreach ($n in $names) {
        $v = [Environment]::GetEnvironmentVariable($n)
        if ($v) { $had += "$n=$v"; [Environment]::SetEnvironmentVariable($n, $null) }
    }
    return $had
}

function Restore-ProxyEnv($saved) {
    if (-not $saved) { return }
    foreach ($item in $saved) {
        $i = $item.IndexOf('=')
        if ($i -gt 0) {
            [Environment]::SetEnvironmentVariable($item.Substring(0, $i),
                                                  $item.Substring($i + 1))
        }
    }
}

# Public downloader: try as configured, and on failure bypass the proxy once.
# Call sites keep using Get-Remote -- the fallback is invisible to them.
function Get-Remote($urls, $out, $what, $minBytes = 1000) {
    try {
        Get-RemoteCore $urls $out $what $minBytes
        return
    } catch {
        $saved = Clear-ProxyEnv
        if (-not $saved) { throw }
        Warn2 "$what failed with the configured proxy -- retrying without it"
        try {
            Get-RemoteCore $urls $out $what $minBytes
            Ok "$what succeeded after bypassing the proxy"
        } catch {
            Restore-ProxyEnv $saved
            throw
        }
    }
}

$script:LogPath = Join-Path $Root 'logs\ootstrap.log'

function Log-Lines($text) {
    try {
        $dir = Split-Path -Parent $script:LogPath
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -Path $script:LogPath -Value $text -Encoding UTF8
    } catch { }
}

# Run python and CAPTURE the output so a failure can show the real reason.
function Invoke-PyLogged {
    param([string[]]$PyArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $PyExe @PyArgs 2>&1 | Out-String
        $rc = $LASTEXITCODE
        Log-Lines ("`$ python " + ($PyArgs -join ' ') + "  -> rc=$rc")
        Log-Lines $out
        return @{ rc = $rc; out = $out }
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Step($m) { Write-Host "  [..] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn2($m){ Write-Host "  [!!] $m" -ForegroundColor Yellow }

function Get-RemoteCore($urls, $out, $what, $minBytes = 1000) {
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
    # get-pip downloads pip/setuptools/wheel from an index. Default to a
    # domestic mirror (overridable via APB_PIP_INDEX) because the official
    # index is unreachable from many networks -- but do not assume it works:
    # try, then retry with pypi.org, then retry with the proxy bypassed.
    $idx = $env:APB_PIP_INDEX
    if (-not $idx) { $idx = 'https://mirrors.aliyun.com/pypi/simple' }
    $r = Invoke-PyLogged -PyArgs @($getpip, '--no-warn-script-location',
                                   '--disable-pip-version-check', '--index-url', $idx)
    if ($r.rc -ne 0) {
        Warn2 "pip install via $idx failed, retrying with pypi.org ..."
        $r = Invoke-PyLogged -PyArgs @($getpip, '--no-warn-script-location',
                                       '--disable-pip-version-check')
    }
    if ($r.rc -ne 0) {
        $savedProxy = Clear-ProxyEnv
        if ($savedProxy) {
            Warn2 "still failing -- retrying with the configured proxy bypassed"
            $r = Invoke-PyLogged -PyArgs @($getpip, '--no-warn-script-location',
                                           '--disable-pip-version-check',
                                           '--index-url', $idx)
            if ($r.rc -ne 0) { Restore-ProxyEnv $savedProxy }
        }
    }
    if ($r.rc -ne 0) {
        # get-pip on Python 3.12 can fail with "No module named 'distutils'"
        # (3.12 removed distutils and older pip builds still import it --
        # measured against the aliyun mirror). pip itself is pure Python,
        # so bundled pip/setuptools/wheel wheels can simply be unpacked.
        $wheelDir0 = Join-Path $Root 'wheels'
        $pipWhls = @()
        if (Test-Path $wheelDir0) {
            $pipWhls = Get-ChildItem -Path $wheelDir0 -Filter '*.whl' -ErrorAction SilentlyContinue |
                       Where-Object { $_.Name -match '^(pip|setuptools|wheel)-' }
        }
        if ($pipWhls.Count -gt 0) {
            Step "get-pip failed -> unpacking bundled pip wheels ($($pipWhls.Count))"
            $site0 = Join-Path $PyDir 'Lib\site-packages'
            New-Item -ItemType Directory -Path $site0 -Force | Out-Null
            Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
            foreach ($w in $pipWhls) {
                try {
                    [System.IO.Compression.ZipFile]::ExtractToDirectory($w.FullName, $site0)
                } catch {
                    Warn2 ("unpack failed: " + $w.Name + " -> " + $_.Exception.Message)
                }
            }
            $chk = Invoke-PyLogged -PyArgs @('-m','pip','--version')
            if ($chk.rc -eq 0) {
                $r = $chk
                Ok "pip installed from bundled wheels (offline)"
            }
        }
    }
    if ($r.rc -ne 0) {
        # Do NOT just throw: show the last lines of the real output.
        Warn2 "pip installation failed. Last output:"
        $tail = ($r.out -split "`r?`n") | Where-Object { $_.Trim() } | Select-Object -Last 12
        foreach ($line in $tail) { Write-Host ("        " + $line) -ForegroundColor DarkGray }
        Warn2 "Full log: $script:LogPath"
        $script:NoPip = $true
    } else {
        Ok "pip installed"
    }
}

# ---------------------------------------------------------------- 4. numpy / pandas
# If pip could not be installed, wheels can still be unpacked by hand:
# a .whl is a zip, and the .pyd/.dll files inside land exactly where
# site-packages expects them. That makes the kit usable on a network
# where pip cannot reach any index at all.
if ($script:NoPip) {
    $wheelDir = Join-Path $Root 'wheels'
    $whls = @()
    if (Test-Path $wheelDir) {
        $whls = Get-ChildItem -Path $wheelDir -Filter '*.whl' -ErrorAction SilentlyContinue
    }
    if ($whls.Count -gt 0) {
        Step "pip unavailable -> unpacking $($whls.Count) wheels directly"
        $site = Join-Path $PyDir 'Lib\site-packages'
        New-Item -ItemType Directory -Path $site -Force | Out-Null
        $unpacked = 0
        Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue
        foreach ($w in $whls) {
            try {
                # Expand-Archive REFUSES files whose extension is not .zip
                # ("only .zip archives are supported" -- measured). A .whl IS
                # a zip, so use ZipFile directly.
                [System.IO.Compression.ZipFile]::ExtractToDirectory($w.FullName, $site)
                $unpacked++
            } catch {
                Warn2 ("unpack failed: " + $w.Name + " -> " + $_.Exception.Message)
            }
        }
        Ok "unpacked $unpacked / $($whls.Count) wheels into site-packages"
    } else {
        Warn2 "no wheels to unpack and no pip -- cannot install numpy/pandas"
    }
}

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
