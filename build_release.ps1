# LiveTranslate - Portable release builder
# Produces a self-contained zip that runs without a system Python install.
# First launch uses a bundled uv to fetch Python 3.12 + GPU-aware dependencies.

param([string]$Version = "")

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[BUILD] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }

$UvUrl  = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
$OutDir = Join-Path $ProjectDir "release"
$Stage  = Join-Path $OutDir "LiveTranslate"

$Sha   = (& git rev-parse --short HEAD).Trim()
$Stamp = Get-Date -Format "yyyyMMdd"
$Tag   = if ($Version) { $Version } else { "$Stamp-$Sha" }
$ZipPath = Join-Path $OutDir "LiveTranslate-portable-$Tag.zip"

# Files only needed for the git-clone workflow; the portable zip ships its own launcher.
$DropList = @("install.bat", "install.ps1", "update.bat", "start.bat",
              "build_release.ps1", "CLAUDE.md", "test_audio.py", ".gitignore", "screenshot")

# ── 1. Clean staging ──
Write-Step "Preparing staging directory..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

# ── 2. Export tracked source via git archive ──
Write-Step "Exporting source (git archive HEAD)..."
$Tar = Join-Path $OutDir "src.tar"
& git archive --format=tar -o $Tar HEAD
if ($LASTEXITCODE -ne 0) { throw "git archive failed" }
& tar -x -f $Tar -C $Stage
Remove-Item $Tar
foreach ($d in $DropList) {
    $p = Join-Path $Stage $d
    if (Test-Path $p) { Remove-Item -Recurse -Force $p }
}
Write-Ok "Source exported"

# ── 3. Download and bundle uv ──
Write-Step "Downloading uv..."
$UvZip = Join-Path $OutDir "uv.zip"
Invoke-WebRequest -Uri $UvUrl -OutFile $UvZip
$ToolsDir = Join-Path $Stage "tools"
New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
Expand-Archive -Path $UvZip -DestinationPath $ToolsDir -Force
Remove-Item $UvZip
$UvExe = Join-Path $ToolsDir "uv.exe"
if (-not (Test-Path $UvExe)) { throw "uv.exe not found after extract" }
Write-Ok ("Bundled " + (& $UvExe --version))

# ── 4. Write portable launcher + bootstrap ──
Write-Step "Writing launcher..."
$StartBat = @'
@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run: setting up environment. This downloads Python and dependencies and may take several minutes...
    powershell -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
    if errorlevel 1 (
        echo.
        echo [ERROR] Setup failed. See messages above.
        pause
        exit /b 1
    )
)
echo Starting LiveTranslate...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
'@
Set-Content -Path (Join-Path $Stage "start.bat") -Value $StartBat -Encoding ASCII

$Bootstrap = @'
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Uv = Join-Path $Root "tools\uv.exe"
$env:UV_LINK_MODE = "copy"
$CrispAsrVersion = "v0.7.2"

function Enable-SystemProxy {
    # uv (Python download) and pip honor *_PROXY env vars but not the Windows
    # registry system proxy; bridge it here. An already-set env proxy wins.
    if ($env:HTTPS_PROXY -or $env:HTTP_PROXY) {
        Write-Host "Using proxy from environment" -ForegroundColor Gray
        return
    }
    try {
        $reg = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings"
        $s = Get-ItemProperty -Path $reg -ErrorAction Stop
        if ($s.ProxyEnable -ne 1 -or -not $s.ProxyServer) { return }
        $server = [string]$s.ProxyServer
        $http = $null; $https = $null
        if ($server -like "*=*") {
            foreach ($part in ($server -split ';')) {
                $kv = $part -split '=', 2
                if ($kv.Count -eq 2 -and $kv[0] -eq 'http')  { $http  = $kv[1] }
                if ($kv.Count -eq 2 -and $kv[0] -eq 'https') { $https = $kv[1] }
            }
        } else {
            $http = $server; $https = $server
        }
        if (-not $http)  { $http  = $https }
        if (-not $https) { $https = $http }
        if (-not $http) { return }
        if ($http  -notmatch '^\w+://') { $http  = "http://$http" }
        if ($https -notmatch '^\w+://') { $https = "http://$https" }
        $env:HTTP_PROXY  = $http
        $env:HTTPS_PROXY = $https
        $env:ALL_PROXY   = $https
        Write-Host "Detected Windows system proxy: $https (applied to uv/pip)" -ForegroundColor Green
    } catch {}
}
Enable-SystemProxy

function Install-CrispAsrNativeRuntime {
    param(
        [string]$PythonExe,
        [bool]$UseCuda
    )

    Write-Host "Installing CrispASR native runtime..." -ForegroundColor Cyan
    try {
        $target = & $PythonExe -c "import crispasr, pathlib; print(pathlib.Path(crispasr.__file__).resolve().parent)"
        if ($LASTEXITCODE -ne 0 -or -not $target) {
            throw "Could not locate the installed crispasr package"
        }
        $target = $target.Trim()
        if (-not (Test-Path $target)) {
            throw "Python package 'crispasr' is not installed"
        }

        $variant = if ($UseCuda) { "cuda" } else { "cpu" }
        $asset = if ($UseCuda) {
            "libcrispasr-windows-x86_64-cuda.tar.gz"
        } else {
            "libcrispasr-windows-x86_64.tar.gz"
        }
        $url = "https://github.com/CrispStrobe/CrispASR/releases/download/$CrispAsrVersion/$asset"
        $tmpDir = Join-Path $env:TEMP "livetranslate-crispasr-$variant"
        $archive = Join-Path $tmpDir $asset
        if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

        Write-Host "Downloading $asset" -ForegroundColor Gray
        Invoke-WebRequest -Uri $url -OutFile $archive
        & tar -xzf $archive -C $tmpDir
        if ($LASTEXITCODE -ne 0) { throw "Failed to extract $asset" }

        $runtimeRoot = Get-ChildItem -Path $tmpDir -Directory | Select-Object -First 1
        if (-not $runtimeRoot) { throw "Extracted CrispASR runtime directory not found" }
        $bin = Join-Path $runtimeRoot.FullName "bin"
        if (-not (Test-Path (Join-Path $bin "crispasr.dll"))) {
            throw "crispasr.dll not found in $asset"
        }

        Copy-Item -Path (Join-Path $bin "*.dll") -Destination $target -Force
        Write-Host "CrispASR native runtime installed ($variant)" -ForegroundColor Green
    } catch {
        if ($UseCuda) {
            Write-Host "CUDA CrispASR runtime failed: $($_.Exception.Message)" -ForegroundColor Yellow
            Install-CrispAsrNativeRuntime -PythonExe $PythonExe -UseCuda $false
        } else {
            Write-Host "CrispASR native runtime installation failed: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "CrispASR will not run until libcrispasr/crispasr.dll is installed" -ForegroundColor Yellow
        }
    }
}

Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
& $Uv venv --python 3.12 --managed-python .venv
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to create venv" -ForegroundColor Red; exit 1 }
$Py = ".venv\Scripts\python.exe"

# Blackwell (sm_120+) needs cu128; older NVIDIA uses cu126; no GPU falls back to CPU
$Index = "https://download.pytorch.org/whl/cpu"
try {
    $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $cc) {
        $cap = [double]($cc.Trim() -split "`n")[0]
        if ($cap -ge 12.0) { $Index = "https://download.pytorch.org/whl/cu128" }
        else { $Index = "https://download.pytorch.org/whl/cu126" }
        Write-Host "NVIDIA GPU detected (compute $cap), using $Index" -ForegroundColor Green
    }
} catch {}
if ($Index -like "*cpu*") { Write-Host "No NVIDIA GPU detected, installing CPU-only PyTorch" -ForegroundColor Yellow }

Write-Host "Installing PyTorch (this may take a while)..." -ForegroundColor Cyan
& $Uv pip install --python $Py torch torchaudio --index-url $Index
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch install failed" -ForegroundColor Red; exit 1 }

Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $Uv sync --python $Py --locked --inexact --no-install-package torch --no-install-package torchaudio
if ($LASTEXITCODE -ne 0) { Write-Host "Dependency install failed" -ForegroundColor Red; exit 1 }

if (Test-Path (Join-Path $Root "repair_torch_metadata.ps1")) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "repair_torch_metadata.ps1") -PythonExe $Py
}

Install-CrispAsrNativeRuntime -PythonExe $Py -UseCuda ($Index -notlike "*cpu*")

& $Uv pip install --python $Py "sherpa-onnx>=1.13.3" "sherpa-onnx-bin>=1.13.3"
if ($LASTEXITCODE -ne 0) {
    Write-Host "sherpa-onnx install failed; sherpa-onnx ASR will be unavailable until installed manually" -ForegroundColor Yellow
}

& $Uv pip install --python $Py funasr --no-deps

Write-Host "Setup complete." -ForegroundColor Green
'@
Set-Content -Path (Join-Path $Stage "bootstrap.ps1") -Value $Bootstrap -Encoding ASCII
Write-Ok "Launcher written"

# ── 5. Zip ──
Write-Step "Creating archive..."
Compress-Archive -Path $Stage -DestinationPath $ZipPath -Force
$sizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Ok "Created $ZipPath ($sizeMb MB)"
Write-Host "`nDone. Distribute: $ZipPath" -ForegroundColor Green
