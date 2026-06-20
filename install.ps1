# LiveTranslate - One-click installer
# Usage: Double-click install.bat (or run: powershell -ExecutionPolicy Bypass -File install.ps1)

param(
    [ValidateSet("cpu", "cuda11", "cuda12")]
    [string]$SherpaOnnxRuntime = "cpu",

    [switch]$DownloadFireRedVAD,

    [ValidateSet("ms", "hf")]
    [string]$FireRedVADHub = "ms",

    [switch]$InstallParakeetCpp,

    [ValidateSet("cpu", "cuda", "vulkan")]
    [string]$ParakeetCppBackend = "cpu",

    [string]$ParakeetCppVersion = "v0.3.2",

    [switch]$DownloadParakeetCppModel,

    [string]$ParakeetCppModel = "tdt_ctc-110m-q4_k"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

function Write-Step { param($msg) Write-Host "`n[$((Get-Date).ToString('HH:mm:ss'))] $msg" -ForegroundColor Cyan }
function Write-Ok   { param($msg) Write-Host "  OK: $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "  WARN: $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  ERROR: $msg" -ForegroundColor Red }

$CrispAsrVersion = "v0.7.2"
$Uv = "uv"

function Install-CrispAsrNativeRuntime {
    param(
        [string]$PythonExe,
        [bool]$UseCuda
    )

    Write-Step "Installing CrispASR native runtime..."
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

        Write-Host "  Downloading $asset" -ForegroundColor Gray
        Invoke-WebRequest -Uri $url -OutFile $archive
        & tar -xzf $archive -C $tmpDir
        if ($LASTEXITCODE -ne 0) { throw "Failed to extract $asset" }

        $root = Get-ChildItem -Path $tmpDir -Directory | Select-Object -First 1
        if (-not $root) { throw "Extracted CrispASR runtime directory not found" }
        $bin = Join-Path $root.FullName "bin"
        if (-not (Test-Path (Join-Path $bin "crispasr.dll"))) {
            throw "crispasr.dll not found in $asset"
        }

        Copy-Item -Path (Join-Path $bin "*.dll") -Destination $target -Force
        Write-Ok "CrispASR native runtime installed ($variant)"
    } catch {
        if ($UseCuda) {
            Write-Warn "CUDA CrispASR runtime failed: $($_.Exception.Message)"
            Install-CrispAsrNativeRuntime -PythonExe $PythonExe -UseCuda $false
        } else {
            Write-Warn "CrispASR native runtime installation failed: $($_.Exception.Message)"
            Write-Warn "CrispASR will not run until libcrispasr/crispasr.dll is installed"
        }
    }
}

function Install-SherpaOnnxRuntime {
    param(
        [string]$PythonExe,
        [string]$Runtime
    )

    Write-Step "Installing sherpa-onnx runtime ($Runtime)..."
    try {
        if ($Runtime -eq "cpu") {
            & $Uv pip install --python $PythonExe "sherpa-onnx>=1.13.3" "sherpa-onnx-bin>=1.13.3"
        } else {
            & $Uv pip uninstall --python $PythonExe sherpa-onnx sherpa-onnx-bin sherpa-onnx-core
            if ($Runtime -eq "cuda11") {
                & $Uv pip install --python $PythonExe --verbose sherpa-onnx=="1.13.3+cuda" --no-index -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
            } elseif ($Runtime -eq "cuda12") {
                & $Uv pip install --python $PythonExe --verbose sherpa-onnx=="1.13.3+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
            }
        }
        if ($LASTEXITCODE -ne 0) { throw "sherpa-onnx install command failed" }
        Write-Ok "sherpa-onnx runtime installed ($Runtime)"
    } catch {
        Write-Warn "sherpa-onnx runtime installation failed: $($_.Exception.Message)"
        Write-Warn "sherpa-onnx ASR will not run until the Python package is installed"
    }
}

function Test-FireRedVadPackage {
    param([string]$PythonExe)

    Write-Step "Checking FireRedVAD Python package..."
    try {
        $version = & $PythonExe -c "import importlib.metadata as m; import fireredvad; print(m.version('fireredvad'))" 2>&1
        if ($LASTEXITCODE -ne 0) { throw $version }
        Write-Ok "fireredvad $($version.Trim())"
    } catch {
        Write-Warn "FireRedVAD package check failed: $($_.Exception.Message)"
        Write-Warn "FireRedVAD VAD mode will be unavailable until fireredvad is installed"
    }
}

function Download-FireRedVADModel {
    param(
        [string]$PythonExe,
        [string]$Hub
    )

    Write-Step "Downloading FireRedVAD Stream-VAD model..."
    $target = Join-Path $ProjectDir "models\FireRedVAD"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    try {
        if ($Hub -eq "ms") {
            Write-Host "  Source: ModelScope xukaituo/FireRedVAD" -ForegroundColor Gray
            & $PythonExe -c "from modelscope import snapshot_download; snapshot_download(model_id='xukaituo/FireRedVAD', local_dir=r'models/FireRedVAD')"
        } else {
            Write-Host "  Source: HuggingFace FireRedTeam/FireRedVAD" -ForegroundColor Gray
            & $PythonExe -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='FireRedTeam/FireRedVAD', local_dir=r'models/FireRedVAD')"
        }
        if ($LASTEXITCODE -ne 0) { throw "FireRedVAD model download command failed" }

        $cmvn = Join-Path $target "Stream-VAD\cmvn.ark"
        $model = Join-Path $target "Stream-VAD\model.pth.tar"
        if (-not (Test-Path $cmvn) -or -not (Test-Path $model)) {
            throw "Downloaded model is missing Stream-VAD\cmvn.ark or Stream-VAD\model.pth.tar"
        }
        Write-Ok "FireRedVAD model ready: models\FireRedVAD\Stream-VAD"
    } catch {
        Write-Warn "FireRedVAD model download failed: $($_.Exception.Message)"
        Write-Warn "You can download it later and place it under models\FireRedVAD"
    }
}

function Install-ParakeetCppRuntime {
    param(
        [string]$PythonExe,
        [string]$Backend,
        [string]$Version
    )

    Write-Step "Installing parakeet.cpp runtime ($Backend, $Version)..."
    $versionNoV = $Version.TrimStart("v")
    $asset = "parakeet-$Version-lib-win-$Backend-x64.zip"
    $url = "https://github.com/mudler/parakeet.cpp/releases/download/$Version/$asset"
    $target = Join-Path $ProjectDir "models\parakeet.cpp\runtime\$Version\$Backend"
    $tmpDir = Join-Path $env:TEMP "livetranslate-parakeet-$Backend"
    $archive = Join-Path $tmpDir $asset
    try {
        if (Test-Path $tmpDir) { Remove-Item -Recurse -Force $tmpDir }
        New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
        New-Item -ItemType Directory -Force -Path $target | Out-Null

        Write-Host "  Downloading $asset" -ForegroundColor Gray
        Invoke-WebRequest -Uri $url -OutFile $archive
        Expand-Archive -Path $archive -DestinationPath $tmpDir -Force

        $dll = Get-ChildItem -Path $tmpDir -Recurse -Filter "*.dll" |
            Where-Object { $_.Name -in @("parakeet.dll", "libparakeet.dll", "parakeet_capi.dll", "libparakeet_capi.dll") } |
            Select-Object -First 1
        if (-not $dll) {
            throw "parakeet C API DLL not found in $asset"
        }

        Copy-Item -Path (Join-Path $tmpDir "*") -Destination $target -Recurse -Force

        if ($Backend -eq "cuda") {
            $cudartAsset = "cudart-parakeet-bin-win-cuda-x64.zip"
            $cudartUrl = "https://github.com/mudler/parakeet.cpp/releases/download/$Version/$cudartAsset"
            $cudartArchive = Join-Path $tmpDir $cudartAsset
            Write-Host "  Downloading $cudartAsset" -ForegroundColor Gray
            Invoke-WebRequest -Uri $cudartUrl -OutFile $cudartArchive
            Expand-Archive -Path $cudartArchive -DestinationPath $target -Force
        }

        $test = & $PythonExe -c "import ctypes, pathlib; root=pathlib.Path(r'$target'); names=('parakeet.dll','libparakeet.dll','parakeet_capi.dll','libparakeet_capi.dll'); dll=next((p for n in names for p in root.rglob(n)), None); assert dll, 'DLL not found'; lib=ctypes.CDLL(str(dll)); lib.parakeet_capi_abi_version.restype=ctypes.c_int; print(lib.parakeet_capi_abi_version())" 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Runtime smoke test failed: $test"
        }
        Write-Ok "parakeet.cpp runtime ready: models\parakeet.cpp\runtime\$Version\$Backend (ABI $($test.Trim()))"
    } catch {
        Write-Warn "parakeet.cpp runtime installation failed: $($_.Exception.Message)"
        Write-Warn "You can manually extract $asset to models\parakeet.cpp\runtime\$Version\$Backend"
    }
}

function Download-ParakeetCppModel {
    param(
        [string]$PythonExe,
        [string]$ModelName
    )

    Write-Step "Downloading parakeet.cpp GGUF model ($ModelName)..."
    $target = Join-Path $ProjectDir "models\parakeet.cpp\models"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $fileName = if ($ModelName.EndsWith(".gguf")) { $ModelName } else { "$ModelName.gguf" }
    try {
        & $PythonExe -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='mudler/parakeet-cpp-gguf', filename=r'$fileName', local_dir=r'models/parakeet.cpp/models')"
        if ($LASTEXITCODE -ne 0) { throw "HuggingFace model download command failed" }
        $modelPath = Join-Path $target $fileName
        if (-not (Test-Path $modelPath)) {
            throw "Downloaded model file not found: $modelPath"
        }
        Write-Ok "parakeet.cpp model ready: models\parakeet.cpp\models\$fileName"
    } catch {
        Write-Warn "parakeet.cpp model download failed: $($_.Exception.Message)"
        Write-Warn "Download a GGUF from mudler/parakeet-cpp-gguf and place it under models\parakeet.cpp\models"
    }
}

function Enable-SystemProxy {
    # uv (Python download) and pip honor *_PROXY env vars but not the Windows
    # registry system proxy; bridge it here. An already-set env proxy wins.
    if ($env:HTTPS_PROXY -or $env:HTTP_PROXY) {
        Write-Host "  Using proxy from environment" -ForegroundColor Gray
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
        Write-Host "  Detected Windows system proxy: $https (applied to uv/pip)" -ForegroundColor Green
    } catch {}
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "   LiveTranslate Installer" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

Enable-SystemProxy

# ── Step 0: Find uv ──
Write-Step "Detecting uv..."
try {
    $uvVersion = & $Uv --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw $uvVersion }
    Write-Ok $uvVersion
} catch {
    Write-Warn "uv not found"
    $hasWinget = $false
    try {
        $null = & winget --version 2>&1
        if ($LASTEXITCODE -eq 0) { $hasWinget = $true }
    } catch {}

    if ($hasWinget) {
        Write-Host ""
        Write-Host "  uv can be installed automatically via winget." -ForegroundColor White
        $answer = Read-Host "  Install uv now? [Y/n]"
        if ($answer -eq "" -or $answer -match "^[Yy]") {
            Write-Step "Installing uv via winget..."
            & winget install Astral.UV --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                Write-Err "winget install failed"
                Read-Host "Press Enter to exit"
                exit 1
            }
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            $uvVersion = & $Uv --version 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Err "uv installed but not found in PATH. Please close this window, reopen, and run install.bat again."
                Read-Host "Press Enter to exit"
                exit 1
            }
            Write-Ok $uvVersion
        } else {
            Write-Err "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and run install.bat again."
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Err "uv is required and winget is not available. Install uv from https://docs.astral.sh/uv/getting-started/installation/"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ── Step 1: Find Python ──
Write-Step "Detecting Python..."

function Find-Python {
    # 3.13+ rejected: no ctranslate2 cp313 wheels (#15), strict SSL breaks torch.hub (#20)
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>&1
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) {
                $exe = $exe.Trim()
                $ver = & $exe --version 2>&1
                Write-Ok "Found $ver ($exe)"
                return $exe
            }
        } catch {}
    }
    # uv-managed Python is project-local and should count as available when it
    # has already been downloaded. Do not download during detection.
    foreach ($v in @("3.12", "3.11", "3.10")) {
        try {
            $exe = & $Uv python find $v --managed-python --no-python-downloads 2>&1
            if ($LASTEXITCODE -eq 0 -and $exe -and (Test-Path $exe.Trim())) {
                $exe = $exe.Trim()
                $ver = & $exe --version 2>&1
                Write-Ok "Found uv-managed $ver ($exe)"
                return $exe
            }
        } catch {}
    }
    # Fall back to plain commands, rejecting unsupported versions.
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 10 -and $minor -le 12) {
                    Write-Ok "Found $ver ($cmd)"
                    return $cmd
                } elseif ($major -eq 3 -and $minor -ge 13) {
                    Write-Warn "$ver is too new (faster-whisper/SSL require 3.10-3.12)"
                } else {
                    Write-Warn "$ver is too old (need 3.10-3.12)"
                }
            }
        } catch {}
    }
    return $null
}

$PythonCmd = Find-Python

if (-not $PythonCmd) {
    Write-Warn "No supported Python (3.10-3.12) found"

    # Try to install via winget
    $hasWinget = $false
    try {
        $null = & winget --version 2>&1
        if ($LASTEXITCODE -eq 0) { $hasWinget = $true }
    } catch {}

    if ($hasWinget) {
        Write-Host ""
        Write-Host "  Python can be installed automatically via winget." -ForegroundColor White
        $answer = Read-Host "  Install Python 3.12 now? [Y/n]"
        if ($answer -eq "" -or $answer -match "^[Yy]") {
            Write-Step "Installing Python 3.12 via winget..."
            & winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
            if ($LASTEXITCODE -ne 0) {
                Write-Err "winget install failed"
                Read-Host "Press Enter to exit"
                exit 1
            }

            # Refresh PATH to pick up newly installed Python
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

            $PythonCmd = Find-Python
            if (-not $PythonCmd) {
                Write-Err "Python installed but not found in PATH. Please close this window, reopen, and run install.bat again."
                Read-Host "Press Enter to exit"
                exit 1
            }
        } else {
            Write-Err "Python 3.10-3.12 is required. Please install from https://www.python.org/downloads/"
            Read-Host "Press Enter to exit"
            exit 1
        }
    } else {
        Write-Err "Python 3.10-3.12 not found and winget is not available."
        Write-Host "  Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "  Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# ── Step 2: Create venv ──
Write-Step "Creating virtual environment..."

# Validate existing venv: even if python.exe is present, the venv could be
# half-built, created with a different Python, or corrupted (see issue #18).
# If it's broken, recreate it; otherwise reuse it.
function Test-VenvHealthy {
    param([string]$VenvPythonExe)
    if (-not (Test-Path $VenvPythonExe)) { return $false }
    try {
        $ver = & $VenvPythonExe --version 2>&1
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($ver -notmatch "Python \d+\.\d+") { return $false }
        return $true
    } catch {
        return $false
    }
}

$VenvPython = ".venv\Scripts\python.exe"
if (Test-Path ".venv") {
    if (Test-VenvHealthy $VenvPython) {
        Write-Ok "Existing venv is healthy, reusing"
    } else {
        Write-Warn "Existing venv is broken or incomplete, recreating..."
        Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
        & $Uv venv --python $PythonCmd .venv
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to create venv"
            Read-Host "Press Enter to exit"
            exit 1
        }
        Write-Ok "Created .venv"
    }
} else {
    & $Uv venv --python $PythonCmd .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create venv"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Ok "Created .venv"
}

$Python = ".venv\Scripts\python.exe"

# ── Step 3: Detect GPU ──
Write-Step "Detecting GPU..."

$HasNvidia = $false
$CudaVer = "cu126"
try {
    $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -eq 0 -and $gpu) {
        $HasNvidia = $true
        Write-Ok "NVIDIA GPU detected: $($gpu.Trim())"

        # Detect compute capability to choose CUDA version
        # Blackwell (sm_120, compute_cap >= 12.0) requires cu128
        $cc = & nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null
        if ($LASTEXITCODE -eq 0 -and $cc) {
            $ccVal = [double]($cc.Trim())
            if ($ccVal -ge 12.0) {
                $CudaVer = "cu128"
                Write-Ok "Blackwell+ architecture (sm_$($cc.Trim() -replace '\.','')) detected, using CUDA 12.8"
            } else {
                Write-Ok "Compute capability $($cc.Trim()), using CUDA 12.6"
            }
        }
    }
} catch {}

if (-not $HasNvidia) {
    Write-Warn "No NVIDIA GPU detected, will install CPU-only PyTorch"
}

# Let user choose
Write-Host ""
if ($HasNvidia) {
    $cudaLabel = if ($CudaVer -eq "cu128") { "CUDA 12.8" } else { "CUDA 12.6" }
    Write-Host "  [1] $cudaLabel (recommended for your NVIDIA GPU)" -ForegroundColor White
    Write-Host "  [2] CPU only" -ForegroundColor White
    $choice = Read-Host "  Select PyTorch version [1]"
    if ($choice -eq "2") { $HasNvidia = $false }
} else {
    Write-Host "  [1] CPU only" -ForegroundColor White
    Write-Host "  [2] CUDA (if you have NVIDIA GPU)" -ForegroundColor White
    $choice = Read-Host "  Select PyTorch version [1]"
    if ($choice -eq "2") { $HasNvidia = $true }
}

# ── Step 4: Install PyTorch ──
Write-Step "Installing PyTorch (this may take a few minutes)..."

if ($HasNvidia) {
    Write-Host "  Using index: $CudaVer" -ForegroundColor Gray
    & $Uv pip install --python $Python torch torchaudio --index-url https://download.pytorch.org/whl/$CudaVer
} else {
    & $Uv pip install --python $Python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

if ($LASTEXITCODE -ne 0) {
    Write-Err "PyTorch installation failed"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "PyTorch installed"

# ── Step 5: Install dependencies ──
Write-Step "Syncing dependencies with uv..."

& $Uv sync --python $Python --locked --inexact --no-install-package torch --no-install-package torchaudio
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to sync dependencies"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "Dependencies synced"

# uv sync intentionally skips torch/torchaudio because the correct wheel index
# depends on GPU support. Clean up stale torch metadata if a previous sync/install
# left duplicate dist-info directories behind.
if (Test-Path ".\repair_torch_metadata.ps1") {
    & powershell -NoProfile -ExecutionPolicy Bypass -File ".\repair_torch_metadata.ps1" -PythonExe $Python
}

# ── Step 6: Verify FireRedVAD dependency ──
Test-FireRedVadPackage -PythonExe $Python

if ($DownloadFireRedVAD) {
    Download-FireRedVADModel -PythonExe $Python -Hub $FireRedVADHub
}

# ── Step 7: Install CrispASR native runtime ──
Install-CrispAsrNativeRuntime -PythonExe $Python -UseCuda $HasNvidia

# ── Step 8: Install sherpa-onnx runtime ──
Install-SherpaOnnxRuntime -PythonExe $Python -Runtime $SherpaOnnxRuntime

# ── Step 9: Install FunASR (no-deps) ──
Write-Step "Installing FunASR (--no-deps)..."

& $Uv pip install --python $Python funasr --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-Warn "FunASR installation failed (non-critical, SenseVoice engine may not work)"
} else {
    Write-Ok "FunASR installed"
}

# ── Step 10: Optional parakeet.cpp runtime/model ──
if ($InstallParakeetCpp) {
    Install-ParakeetCppRuntime -PythonExe $Python -Backend $ParakeetCppBackend -Version $ParakeetCppVersion
}

if ($DownloadParakeetCppModel) {
    Download-ParakeetCppModel -PythonExe $Python -ModelName $ParakeetCppModel
}

# ── Done ──
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start LiveTranslate:" -ForegroundColor White
Write-Host "    Double-click start.bat" -ForegroundColor Yellow
Write-Host "    or run: .venv\Scripts\python.exe main.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "  First launch will download ASR models (~1GB)." -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
