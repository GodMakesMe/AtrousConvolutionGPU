$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$binDir = Join-Path $projectRoot "bin"
$src = Join-Path $projectRoot "src\main.cu"
$out = Join-Path $binDir "atrous.exe"

$nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
$nvccPath = $null
if ($nvccCommand) {
    $nvccPath = $nvccCommand.Source
} elseif ($env:CUDA_PATH) {
    $candidate = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
    if (Test-Path $candidate) {
        $nvccPath = $candidate
    }
}

if (-not $nvccPath) {
    throw "Could not find nvcc. Install CUDA Toolkit and ensure nvcc is available in PATH or CUDA_PATH is set."
}

if (-not (Test-Path $binDir)) {
    New-Item -ItemType Directory -Path $binDir | Out-Null
}

$nvccArchFlag = $null
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($env:NVCC_SM) {
    $cc = ($env:NVCC_SM -replace "[^0-9]", "")
    if ($cc.Length -ge 2) {
        $nvccArchFlag = "-gencode=arch=compute_$cc,code=sm_$cc"
    }
} elseif ($nvidiaSmi) {
    $ccRaw = & $nvidiaSmi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1
    if ($ccRaw) {
        $cc = ($ccRaw.Trim() -replace "[^0-9]", "")
        if ($cc.Length -ge 2) {
            $nvccArchFlag = "-gencode=arch=compute_$cc,code=sm_$cc"
        }
    }
}

$nvccArgs = @("-O3", "-std=c++17")
if ($nvccArchFlag) {
    $nvccArgs += $nvccArchFlag
}
$nvccArgs += @($src, "-o", $out)

$clCommand = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clCommand) {
    & $nvccPath @nvccArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
} else {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "Build failed: cl.exe not found and vswhere.exe is missing. Install Visual Studio Build Tools C++ workload."
    }

    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsInstall) {
        throw "Build failed: cl.exe not found and no Visual Studio C++ toolchain installation was detected."
    }

    $vcvars = Join-Path $vsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) {
        throw "Build failed: could not locate vcvars64.bat at $vcvars"
    }

    $archPart = if ($nvccArchFlag) { "$nvccArchFlag " } else { "" }
    $cmd = '"{0}" >nul && "{1}" -O3 -std=c++17 {2}"{3}" -o "{4}"' -f $vcvars, $nvccPath, $archPart, $src, $out
    cmd.exe /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
}

Write-Host "Built: $out"
