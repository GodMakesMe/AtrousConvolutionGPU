$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$exe = Join-Path $projectRoot "bin\atrous.exe"

if (-not (Test-Path $exe)) {
    & (Join-Path $PSScriptRoot "build.ps1")
}

# Usage: .\scripts\run.ps1 [width] [height] [dilation] [iterations]
& $exe @args
