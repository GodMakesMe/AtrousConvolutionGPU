$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$exe = Join-Path $projectRoot "bin\atrous.exe"
$outputCsv = Join-Path $projectRoot "results\benchmark.csv"

if (-not (Test-Path $exe)) {
    & (Join-Path $PSScriptRoot "build.ps1")
}

"width,height,dilation,iterations,block_x,block_y,cpu_ms,cpu_aspp_ms,gpu_basic_ms,gpu_tiled_ms,gpu_aspp_ms,speedup_basic,speedup_tiled,speedup_aspp,diff_basic,diff_tiled,diff_aspp" | Set-Content $outputCsv

$width = 2048
$height = 2048
$iterations = 30
$dilations = @(1, 2, 4, 8)
$blockSizes = @(
    @(8, 8),
    @(16, 16),
    @(32, 8)
)

foreach ($d in $dilations) {
    foreach ($block in $blockSizes) {
        $bx = $block[0]
        $by = $block[1]
        Write-Host "Running dilation=$d block=${bx}x${by}"

        $runLog = Join-Path $projectRoot "results\run_d${d}_b${bx}x${by}.txt"
        $text = (& $exe $width $height $d $iterations $bx $by) | Tee-Object -FilePath $runLog
        $joined = ($text -join "`n")

        $cpu = [double]([regex]::Match($joined, "CPU avg ms:\s*([0-9.]+)").Groups[1].Value)
        $cpuAspp = [double]([regex]::Match($joined, "CPU ASPP avg ms:\s*([0-9.]+)").Groups[1].Value)
        $gpuBasic = [double]([regex]::Match($joined, "GPU basic avg ms:\s*([0-9.]+)").Groups[1].Value)
        $gpuTiled = [double]([regex]::Match($joined, "GPU tiled avg ms:\s*([0-9.]+)").Groups[1].Value)
        $gpuAspp = [double]([regex]::Match($joined, "GPU ASPP avg ms:\s*([0-9.]+)").Groups[1].Value)
        $spBasic = [double]([regex]::Match($joined, "Speedup basic:\s*([0-9.]+)").Groups[1].Value)
        $spTiled = [double]([regex]::Match($joined, "Speedup tiled:\s*([0-9.]+)").Groups[1].Value)
        $spAspp = [double]([regex]::Match($joined, "Speedup ASPP:\s*([0-9.]+)").Groups[1].Value)
        $diffBasic = [double]([regex]::Match($joined, "Validation max abs diff \(CPU vs basic\):\s*([0-9.]+)").Groups[1].Value)
        $diffTiled = [double]([regex]::Match($joined, "Validation max abs diff \(CPU vs tiled\):\s*([0-9.]+)").Groups[1].Value)
        $diffAspp = [double]([regex]::Match($joined, "Validation max abs diff \(CPU ASPP vs GPU ASPP\):\s*([0-9.]+)").Groups[1].Value)

        "$width,$height,$d,$iterations,$bx,$by,$cpu,$cpuAspp,$gpuBasic,$gpuTiled,$gpuAspp,$spBasic,$spTiled,$spAspp,$diffBasic,$diffTiled,$diffAspp" | Add-Content $outputCsv
    }
}

Write-Host "Saved benchmark metrics to $outputCsv"
