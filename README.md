# High-Performance Atrous Convolution (CUDA)

This capstone project starts with three implementations of atrous (dilated) convolution on a grayscale image:

- CPU baseline (C++)
- GPU basic kernel (global memory)
- GPU optimized kernel (shared memory tile + constant-memory kernel)
- Advanced GPU ASPP mode (multi-dilation branches in parallel CUDA streams + fused output)

## Project Structure

- `src/main.cu`: CPU + CUDA implementations and benchmark runner
- `scripts/build.ps1`: build command using `nvcc`
- `scripts/build.sh`: Linux/macOS build command using `nvcc`
- `scripts/run.ps1`: run executable (auto-builds if needed)
- `scripts/run.sh`: Linux/macOS runner (auto-builds if needed)
- `scripts/benchmark.ps1`: runs dilation + block-size sweeps and stores metrics
- `scripts/benchmark.sh`: Linux/macOS benchmark sweep and CSV export
- `scripts/plot_results.py`: generates speedup plots from benchmark CSV
- `scripts/generate_sample_pgm.py`: creates a sample grayscale input image
- `scripts/download_dataset.py`: downloads a real image dataset and converts to PGM
- `results/`: generated outputs and logs

## Requirements

- CUDA Toolkit (with `nvcc` in PATH)
- NVIDIA GPU + compatible driver
- One of the following shells:
	- PowerShell (Windows)
	- Bash (Linux/macOS)

If build says `Could not find nvcc`:

1. Install CUDA Toolkit from NVIDIA.
2. Open a new PowerShell terminal.
3. Verify with:

```powershell
nvcc --version
```

4. If still missing, set environment variable `CUDA_PATH` to your CUDA install (example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5`) and ensure `%CUDA_PATH%\bin` is in your PATH.

## Quick Start (PowerShell)

From project root:

```powershell
./scripts/build.ps1
./scripts/run.ps1 2048 2048 2 20
```

Arguments:

```text
run.ps1 [width] [height] [dilation] [iterations] [block_x] [block_y]
```

Optional direct executable arguments for image mode:

```text
atrous.exe [width] [height] [dilation] [iterations] [block_x] [block_y] [input_pgm] [output_pgm]
```

Defaults:

- width: 2048
- height: 2048
- dilation: 2
- iterations: 20
- block_x: 16
- block_y: 16

## Quick Start (Linux/macOS Bash)

From project root:

```bash
chmod +x scripts/*.sh
./scripts/build.sh
./scripts/run.sh 2048 2048 2 20
```

Arguments:

```text
run.sh [width] [height] [dilation] [iterations] [block_x] [block_y]
```

## Real Image Input (PGM)

Generate a sample image:

```powershell
python scripts/generate_sample_pgm.py
```

Run convolution on the sample image:

```powershell
./bin/atrous.exe 1024 768 2 20 16 16 data/sample_input.pgm results/output_from_file.pgm
```

Notes:

- Input currently supports PGM (`P5` or `P2`) grayscale.
- When input image is provided, image width/height override CLI width/height.

## Download Real Dataset

Download a small real dataset (lena, baboon, fruits, sudoku, smarties) and convert to PGM:

```powershell
pip install -r requirements.txt
python scripts/download_dataset.py
```

Generated folders:

- `data/dataset/raw` (downloaded PNG files)
- `data/dataset/pgm` (PGM files for CUDA app)

Run on a downloaded dataset image:

```powershell
./bin/atrous.exe 512 512 2 20 16 16 data/dataset/pgm/lena.pgm results/lena_out.pgm
```

## Benchmark Sweep

```powershell
./scripts/benchmark.ps1
```

```bash
./scripts/benchmark.sh
```

This generates:

- `results/run_d*_b*x*.txt` (run logs per dilation and block size)
- `results/benchmark.csv`
- `results/output_tiled.pgm` (last run)

`results/benchmark.csv` now contains timing/quality metrics:

- `cpu_ms`, `gpu_basic_ms`, `gpu_tiled_ms`
- `cpu_aspp_ms`, `gpu_aspp_ms`
- `speedup_basic`, `speedup_tiled`, `speedup_aspp`
- `diff_basic`, `diff_tiled`, `diff_aspp`

## Plot Results

Install Python plotting dependency:

```powershell
pip install -r requirements.txt
```

Generate plots from benchmark CSV:

```powershell
python scripts/plot_results.py
```

This writes:

- `results/speedup_vs_dilation.png`
- `results/blocksize_speedup.png`