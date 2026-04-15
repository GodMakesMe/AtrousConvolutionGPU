#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE="${PROJECT_ROOT}/bin/atrous"
OUTPUT_CSV="${PROJECT_ROOT}/results/benchmark.csv"

if [[ ! -x "${EXE}" ]]; then
  "${SCRIPT_DIR}/build.sh"
fi

printf '%s\n' "width,height,dilation,iterations,block_x,block_y,cpu_ms,cpu_aspp_ms,gpu_basic_ms,gpu_tiled_ms,gpu_aspp_ms,speedup_basic,speedup_tiled,speedup_aspp,diff_basic,diff_tiled,diff_aspp" > "${OUTPUT_CSV}"

WIDTH=2048
HEIGHT=2048
ITERATIONS=30
DILATIONS=(1 2 4 8)
BLOCKS=("8 8" "16 16" "32 8")

for D in "${DILATIONS[@]}"; do
  for PAIR in "${BLOCKS[@]}"; do
    read -r BX BY <<< "${PAIR}"
    echo "Running dilation=${D} block=${BX}x${BY}"

    RUN_LOG="${PROJECT_ROOT}/results/run_d${D}_b${BX}x${BY}.txt"
    OUTPUT="$(${EXE} ${WIDTH} ${HEIGHT} ${D} ${ITERATIONS} ${BX} ${BY} | tee "${RUN_LOG}")"

    CPU=$(echo "${OUTPUT}" | sed -n 's/.*CPU avg ms:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    CPU_ASPP=$(echo "${OUTPUT}" | sed -n 's/.*CPU ASPP avg ms:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    GPU_BASIC=$(echo "${OUTPUT}" | sed -n 's/.*GPU basic avg ms:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    GPU_TILED=$(echo "${OUTPUT}" | sed -n 's/.*GPU tiled avg ms:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    GPU_ASPP=$(echo "${OUTPUT}" | sed -n 's/.*GPU ASPP avg ms:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    SP_BASIC=$(echo "${OUTPUT}" | sed -n 's/.*Speedup basic:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    SP_TILED=$(echo "${OUTPUT}" | sed -n 's/.*Speedup tiled:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    SP_ASPP=$(echo "${OUTPUT}" | sed -n 's/.*Speedup ASPP:[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    DIFF_BASIC=$(echo "${OUTPUT}" | sed -n 's/.*Validation max abs diff (CPU vs basic):[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    DIFF_TILED=$(echo "${OUTPUT}" | sed -n 's/.*Validation max abs diff (CPU vs tiled):[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)
    DIFF_ASPP=$(echo "${OUTPUT}" | sed -n 's/.*Validation max abs diff (CPU ASPP vs GPU ASPP):[[:space:]]*\([0-9.]*\).*/\1/p' | head -n1)

    printf '%s\n' "${WIDTH},${HEIGHT},${D},${ITERATIONS},${BX},${BY},${CPU},${CPU_ASPP},${GPU_BASIC},${GPU_TILED},${GPU_ASPP},${SP_BASIC},${SP_TILED},${SP_ASPP},${DIFF_BASIC},${DIFF_TILED},${DIFF_ASPP}" >> "${OUTPUT_CSV}"
  done
done

echo "Saved benchmark metrics to ${OUTPUT_CSV}"
