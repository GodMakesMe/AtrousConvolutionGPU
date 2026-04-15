#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/bin"
SRC="${PROJECT_ROOT}/src/main.cu"
OUT="${BIN_DIR}/atrous"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "Could not find nvcc. Install CUDA Toolkit and ensure nvcc is in PATH." >&2
  exit 1
fi

mkdir -p "${BIN_DIR}"

NVCC_ARCH_FLAG=""
if [[ -n "${NVCC_SM:-}" ]]; then
  CC="$(echo "${NVCC_SM}" | tr -cd '0-9')"
  if [[ ${#CC} -ge 2 ]]; then
    NVCC_ARCH_FLAG="-gencode=arch=compute_${CC},code=sm_${CC}"
  fi
elif command -v nvidia-smi >/dev/null 2>&1; then
  CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  CC="$(echo "${CC_RAW}" | tr -cd '0-9')"
  if [[ ${#CC} -ge 2 ]]; then
    NVCC_ARCH_FLAG="-gencode=arch=compute_${CC},code=sm_${CC}"
  fi
fi

ARGS=("-O3" "-std=c++17")
if [[ -n "${NVCC_ARCH_FLAG}" ]]; then
  ARGS+=("${NVCC_ARCH_FLAG}")
fi
ARGS+=("${SRC}" "-o" "${OUT}")

nvcc "${ARGS[@]}"
echo "Built: ${OUT}"
