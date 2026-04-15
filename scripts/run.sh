#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXE="${PROJECT_ROOT}/bin/atrous"

if [[ ! -x "${EXE}" ]]; then
  "${SCRIPT_DIR}/build.sh"
fi

"${EXE}" "$@"
