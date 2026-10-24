#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"

"${BUILD_DIR}/fiber_pfczm" --smoke --output="${RESULTS_DIR}/smoke"
