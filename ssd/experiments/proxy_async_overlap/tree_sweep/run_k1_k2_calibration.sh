#!/usr/bin/env bash
# Compatibility entry point.  The maintained tool lives under ssd/tools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${ROOT}/tools/duet_calibration/calibrate_k_balance.sh" "$@"
