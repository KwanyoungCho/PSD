#!/usr/bin/env bash
# Compatibility entry point.  The maintained tool lives under ssd/tools.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "${ROOT}/tools/duet_calibration/collect_tree_thresholds.sh" "$@"
