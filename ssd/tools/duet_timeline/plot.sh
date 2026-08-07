#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PROFILE_DIR [plotter options...]" >&2
  exit 2
fi

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSD_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"
PROFILE_DIR="$1"
shift
PY="${PY:-python}"

"$PY" "$SSD_ROOT/bench/plot_duet_aligned_timeline.py" \
  "$PROFILE_DIR" --causality-shift "$@"
