#!/usr/bin/env python3
"""Compatibility entry point; use tools/duet_calibration/analyze_thresholds.py."""
from pathlib import Path
import importlib.util
import sys

_PATH = Path(__file__).resolve().parents[3] / "tools/duet_calibration/analyze_thresholds.py"
_SPEC = importlib.util.spec_from_file_location("duet_calibration_thresholds", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
globals().update({k: v for k, v in vars(_MODULE).items() if not k.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(main())
