"""CPU-only checks for handoff; writes solely to a fresh output directory."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    research_roots = [
        ROOT / "results/residial_dist", ROOT / "results/duet_calibration",
        ROOT / "results/duet_tree_analysis", ROOT / "results/duet_tree_al_full",
        ROOT / "results/duet_tree_posthoc", ROOT / "results/duet_tree_followup",
        ROOT / "ssd/experiments/proxy_source_ablation", HERE,
    ]
    files = {p for folder in research_roots for p in folder.rglob("*.py")}
    files.update((ROOT / "ssd/ssd").rglob("*.py"))
    files.add(ROOT / "ssd/bench/bench.py")
    files.add(ROOT / "ssd/tests/test_entropy_probe.py")
    for path in sorted(files):
        ast.parse(path.read_text(), filename=str(path))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    calls = [[sys.executable, str(ROOT / "ssd/tests/test_entropy_probe.py")]]
    # Import the original source, then redirect only the result destination.
    # ROOT and imported pure helpers keep referencing the unchanged repository.
    harness = """import runpy, sys
from pathlib import Path
path=Path(sys.argv[1]); sys.path.insert(0,str(path.parent))
namespace=runpy.run_path(str(path),run_name='handoff_check')
fn=namespace['main']; fn.__globals__['HERE']=Path(sys.argv[2]); fn()
"""
    for name in ("check_math.py", "check_frontier.py"):
        calls.append([sys.executable, "-c", harness,
                      str(ROOT / "results/duet_tree_followup" / name), str(dest)])
    rows = []
    for index, cmd in enumerate(calls):
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (dest / f"check_{index}.log").write_text(result.stdout)
        rows.append(dict(check=["entropy", "math", "frontier"][index],
                         exit_code=result.returncode, log=f"check_{index}.log"))
        print(rows[-1], flush=True)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   python=sys.version, syntax_files=len(files), checks=rows,
                   passed=all(r["exit_code"] == 0 for r in rows),
                   scope="CPU only; no actual-model or GPU validation")
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
