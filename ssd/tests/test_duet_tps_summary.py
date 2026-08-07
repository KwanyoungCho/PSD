import importlib.util
import sys
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1]
          / "tools/duet_calibration/summarize_tps.py")
SPEC = importlib.util.spec_from_file_location("duet_tps_summary", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _run(root, mode, k1, k2, seed, tps, tok):
    path = root / f"{mode}_k1_{k1}_k2_{k2}_seed_{seed}"
    path.mkdir()
    path.joinpath("run.log").write_text(
        f"Final Decode Throughput: {tps}tok/s\n"
        f"Avg Tokens per step (incl recovery): {tok}\n"
        "Avg Phase 1 (draft) Hit Rate: 0.5\n"
        "Avg Phase 2 (proxy) Hit Rate: 0.2\n"
        "Avg Phase 1 Accepted Len: 4.0\n"
        "Avg Phase 2 Accepted Len: 2.0\n"
        "EXIT:0\n")


def test_summary_and_paired_delta(tmp_path):
    _run(tmp_path, "tree", 9, 3, 42, 70, 4.0)
    _run(tmp_path, "tree", 9, 4, 42, 75, 4.2)
    _run(tmp_path, "tree", 9, 3, 123, 72, 4.1)
    _run(tmp_path, "tree", 9, 4, 123, 73, 4.2)
    rows = MOD.load_runs(tmp_path)
    groups = MOD.summarize(rows)
    assert [x["tps_mean"] for x in groups] == [71, 74]
    paired = MOD.paired_tps(rows)
    assert paired[0]["right_minus_left_tps_mean"] == 3
    assert paired[0]["deltas"] == [5, 1]
