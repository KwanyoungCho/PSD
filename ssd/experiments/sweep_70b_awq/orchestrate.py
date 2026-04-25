"""70B AWQ + TinyLlama AWQ MESA/async-SSD parameter sweep orchestrator.

Implements the staged plan from MESA-PARAMETER-SWEEP-PLAN.md adaptively —
each stage reads completed runs from state.json and decides what's next.

Stack (fixed):
  target = layerskip-llama2-70B (AWQ)
  draft  = TinyLlama-1.1B (AWQ)
  GPUs   = 5 (TP=4 target + 1 draft) on CUDA_VISIBLE_DEVICES=0..4
  B = 1, temp = 0.6, max_model_len = 2048

Stages:
  0 — smoke (small)
  1A — async SSD coarse:    k bracket → f sweep → k recheck
  1B — MESA coarse:         k bracket → f sweep → dfo splits (mid exit, policy A)
  2  — MESA policy A vs B:  top-3 from 1B
  3  — MESA exit sweep:     top-3 from 1B/2 × {1/2, 7/12, 2/3}
  4  — fine sweep:          local refine around best async + best MESA
  5  — confirmation:        top few at numseqs=200, output_len=256

Usage: python orchestrate.py [--stage <0|1A|1B|2|3|4|5|all>]  [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_ONE = ROOT / "run_one.sh"
PARSE = ROOT / "parse.py"
STATE = ROOT / "state.json"
LOG = ROOT / "orchestrator.log"

# 70B has 80 layers
L = 80
EXIT_MID = 7 * L // 12       # 46
EXIT_LIST = [L // 2, EXIT_MID, 2 * L // 3]   # [40, 46, 53]

PORT_BASE = 12700


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"runs": {}}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def cfg_id(stage: str, params: dict) -> str:
    """Deterministic short id used as out subdir."""
    parts = [stage]
    for k in ("mode", "k", "f", "dfo", "exit", "policy", "ns", "ol", "flh"):
        if k in params:
            v = params[k]
            if isinstance(v, list):
                v = "_".join(str(x) for x in v)
            parts.append(f"{k}{v}")
    return "/".join([stage, "_".join(parts[1:])])


def args_for(params: dict) -> list[str]:
    """Translate a params dict to bench.py argv."""
    a = ["--numseqs", str(params.get("ns", 64)),
         "--output_len", str(params.get("ol", 128))]
    if params["mode"] == "ar":
        return a
    a += ["--async", "--spec", "--k", str(params["k"])]
    if "flh" in params:
        a += ["--flh", *[str(x) for x in params["flh"]]]
    else:
        a += ["--f", str(params["f"])]
    if params["mode"] == "mesa":
        a += ["--mesa",
              "--mesa_exit_layer", str(params["exit"]),
              "--mesa_draft_fan_out", str(params["dfo"]),
              "--mesa_policy", params.get("policy", "a")]
    return a


def already_run(state: dict, run_id: str) -> dict | None:
    return state["runs"].get(run_id)


def run_one(params: dict, run_id: str, port: int) -> dict:
    """Invoke run_one.sh sequentially. Returns parsed metrics."""
    out_subdir = run_id
    bench_args = args_for(params)
    env = os.environ.copy()
    env["SSD_DIST_PORT"] = str(port)
    log(f"START {run_id} :: {' '.join(bench_args)}")
    t0 = time.time()
    proc = subprocess.run(
        ["bash", str(RUN_ONE), out_subdir, *bench_args],
        env=env, cwd=str(ROOT), capture_output=False,
    )
    elapsed = time.time() - t0
    log_path = ROOT / out_subdir / "run.log"
    parsed = json.loads(subprocess.check_output(
        [sys.executable, str(PARSE), str(log_path)]).decode())
    parsed["elapsed_s"] = round(elapsed, 1)
    parsed["exit_code"] = proc.returncode
    parsed["params"] = params
    log(f"END   {run_id} :: TP={parsed.get('throughput')} accept={parsed.get('accept')} "
        f"CH={parsed.get('cache_hit')} draft_ms={parsed.get('draft_ms')} "
        f"verify_ms={parsed.get('verify_ms')} ({elapsed:.0f}s)")
    return parsed


def maybe_run(state: dict, params: dict, stage: str, port: int) -> dict:
    rid = cfg_id(stage, params)
    prev = already_run(state, rid)
    if prev and prev.get("completed"):
        log(f"SKIP  {rid} (cached TP={prev.get('throughput')})")
        return prev
    parsed = run_one(params, rid, port)
    state["runs"][rid] = parsed
    save_state(state)
    return parsed


def filter_runs(state: dict, **kw) -> list[dict]:
    out = []
    for rid, r in state["runs"].items():
        if not r.get("completed"):
            continue
        p = r.get("params", {})
        if all(p.get(k) == v for k, v in kw.items()):
            out.append({"id": rid, **r})
    return out


def best(runs: list[dict], key: str = "throughput") -> dict | None:
    runs = [r for r in runs if r.get(key) is not None]
    return max(runs, key=lambda r: r[key]) if runs else None


# ─────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────

def stage_0_smoke(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 0: SMOKE")
    cfgs = [
        {"mode": "ar", "ns": 16, "ol": 64},
        {"mode": "async", "k": 5, "f": 3, "ns": 16, "ol": 64},
        {"mode": "mesa", "k": 5, "f": 4, "dfo": 2, "exit": EXIT_MID, "policy": "a", "ns": 16, "ol": 64},
    ]
    for c in cfgs:
        maybe_run(state, c, "stage0", next(port_iter))


def stage_1a_async_coarse(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 1A: ASYNC SSD COARSE")
    NS, OL = 64, 128
    # Step A: bracket k at f=3
    for k in (3, 5, 7):
        maybe_run(state, {"mode": "async", "k": k, "f": 3, "ns": NS, "ol": OL},
                  "stage1A", next(port_iter))
    best_k_run = best(filter_runs(state, mode="async"))
    best_k = best_k_run["params"]["k"]
    log(f"  best k (at f=3) = {best_k}")
    # Step B: f sweep at best_k
    for f in (2, 4, 6, 8):                       # 3 already done
        maybe_run(state, {"mode": "async", "k": best_k, "f": f, "ns": NS, "ol": OL},
                  "stage1A", next(port_iter))
    best_run = best(filter_runs(state, mode="async"))
    bk, bf = best_run["params"]["k"], best_run["params"]["f"]
    log(f"  best (k, f) so far = ({bk}, {bf}) TP={best_run['throughput']}")
    # Step C: re-check k around best f
    for k in (max(2, bk - 1), bk + 1):
        maybe_run(state, {"mode": "async", "k": k, "f": bf, "ns": NS, "ol": OL},
                  "stage1A", next(port_iter))
    final = best(filter_runs(state, mode="async"))
    log(f"  STAGE 1A best: {final['id']} TP={final['throughput']}")


def stage_1b_mesa_coarse(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 1B: MESA COARSE")
    NS, OL = 64, 128
    POLICY = "a"
    EXIT = EXIT_MID
    # Step A+B: bracket k at (f=6, dfo=2, exit=mid, policy A)
    for k in (3, 5, 7):
        maybe_run(state, {"mode": "mesa", "k": k, "f": 6, "dfo": 2,
                          "exit": EXIT, "policy": POLICY, "ns": NS, "ol": OL},
                  "stage1B", next(port_iter))
    rs = filter_runs(state, mode="mesa", exit=EXIT, policy=POLICY, dfo=2)
    bk = best(rs)["params"]["k"]
    log(f"  best k = {bk}")
    # Step C+D: f × dfo grid at bk
    grid = []
    for f in (4, 6, 8):
        # Always include approx half + at least one < half
        if f == 4:   dfos = [1, 2]
        elif f == 6: dfos = [2, 3]
        else:        dfos = [2, 4]
        for dfo in dfos:
            grid.append((f, dfo))
    for f, dfo in grid:
        maybe_run(state, {"mode": "mesa", "k": bk, "f": f, "dfo": dfo,
                          "exit": EXIT, "policy": POLICY, "ns": NS, "ol": OL},
                  "stage1B", next(port_iter))
    rs = filter_runs(state, mode="mesa", exit=EXIT, policy=POLICY)
    rs = [r for r in rs if r["params"]["k"] == bk]
    rs.sort(key=lambda r: -(r.get("throughput") or 0))
    log(f"  STAGE 1B top-3 at k={bk} exit={EXIT} policy={POLICY}:")
    for r in rs[:3]:
        log(f"    {r['id']} TP={r['throughput']} accept={r['accept']}")


def stage_2_policy_compare(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 2: POLICY A vs B (top-3 from 1B)")
    NS, OL = 64, 128
    rs = filter_runs(state, mode="mesa", exit=EXIT_MID, policy="a")
    rs.sort(key=lambda r: -(r.get("throughput") or 0))
    for r in rs[:3]:
        p = r["params"]
        b_params = {**p, "policy": "b"}
        maybe_run(state, b_params, "stage2", next(port_iter))
    log("  STAGE 2 done. A vs B comparison:")
    for r in rs[:3]:
        p = r["params"]
        b_runs = filter_runs(state, mode="mesa", k=p["k"], f=p["f"],
                              dfo=p["dfo"], exit=EXIT_MID, policy="b")
        b_tp = b_runs[0]["throughput"] if b_runs else None
        log(f"    k={p['k']} f={p['f']} dfo={p['dfo']}: "
            f"A={r['throughput']} B={b_tp}")


def stage_3_exit_sweep(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 3: EXIT LAYER SWEEP (top-3 settings)")
    NS, OL = 64, 128
    # collect best (mode=mesa) from any exit/policy seen so far, take distinct (k,f,dfo,policy)
    rs = filter_runs(state, mode="mesa")
    rs.sort(key=lambda r: -(r.get("throughput") or 0))
    seen = set()
    settings = []
    for r in rs:
        p = r["params"]
        key = (p["k"], p["f"], p["dfo"], p.get("policy", "a"))
        if key in seen:
            continue
        seen.add(key)
        settings.append(p)
        if len(settings) == 3:
            break
    for p in settings:
        for ex in EXIT_LIST:
            params = {**p, "exit": ex, "ns": NS, "ol": OL}
            maybe_run(state, params, "stage3", next(port_iter))


def stage_4_fine(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 4: FINE SWEEP (refine local neighborhoods)")
    NS, OL = 128, 256
    # Async best
    a = best(filter_runs(state, mode="async"))
    if a:
        bk, bf = a["params"]["k"], a["params"]["f"]
        for k in {max(2, bk - 1), bk, bk + 1}:
            for f in {max(1, bf - 1), bf, bf + 1}:
                maybe_run(state, {"mode": "async", "k": k, "f": f, "ns": NS, "ol": OL},
                          "stage4", next(port_iter))
    # MESA best
    m = best(filter_runs(state, mode="mesa"))
    if m:
        p = m["params"]
        for k in {max(2, p["k"] - 1), p["k"], p["k"] + 1}:
            for f in {max(2, p["f"] - 1), p["f"], p["f"] + 1}:
                # keep dfo ≈ p["dfo"] and clamp to (0, f)
                dfo = max(1, min(f - 1, p["dfo"]))
                maybe_run(state, {"mode": "mesa", "k": k, "f": f, "dfo": dfo,
                                   "exit": p["exit"], "policy": p.get("policy", "a"),
                                   "ns": NS, "ol": OL},
                          "stage4", next(port_iter))


def stage_5_confirm(state: dict, port_iter) -> None:
    log("=" * 60); log("STAGE 5: CONFIRMATION (numseqs=200, output_len=256)")
    NS, OL = 200, 256
    a = best(filter_runs(state, mode="async"))
    m = best(filter_runs(state, mode="mesa"))
    cfgs = [{"mode": "ar", "ns": NS, "ol": OL}]
    if a:
        cfgs.append({**a["params"], "ns": NS, "ol": OL})
    if m:
        cfgs.append({**m["params"], "ns": NS, "ol": OL})
    for c in cfgs:
        maybe_run(state, c, "stage5", next(port_iter))


def write_summary(state: dict) -> None:
    out = ROOT / "SUMMARY.md"
    rs = list(state["runs"].values())
    rs = [r for r in rs if r.get("completed")]
    rs.sort(key=lambda r: -(r.get("throughput") or 0))

    lines = ["# 70B AWQ Sweep Summary\n",
             f"Total completed runs: {len(rs)}",
             f"Stack: layerskip-llama2-70B (AWQ TP=4) + TinyLlama-1.1B (AWQ TP=1)\n"]
    lines.append("| run | mode | k | f | dfo | exit | policy | TP | accept | CH | draft_ms | verify_ms | tok/step |")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        p = r["params"]
        def g(k, fmt=""):
            v = r.get(k)
            if v is None: return "—"
            return f"{v:{fmt}}" if fmt else str(v)
        lines.append("| {id} | {mode} | {k} | {f} | {dfo} | {exit} | {policy} | "
                     "{tp} | {ac} | {ch} | {dms} | {vms} | {ts} |".format(
            id=r["id"].split("/")[-1],
            mode=p.get("mode"), k=p.get("k", "—"), f=p.get("f", "—"),
            dfo=p.get("dfo", "—"), exit=p.get("exit", "—"),
            policy=p.get("policy", "—"),
            tp=g("throughput", ".2f"),
            ac=g("accept", ".2f"),
            ch=g("cache_hit", ".2f"),
            dms=g("draft_ms", ".2f"),
            vms=g("verify_ms", ".2f"),
            ts=g("tok_per_step", ".2f"),
        ))
    out.write_text("\n".join(lines) + "\n")
    log(f"Wrote {out}")


def port_iterator(start: int = PORT_BASE):
    p = start
    while True:
        yield p
        p += 2          # leave gap for any retried port


STAGES = {
    "0": stage_0_smoke,
    "1A": stage_1a_async_coarse,
    "1B": stage_1b_mesa_coarse,
    "2": stage_2_policy_compare,
    "3": stage_3_exit_sweep,
    "4": stage_4_fine,
    "5": stage_5_confirm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", help="0|1A|1B|2|3|4|5|all")
    args = ap.parse_args()
    state = load_state()
    ports = port_iterator()
    if args.stage == "all":
        for s in ("0", "1A", "1B", "2", "3", "4", "5"):
            STAGES[s](state, ports)
            write_summary(state)
    else:
        STAGES[args.stage](state, ports)
        write_summary(state)


if __name__ == "__main__":
    main()
