"""Build FINAL.md from state.json — calibration tables + decision-gate answers."""
import json
from pathlib import Path

ROOT = Path("/home/chokwans99/PSD/ssd/experiments/sweep_70b_awq")
state = json.loads((ROOT / "state.json").read_text())
runs = [(rid, r) for rid, r in state["runs"].items() if r.get("completed")]


def fmt_tbl(rows, headers, aligns):
    """rows: list of cells, headers: list, aligns: 'l'|'r'."""
    sep = ["---:" if a == "r" else "---" for a in aligns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(sep) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def fmt_n(v, fmt=".2f"):
    if v is None:
        return "—"
    return f"{v:{fmt}}"


def collect(mode, ns=None, ol=None):
    out = []
    for rid, r in runs:
        p = r["params"]
        if p.get("mode") != mode:
            continue
        if ns is not None and p.get("ns") != ns:
            continue
        if ol is not None and p.get("ol") != ol:
            continue
        out.append({"id": rid, **r, **{"_p": p}})
    out.sort(key=lambda r: -(r.get("throughput") or 0))
    return out


def best(mode, ns=None, ol=None):
    rs = collect(mode, ns, ol)
    return rs[0] if rs else None


# ── Async tables ──
def async_row(r):
    p = r["_p"]
    return [
        p["k"], p["f"],
        fmt_n(r.get("throughput")),
        fmt_n(r.get("draft_ms")),
        fmt_n(r.get("verify_ms")),
        fmt_n(r.get("accept")),
        fmt_n(r.get("cache_hit")),
        fmt_n(r.get("tok_per_step")),
    ]


def mesa_row(r):
    p = r["_p"]
    return [
        p["k"], p["f"], p["dfo"], p["exit"], p.get("policy", "a"),
        fmt_n(r.get("throughput")),
        fmt_n(r.get("draft_ms")),
        fmt_n(r.get("verify_ms")),
        fmt_n(r.get("accept")),
        fmt_n(r.get("cache_hit")),
        fmt_n(r.get("tok_per_step")),
    ]


# ── Build report ──
lines = []

lines.append("# 70B AWQ MESA / Async-SSD Parameter Sweep — Final Report\n")
lines.append("**Stack**: layerskip-llama2-70B (AWQ TP=4) + TinyLlama-1.1B (AWQ TP=1)  ")
lines.append("**Hardware**: 5× RTX 3090 (4 target TP + 1 draft)  ")
lines.append("**Workload**: B=1, temp=0.6, max_model_len=2048, random prompts  ")
lines.append("**Reference**: AR baseline = 32.87 tok/s (from `experiments/quant_awq/70b/ar/`)\n")

lines.append("Sweep ran 51 configs across stages 0→5 (orchestrate.py adaptive driver).")
lines.append(f"All raw data in `state.json`; per-config logs in `stage{{0,1A,1B,2,3,4,5}}/{{config}}/run.log`.\n")

# ── Stage 5 (Confirmation, NS=200/OL=256) ──
lines.append("## Stage 5: Confirmation (numseqs=200, output_len=256)\n")
lines.append("Final calibrated operating points, measured with the largest workload.\n")

s5_async = collect("async", ns=200, ol=256)
s5_mesa = collect("mesa", ns=200, ol=256)
s5 = s5_async + s5_mesa
rows = []
for r in s5:
    p = r["_p"]
    if p.get("mode") == "async":
        rows.append([f"async k={p['k']} f={p['f']}", fmt_n(r.get("throughput")),
                     fmt_n(r.get("draft_ms")), fmt_n(r.get("verify_ms")),
                     fmt_n(r.get("accept")), fmt_n(r.get("cache_hit"))])
    else:
        rows.append([f"mesa k={p['k']} f={p['f']} dfo={p['dfo']} exit={p['exit']} {p.get('policy')}",
                     fmt_n(r.get("throughput")),
                     fmt_n(r.get("draft_ms")), fmt_n(r.get("verify_ms")),
                     fmt_n(r.get("accept")), fmt_n(r.get("cache_hit"))])
rows.append(["AR (target only)", "32.87", "—", "—", "—", "—"])
lines.append(fmt_tbl(rows,
                    ["config", "TP (tok/s)", "draft_ms", "verify_ms", "accept", "cache_hit"],
                    ["l", "r", "r", "r", "r", "r"]))

if s5_async and s5_mesa:
    a = s5_async[0]
    m = s5_mesa[0]
    speedup = (a["throughput"] - m["throughput"]) / m["throughput"] * 100
    lines.append(f"\n**Verdict**: Async (k=7, f=8) wins by **{a['throughput']:.2f} vs {m['throughput']:.2f} tok/s** "
                 f"({speedup:+.1f}% async vs MESA). Both ~2.2-2.3× over AR.\n")

# ── Async calibration table ──
lines.append("## Async SSD Calibration Table\n")
lines.append("All async runs across all stages, sorted by TP.\n")
async_runs = collect("async")
rows = []
for r in async_runs:
    p = r["_p"]
    rows.append([
        f"NS={p.get('ns')}/OL={p.get('ol')}",
        p["k"], p["f"],
        fmt_n(r.get("throughput")),
        fmt_n(r.get("draft_ms")),
        fmt_n(r.get("verify_ms")),
        fmt_n(r.get("accept")),
        fmt_n(r.get("cache_hit")),
        fmt_n(r.get("tok_per_step")),
    ])
lines.append(fmt_tbl(rows,
                    ["workload", "k", "f", "TP", "draft_ms", "verify_ms", "accept", "CH", "T/S"],
                    ["l", "r", "r", "r", "r", "r", "r", "r", "r"]))

# ── MESA calibration table ──
lines.append("\n## MESA Calibration Table\n")
lines.append("All MESA runs across all stages, sorted by TP.\n")
mesa_runs = collect("mesa")
rows = []
for r in mesa_runs:
    p = r["_p"]
    rows.append([
        f"NS={p.get('ns')}/OL={p.get('ol')}",
        p["k"], p["f"], p["dfo"], p["exit"], p.get("policy", "a"),
        fmt_n(r.get("throughput")),
        fmt_n(r.get("draft_ms")),
        fmt_n(r.get("verify_ms")),
        fmt_n(r.get("accept")),
        fmt_n(r.get("cache_hit")),
        fmt_n(r.get("tok_per_step")),
    ])
lines.append(fmt_tbl(rows,
                    ["workload", "k", "f", "dfo", "exit", "pol", "TP", "draft_ms", "verify_ms", "accept", "CH", "T/S"],
                    ["l", "r", "r", "r", "r", "l", "r", "r", "r", "r", "r", "r"]))

# ── Policy A vs B ──
lines.append("\n## Policy A vs B Comparison (Stage 2 + Stage 3, exit=46/40/53)\n")
b_runs = collect("mesa")
b_runs = [r for r in b_runs if r["_p"].get("policy") == "b"]
pol_rows = []
for b in b_runs:
    p = b["_p"]
    # find matching policy A
    for a in mesa_runs:
        ap = a["_p"]
        if (ap.get("policy") == "a"
                and ap["k"] == p["k"] and ap["f"] == p["f"]
                and ap["dfo"] == p["dfo"] and ap["exit"] == p["exit"]
                and ap.get("ns") == p.get("ns") and ap.get("ol") == p.get("ol")):
            delta = b["throughput"] - a["throughput"]
            pct = delta / a["throughput"] * 100
            pol_rows.append([
                f"k={p['k']} f={p['f']} dfo={p['dfo']} exit={p['exit']}",
                fmt_n(a["throughput"]),
                fmt_n(b["throughput"]),
                f"{delta:+.2f}",
                f"{pct:+.1f}%",
            ])
            break
lines.append(fmt_tbl(pol_rows, ["config (NS=64/OL=128)", "Policy A", "Policy B", "ΔTP", "Δ%"],
                    ["l", "r", "r", "r", "r"]))
lines.append("\nPolicy B wins in some configs at smaller f (f=4, dfo=2), Policy A wins at larger f (f=8). "
             "**Effect is small (~±1 tok/s). Policy A remains the more robust default.**\n")

# ── Decision Gate ──
lines.append("\n## Decision Gate Answers\n")
lines.append("Per the explicit decision gate at the end of MESA-PARAMETER-SWEEP-PLAN.md:\n")

a_best = best("async", ns=200, ol=256) or best("async", ns=128, ol=256)
m_best = best("mesa", ns=200, ol=256) or best("mesa", ns=128, ol=256)

ap = a_best["_p"]
mp = m_best["_p"]

lines.append(f"### 1. Best async-SSD parameter set\n")
lines.append(f"`k={ap['k']}, f={ap['f']}` — TP **{a_best['throughput']:.2f}** tok/s "
             f"(NS={ap.get('ns')}/OL={ap.get('ol')})\n")
lines.append(f"Coarse Stage 1A had picked k=6/f=8 (TP=66.47 at NS=64). Fine Stage 4 at NS=128 "
             f"surfaced k=7/f=8 (TP=79.41) which held up at NS=200 (TP=76.78). Higher k saturates "
             f"draft step time (k=7 f=9 dropped 13 tok/s vs k=7 f=8). Lower k has less budget but "
             f"draft is faster — sweet spot is k=7/f=8.\n")

lines.append(f"### 2. Best MESA parameter set\n")
lines.append(f"`k={mp['k']}, f={mp['f']}, dfo={mp['dfo']}, exit_layer={mp['exit']}, "
             f"policy={mp.get('policy')}` — TP **{m_best['throughput']:.2f}** tok/s "
             f"(NS={mp.get('ns')}/OL={mp.get('ol')})\n")
lines.append(f"Stage 1B coarse picked k=5/f=8/dfo=4 (TP=60.99 at NS=64). Stage 3 exit-layer sweep "
             f"surfaced exit=53 as best. Stage 4 fine settled on k=6/f=3/dfo=2 (TP=72.81 at NS=128). "
             f"Stage 5 confirmed at NS=200: TP=72.25. **Smaller total budget (f=3) plus deeper exit "
             f"(53/80 = 66%) beats the larger budget MESA configs** — the deeper proxy gives better "
             f"correction quality, and small f keeps draft step bounded.\n")

a_tp = a_best['throughput']
m_tp = m_best['throughput']
gap = (a_tp - m_tp) / m_tp * 100

lines.append(f"### 3. Is draft still the dominant bottleneck?\n")
lines.append(f"**Mostly balanced now, but draft slightly dominates MESA**:\n")
lines.append(f"- Best async (k=7/f=8 NS=200): draft={a_best.get('draft_ms'):.1f}ms, verify={a_best.get('verify_ms'):.1f}ms — "
             f"draft ≈ verify (~equal)\n")
lines.append(f"- Best MESA (k=6/f=3 NS=200): draft={m_best.get('draft_ms'):.1f}ms, verify={m_best.get('verify_ms'):.1f}ms — "
             f"**draft still 2-3 ms over verify** (2-pass overhead)\n")
lines.append(f"\nIn Stage 1A coarse (NS=64) draft and verify were both ~33-46 ms across configs; in fine sweep "
             f"the draft path (esp for MESA) remains the slightly slower side. AWQ on the draft already "
             f"removed the worst draft-bound asymmetry.\n")

lines.append(f"### 4. Gap large enough to justify Phase 1 / Phase 2 redesign?\n")
lines.append(f"**No, not at this stack.** Async beats MESA by **{gap:.1f}%** ({a_tp:.2f} vs {m_tp:.2f} tok/s). "
             f"The MESA 2-pass structural cost (~2-3 ms extra draft) is small in absolute terms but "
             f"larger than the cache-hit gain (MESA CH=0.83 vs async CH=0.79 — only +0.04). Token "
             f"efficiency is genuinely improved (MESA accept ≈ async, plus better CH), but it doesn't "
             f"compound to throughput because Phase 2 replay cost isn't free even at this scale.\n\n"
             f"A shorter-Phase-1 redesign would need to recover several ms per step *and* preserve "
             f"the proxy's correction quality. At a 4.5 tok/s gap, the engineering cost of new layout/"
             f"graph variants is unlikely to pay off on this stack. **Recommendation: defer redesign**, "
             f"keep async (k=7, f=8) as the production default.\n")

lines.append(f"### 5. Is Policy B robust enough to be the default?\n")
lines.append(f"**No, Policy A remains the more robust default.** From Stage 2/3:\n")
lines.append(f"- Policy B helps at small budget (f=4, dfo=2): exit=40 +1.8, exit=46 -0.9, exit=53 -2.1\n")
lines.append(f"- Policy B hurts at larger budget (f=8, dfo=4 → -1.7) and at f=6/dfo=3 → -1.7\n")
lines.append(f"- No config showed Policy B improving by more than ~+1.8 tok/s\n")
lines.append(f"\nPolicy B's joint `ĥ_i × r̂_i(v)` ranking concentrates budget on a few high-confidence "
             f"(position, token) pairs. On 70B this matches the residual correction signal in some "
             f"regimes but starves alternative branches — net wash. Keep Policy A as default; expose "
             f"`--mesa_policy b` for users who want to experiment.\n")

# ── Summary ──
lines.append(f"\n## Bottom Line\n")
lines.append(f"| Mode | Best Config | TP (NS=200) | vs AR | vs Async |\n|---|---|---:|---:|---:|")
lines.append(f"| AR | (target only, TP=4) | 32.87 | 1.00× | 0.43× |")
lines.append(f"| Async SSD | k=7, f=8 | **{a_tp:.2f}** | **{a_tp/32.87:.2f}×** | **1.00×** |")
lines.append(f"| MESA | k=6, f=3, dfo=2, exit=53, policy=A | {m_tp:.2f} | {m_tp/32.87:.2f}× | {m_tp/a_tp:.2f}× |\n")
lines.append(f"**Production default**: Async SSD with k=7, f=8 on this stack.")

(ROOT / "FINAL.md").write_text("\n".join(lines) + "\n")
print(f"Wrote {ROOT / 'FINAL.md'}")
print("=" * 60)
print((ROOT / "FINAL.md").read_text()[:2000])
