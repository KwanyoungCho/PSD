"""Aggregate the 3-arm proxy-source ablation logs into a per-arm table."""
import os, re, sys, statistics as st

LOGS = os.environ.get("LOGS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
ARMS = ["residual", "proxy", "draft"]
SEEDS = [42, 123, 7]
PATS = {
    "cache_hit":  r"Avg Cache Hits: ([\d.]+)",
    "p1_hit":     r"Avg Phase 1 \(draft\) Hit Rate: ([\d.]+)",
    "p2_hit":     r"Avg Phase 2 \(proxy\) Hit Rate: ([\d.]+)",
    "p1_al":      r"Avg Phase 1 Accepted Len: ([\d.]+)",
    "p2_al":      r"Avg Phase 2 Accepted Len: ([\d.]+)",
    "tok_step":   r"Avg Tokens per step \(incl recovery\): ([\d.]+)",
    "acc_len":    r"Avg Fraction of Speculated Tokens Accepted: ([\d.]+)",
    "tok_hit":    r"Avg Tokens per step on Cache Hit: ([\d.]+)",
    "decode_tps": r"Final Decode Throughput: ([\d.]+)tok/s",
}

def parse(path):
    txt = open(path, errors="ignore").read()
    out = {}
    for k, p in PATS.items():
        m = re.findall(p, txt)
        out[k] = float(m[-1]) if m else None
    return out

data = {a: {s: parse(f"{LOGS}/{a}_seed{s}.log") for s in SEEDS} for a in ARMS}

cols = ["p2_hit", "p1_hit", "cache_hit", "tok_step", "acc_len", "tok_hit", "decode_tps"]
hdr = ["p2_hit", "p1_hit", "cache", "tok/step", "accept", "tok@hit", "TPS"]

print("=== per-seed ===")
print(f"{'arm':9s} {'seed':>5s} " + " ".join(f"{h:>9s}" for h in hdr))
for a in ARMS:
    for s in SEEDS:
        d = data[a][s]
        print(f"{a:9s} {s:5d} " + " ".join(
            f"{d[c]:9.3f}" if d[c] is not None else f"{'--':>9s}" for c in cols))

print("\n=== mean over 3 seeds (±  = min/max spread) ===")
print(f"{'arm':9s} " + " ".join(f"{h:>16s}" for h in hdr))
means = {}
for a in ARMS:
    row, means[a] = [], {}
    for c in cols:
        v = [data[a][s][c] for s in SEEDS if data[a][s][c] is not None]
        if not v: row.append(f"{'--':>16s}"); continue
        m = st.mean(v); means[a][c] = m
        row.append(f"{m:9.3f}±{(max(v)-min(v))/2:5.3f}")
    print(f"{a:9s} " + " ".join(row))

print("\n=== vs residual (champion) ===")
print(f"{'arm':9s} " + " ".join(f"{h:>12s}" for h in hdr))
for a in ARMS:
    if a == "residual": continue
    row = []
    for c in cols:
        b, m = means["residual"].get(c), means[a].get(c)
        row.append(f"{(m-b)/b*100:+11.1f}%" if b else f"{'--':>12s}")
    print(f"{a:9s} " + " ".join(row))
