"""Parse all run.log under tmp/full_matrix and produce a consolidated report."""
import os
import re
from pathlib import Path

OUT = Path("/home/chokwans99/PSD/ssd/tmp/full_matrix")

DESC = {
    "A": "Llama-3-8B (bf16 native, TP=2)",
    "B": "Llama-2-7B (fp16 → bf16 runtime override, TP=2)",
    "C": "CodeLlama-34B (fp16 → bf16 runtime override, TP=4)",
}
EXP = {
    "01_ar_dense":         "AR dense",
    "02_ar_int4":          "AR INT4",
    "03_spec_dense":       "async spec dense (k=7 geo)",
    "04_spec_int4":        "async spec INT4",
    "05_mesa_mid_dense":   "MESA dense, exit=mid (~2L/3)",
    "06_mesa_mid_int4":    "MESA INT4, exit=mid",
    "07_mesa_early_dense": "MESA dense, exit=early (~L/2)",
    "08_mesa_early_int4":  "MESA INT4, exit=early",
}


def parse_log(p: Path) -> dict:
    if not p.exists():
        return {}
    text = p.read_text(errors="replace")
    d = {"tp": None, "accept": None, "tok_step": None, "cache_hit": None,
         "time": None, "total_tok": None, "mode": None, "quant_summary": None, "error": None}
    def g(pat, typ=None):
        m = re.search(pat, text)
        if not m: return None
        v = m.group(1)
        return typ(v) if typ else v
    d["tp"]        = g(r"Total Throughput:\s*([\d.]+)", float)
    d["time"]      = g(r"Time:\s*([\d.]+)s", float)
    d["total_tok"] = g(r"Total:\s*([\d.]+)tok", int)
    d["accept"]    = g(r"Avg Fraction of Speculated Tokens Accepted:\s*([\d.]+)", float)
    d["tok_step"]  = g(r"Avg Tokens per step \(incl recovery\):\s*([\d.]+)", float)
    d["cache_hit"] = g(r"Avg Cache Hits:\s*([\d.]+)", float)
    m = re.search(r"\[quant\] summary:[^\n]+", text)
    if m: d["quant_summary"] = m.group(0)
    for pat in [r"ValueError:\s*([^\n]+)", r"(QuantizedLinearNotImplementedError)",
                r"(CUDA out of memory)", r"RuntimeError:\s*([^\n]+)", r"(Killed)"]:
        m = re.search(pat, text)
        if m:
            d["error"] = (m.group(1) if m.lastindex else m.group(0))[:120]
            break
    return d


def fmt(v, spec=".2f"):
    if v is None: return "—"
    try:
        return format(v, spec)
    except Exception:
        return str(v)


def pct(new, base):
    if new is None or base is None or base == 0:
        return "—"
    return f"{(new/base - 1)*100:+.1f}%"


def main():
    subs = sorted([d for d in OUT.iterdir() if d.is_dir() and re.match(r"^[A-C]_\d+_", d.name)])
    results = {d.name: parse_log(d / "run.log") for d in subs}

    print(f"# Full Bench Matrix Results\n")
    print(f"Run at: {os.popen('date').read().strip()}")
    print(f"Total experiments: {len(subs)}\n")

    for grp in "ABC":
        grp_tags = [t for t in results if t.startswith(f"{grp}_")]
        if not grp_tags: continue
        print(f"\n## Target {grp}: {DESC[grp]}\n")
        print("| # | experiment | TP (tok/s) | accept | tok/step | cache_hit | time | note |")
        print("|---|---|---|---|---|---|---|---|")
        for t in sorted(grp_tags):
            exp_key = t[2:]
            r = results[t]
            name = EXP.get(exp_key, exp_key)
            tp = fmt(r.get('tp'))
            if r.get('tp') is None: tp = "**FAIL**"
            acc = fmt(r.get('accept'))
            ts = fmt(r.get('tok_step'))
            ch = fmt(r.get('cache_hit'))
            tm = fmt(r.get('time'), ".1f") + "s" if r.get('time') else "—"
            err = r.get('error') or ""
            idx = exp_key.split("_")[0]
            print(f"| {idx} | {name} | {tp} | {acc} | {ts} | {ch} | {tm} | {err[:50]} |")

        # INT4 vs dense comparison
        print(f"\n### Target {grp} — INT4 vs dense\n")
        print("| config | dense TP | INT4 TP | Δ TP | dense accept | INT4 accept | Δ accept |")
        print("|---|---|---|---|---|---|---|")
        pairs = [
            ("AR", "01_ar_dense", "02_ar_int4"),
            ("async spec", "03_spec_dense", "04_spec_int4"),
            ("MESA mid", "05_mesa_mid_dense", "06_mesa_mid_int4"),
            ("MESA early", "07_mesa_early_dense", "08_mesa_early_int4"),
        ]
        for label, dk, ik in pairs:
            d_ = results.get(f"{grp}_{dk}", {})
            i_ = results.get(f"{grp}_{ik}", {})
            dtp = d_.get("tp"); itp = i_.get("tp")
            dac = d_.get("accept"); iac = i_.get("accept")
            delta_tp = pct(itp, dtp)
            if i_.get("error") and itp is None:
                delta_tp = f"INT4 FAIL: {i_.get('error','')[:35]}"
            delta_acc = "—"
            if dac is not None and iac is not None:
                delta_acc = f"{iac-dac:+.2f}"
            print(f"| {label} | {fmt(dtp)} | {fmt(itp) if itp else 'FAIL'} | {delta_tp} | "
                  f"{fmt(dac)} | {fmt(iac)} | {delta_acc} |")


if __name__ == "__main__":
    main()
