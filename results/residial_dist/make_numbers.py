"""Every number quoted in REPORT.md, regenerated from the raw probe JSONs."""
import glob, json, os
import numpy as np

SRC = ("/home/chokwans99/PSD/ssd/experiments/proxy_source_ablation/"
       "probe_ds_seed_20260909/out")
runs = {}
for f in sorted(glob.glob(f"{SRC}/*.json")):
    ds, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
    runs.setdefault(ds, {})[int(seed)] = json.load(open(f))
DS = list(runs); SE = sorted(runs[DS[0]])
d0 = runs[DS[0]][SE[0]]
L = d0["layers"]; B = d0["budgets"]; H = d0["hpol"]; R = d0["rsrc"]
EL = d0["exit_layer"]; bi = B.index(15); li = L.index(EL)


def cell(d, l, h, r, b=15):
    return d["hit"][L.index(l)][H.index(h)][R.index(r)][B.index(b)]


def tab(title, fn, fmt="{:+.4f}"):
    print(f"\n### {title}")
    print(f"{'dataset':>10s} " + " ".join(f"{s:>9d}" for s in SE)
          + f" {'mean':>9s} {'±range':>8s}")
    allv = []
    for ds in DS:
        vs = [fn(runs[ds][s]) for s in SE]; allv += vs
        print(f"{ds:>10s} " + " ".join(fmt.format(v) for v in vs)
              + f" {fmt.format(np.mean(vs)):>9s} {(max(vs)-min(vs))/2:8.4f}")
    print(f"{'ALL':>10s} " + " " * (10 * len(SE))
          + f" {fmt.format(np.mean(allv)):>9s}")
    return allv


print("=" * 78)
print("P2 candidate-form study -- every quoted number")
print("=" * 78)
print(f"runs        : {len(DS)} datasets x {len(SE)} seeds = {len(DS)*len(SE)}")
print(f"steps/run   : {[runs[ds][s]['n_steps'] for ds in DS for s in SE]}")
print(f"total steps : {sum(runs[ds][s]['n_steps'] for ds in DS for s in SE)}")
print(f"exit layer  : {EL} of {len(L)}   budget R=15   top_m={d0['top_m']}")
print(f"probe layers: all {len(L)}")

print("\n" + "-" * 78 + "\n## 1. Is the formula right?  (exact inputs, no approximation)")
tab("formula gain  U(htrue x rT) - U(htrue x pT)",
    lambda d: cell(d, L[0], "htrue", "rT") - cell(d, L[0], "htrue", "pT"))
tab("optimality check  ceiling - U(htrue x rT)   (must be >= 0, ~0)",
    lambda d: d["ceiling"][bi] - cell(d, L[0], "htrue", "rT"))
tab("TVD(residual form, r_true) at the FINAL layer 79  (~0 expected)",
    lambda d: d["approx"][L.index(79)][0], "{:.4f}")
tab("TVD(proxy form, r_true) at the FINAL layer 79  (= pure form bias)",
    lambda d: d["approx"][L.index(79)][1], "{:.4f}")

print("\n" + "-" * 78 + f"\n## 2. What actually happens at the deployed exit layer {EL}")
tab(f"observed gap  U(hhat x rE) - U(hhat x pE)",
    lambda d: cell(d, EL, "hhat", "rE") - cell(d, EL, "hhat", "pE"))
tab("U(hhat x rE)  -- what DUET runs today", lambda d: cell(d, EL, "hhat", "rE"), "{:.4f}")
tab("U(hhat x pE)  -- Mirror-SD-style token ranking",
    lambda d: cell(d, EL, "hhat", "pE"), "{:.4f}")
tab("ceiling (Bayes-optimal, exact inputs)", lambda d: d["ceiling"][bi], "{:.4f}")

print("\n" + "-" * 78 + "\n## 3. Decomposition of the sign flip (h fixed to h_true)")
tab("epsilon cost of the residual form  U(rT) - U(rE)",
    lambda d: cell(d, L[0], "htrue", "rT") - cell(d, EL, "htrue", "rE"))
tab("epsilon cost of the proxy form     U(pT) - U(pE)",
    lambda d: cell(d, L[0], "htrue", "pT") - cell(d, EL, "htrue", "pE"))
tab("identity residual: gap - [bias - (eps_r - eps_p)]  (must be ~0)",
    lambda d: (cell(d, EL, "htrue", "rE") - cell(d, EL, "htrue", "pE"))
    - ((cell(d, L[0], "htrue", "rT") - cell(d, L[0], "htrue", "pT"))
       - ((cell(d, L[0], "htrue", "rT") - cell(d, EL, "htrue", "rE"))
          - (cell(d, L[0], "htrue", "pT") - cell(d, EL, "htrue", "pE")))),
    "{:+.2e}")

print("\n" + "-" * 78 + "\n## 4. Character of the early-exit distribution")
tab(f"H(p^E) - H(p_T) at layer {EL}   (>0 = flatter)",
    lambda d: d["approx"][li][5] - d["approx"][li][6])
tab(f"TVD(p^E,p^D) - TVD(p_T,p^D) at layer {EL}   (>0 = NOT pulled to draft)",
    lambda d: d["approx"][li][3] - d["approx"][li][4])
tab(f"TVD(residual form) - TVD(proxy form) vs r_true at layer {EL}",
    lambda d: d["approx"][li][0] - d["approx"][li][1])

print("\n" + "-" * 78 + "\n## 5. Controls")
tab("crossover layer (lowest layer where residual wins)",
    lambda d: next((l for l in L if cell(d, l, "hhat", "rE")
                    > cell(d, l, "hhat", "pE")), 99), "{:9.0f}")
tab(f"position-budgeting control  U(unif x rE) - U(unif x pE) at {EL}",
    lambda d: cell(d, EL, "unif", "rE") - cell(d, EL, "unif", "pE"))
print("\n### candidate-pool depth control (top_m), gap at exit layer, hhat")
print(f"{'top_m':>8s} " + " ".join(f"{ds:>10s}" for ds in DS))
for j, m in enumerate(d0["ms"]):
    row = []
    for ds in DS:
        v = [runs[ds][s]["msweep"][j][2][bi] - runs[ds][s]["msweep"][j][3][bi]
             for s in SE]
        row.append("n/a" if all(x == 0 for x in v) else f"{np.mean(v):+.4f}")
    print(f"{m:8d} " + " ".join(f"{x:>10s}" for x in row))

print("\n### synthetic-noise control (all runs averaged, h fixed to h_true)")
nt = np.stack([runs[ds][s]["noise_tvd"] for ds in DS for s in SE]).mean(0)
nh = np.stack([runs[ds][s]["noise_hit"] for ds in DS for s in SE]).mean(0)
r0 = np.mean([cell(runs[ds][s], L[0], "htrue", "rT")
              - cell(runs[ds][s], L[0], "htrue", "pT") for ds in DS for s in SE])
print(f"{'delta':>7s} {'TVD':>7s} {'gap':>8s}")
print(f"{0.0:7.2f} {0.0:7.4f} {r0:+8.4f}")
for j, dl in enumerate(d0["deltas"]):
    print(f"{dl:7.2f} {nt[j]:7.4f} {nh[j][0][bi]-nh[j][1][bi]:+8.4f}")

print("\n" + "-" * 78 + "\n## 6. Where the loss lives (overlap = 1 - TVD(p_T,p^D))")
print(f"{'bucket':>12s} {'mass':>8s} {'residual':>9s} {'proxy':>8s} {'gap':>8s}")
lab = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
bs = np.stack([runs[ds][s]["bins"] for ds in DS for s in SE])
mass = (bs[:, :, 0] / bs[:, :, 0].sum(1, keepdims=True)).mean(0)
rr = (bs[:, :, 1] / np.maximum(bs[:, :, 0], 1e-12)).mean(0)
pp = (bs[:, :, 2] / np.maximum(bs[:, :, 0], 1e-12)).mean(0)
for k in range(4):
    print(f"{lab[k]:>12s} {mass[k]:8.3f} {rr[k]:9.4f} {pp[k]:8.4f} {rr[k]-pp[k]:+8.4f}")

print("\n" + "-" * 78 + "\n## 7. Hazard estimate quality")
tab(f"TVD(h_hat, h_true) at layer {EL}", lambda d: d["diag"][li][2], "{:.4f}")
tab(f"P(argmax h_hat = argmax h_true) at layer {EL}",
    lambda d: d["diag"][li][4], "{:.4f}")
tab(f"value of hazard weighting, exact:  U(htrue x rT) - U(unif x rT)",
    lambda d: cell(d, L[0], "htrue", "rT") - cell(d, L[0], "unif", "rT"))
tab(f"value of hazard weighting, realised: U(hhat x rE) - U(unif x rE)",
    lambda d: cell(d, EL, "hhat", "rE") - cell(d, EL, "unif", "rE"))
tab("effective number of budgeted positions (of K+1)",
    lambda d: d["eff_positions"], "{:9.2f}")
tab("share of budget landing on the all-accept row",
    lambda d: d["budget_share_posK"], "{:9.3f}")
tab("reject mass (1 - all-accept)", lambda d: d["reject_mass"], "{:9.3f}")

print("\n" + "-" * 78 + "\n## 8. Budget sweep at the exit layer (mean over all runs)")
print(f"{'R':>5s} {'residual':>9s} {'proxy':>8s} {'gap':>8s} {'ceiling':>8s} "
      f"{'exact-input gap':>16s}")
for j, b in enumerate(B):
    a = np.mean([runs[ds][s]["hit"][li][H.index("hhat")][R.index("rE")][j]
                 for ds in DS for s in SE])
    p = np.mean([runs[ds][s]["hit"][li][H.index("hhat")][R.index("pE")][j]
                 for ds in DS for s in SE])
    c = np.mean([runs[ds][s]["ceiling"][j] for ds in DS for s in SE])
    e = np.mean([runs[ds][s]["hit"][0][H.index("htrue")][R.index("rT")][j]
                 - runs[ds][s]["hit"][0][H.index("htrue")][R.index("pT")][j]
                 for ds in DS for s in SE])
    print(f"{b:5d} {a:9.4f} {p:8.4f} {a-p:+8.4f} {c:8.4f} {e:+16.4f}")
