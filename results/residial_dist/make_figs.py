"""Figures for the P2 candidate-form study.  One question per figure.

Everything is averaged over the 12 runs (4 datasets x 3 seeds); the shaded
band is min..max across those runs.  Labels are English so the figures render
without a Korean font; the analysis lives in REPORT.md.
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.labelsize": 12, "legend.fontsize": 11,
                     "figure.facecolor": "white"})

SRC = ("/home/chokwans99/PSD/ssd/experiments/proxy_source_ablation/"
       "probe_ds_seed_20260909/out")
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(FIG, exist_ok=True)

DATA = []
for f in sorted(glob.glob(f"{SRC}/*.json")):
    ds, seed = os.path.basename(f)[:-5].rsplit("_seed", 1)
    DATA.append((ds, int(seed), json.load(open(f))))
d0 = DATA[0][2]
L = np.array(d0["layers"]); B = d0["budgets"]
H = d0["hpol"]; R = d0["rsrc"]; EL = d0["exit_layer"]; bi = B.index(15)
RES, PRX = "#c0392b", "#1f6feb"     # residual = red, proxy = blue


def col(l, h, r, b=15):
    return np.array([d["hit"][L.tolist().index(l)][H.index(h)][R.index(r)][B.index(b)]
                     for _, _, d in DATA])


def curve(h, r):                     # [12, 80]
    i = R.index(r); j = H.index(h)
    return np.array([[d["hit"][k][j][i][bi] for k in range(len(L))]
                     for _, _, d in DATA])


def band(ax, x, arr, c, label, ls="-"):
    ax.plot(x, arr.mean(0), color=c, lw=2.4, ls=ls, label=label)
    ax.fill_between(x, arr.min(0), arr.max(0), color=c, alpha=.18, lw=0)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}", dpi=140); plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- fig 1 -----
fig, ax = plt.subplots(figsize=(8.6, 5.4))
rr, pp = curve("hhat", "rE"), curve("hhat", "pE")
xc = next(l for l, v in zip(L, (rr - pp).mean(0)) if v > 0)
ax.axvspan(xc, 79.5, color=RES, alpha=.07)
band(ax, L, rr, RES, "ours   $[p^E-p^D]_+$")
band(ax, L, pp, PRX, "Mirror-SD   $p^E$")
ax.axvline(EL, color="k", ls=":", lw=1.8)
i56 = L.tolist().index(EL)
ax.plot([EL, EL], [rr.mean(0)[i56], pp.mean(0)[i56]], color="k", lw=1.2)
for v, c, va in ((pp.mean(0)[i56], PRX, "bottom"), (rr.mean(0)[i56], RES, "top")):
    ax.plot([EL], [v], "o", color=c, ms=8, zorder=5)
    ax.text(EL - 2, v, f"{v:.3f}", color=c, ha="right", va=va,
            fontsize=12, weight="bold")
ax.text(EL - 2.5, 0.04, f"deployed exit = {EL}", rotation=90, ha="right", fontsize=11)
ax.text((xc + 79.5) / 2, 0.06, "$p^E \\approx p_T$\nours wins here", color=RES,
        ha="center", fontsize=10.5, weight="bold")
ax.annotate("HIGHER = BETTER", (2, 0.86), fontsize=11, weight="bold",
            color="#374151")
ax.set(xlabel="early-exit layer (of 80)", xlim=(-1, 79.8), ylim=(0, 0.93),
       ylabel="coverage of the true recovery mass   $U$   (R=15)",
       title="Fig 1.  Which candidate form covers the target's\nactual recovery distribution better?")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93)); ax.grid(alpha=.3)
save(fig, "01_coverage.png")

# ---------------------------------------------------------------- fig 2 -----
g = curve("hhat", "rE") - curve("hhat", "pE")
cross = next(l for l, v in zip(L, g.mean(0)) if v > 0)
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.axhspan(-0.12, 0, color=PRX, alpha=.07)
ax.axhspan(0, 0.05, color=RES, alpha=.07)
band(ax, L, g, "#111827", "ours $-$ Mirror-SD")
ax.axhline(0, color="k", lw=1.2)
ax.axvline(EL, color="k", ls=":", lw=1.6)
ax.plot([cross], [0], "o", color="k", ms=8, zorder=5)
ax.annotate(f"crossover\nlayer {cross}", (cross, 0), (cross - 22, 0.018),
            arrowprops=dict(arrowstyle="->"), fontsize=11)
ax.annotate(f"at the deployed exit:\n{g.mean(0)[L.tolist().index(EL)]:+.3f}",
            (EL, g.mean(0)[L.tolist().index(EL)]), (EL - 34, -0.075),
            arrowprops=dict(arrowstyle="->"), fontsize=11)
ax.text(3, 0.028, "our formula wins", color=RES, fontsize=12, weight="bold")
ax.text(3, -0.113, "Mirror-SD wins", color=PRX, fontsize=12, weight="bold")
ax.set(xlabel="early-exit layer (of 80)", ylabel="$U$(ours) $-$ $U$(Mirror-SD)",
       ylim=(-0.12, 0.05),
       title="Fig 2.  The sign only turns in our favour\nin the last few layers")
ax.grid(alpha=.3)
save(fig, "02_gap.png")

# ---------------------------------------------------------------- fig 3 -----
bias = (col(L[0], "htrue", "rT") - col(L[0], "htrue", "pT")).mean()
eps_r = (col(L[0], "htrue", "rT") - col(EL, "htrue", "rE")).mean()
eps_p = (col(L[0], "htrue", "pT") - col(EL, "htrue", "pE")).mean()
net = bias - eps_r + eps_p
steps = [("formula gain\n(exact inputs)", bias, "#16a34a"),
         ("cost of $\\epsilon$\non ours", -eps_r, RES),
         ("cost of $\\epsilon$\navoided by\nMirror-SD", +eps_p, PRX)]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
run = 0.0
for i, (lab, v, c) in enumerate(steps):
    ax.bar(i, v, .62, bottom=run, color=c)
    ax.text(i, run + v / 2, f"{v:+.3f}", ha="center", va="center",
            color="white", weight="bold", fontsize=13)
    ax.plot([i - .31, i + 1.31], [run + v, run + v], color="#9ca3af",
            ls="--", lw=1.1, zorder=0)
    run += v
ax.bar(3, net, .62, color="#111827")
ax.text(3, net / 2, f"{net:+.3f}", ha="center", va="center",
        color="white", weight="bold", fontsize=13)
ax.axhline(0, color="k", lw=1.2)
lo = min(0.0, bias - eps_r)
ax.set_ylim(lo - 0.022, max(0.0, bias) + 0.022)
ax.set_xticks(range(4))
ax.set_xticklabels([s[0] for s in steps] + ["net at\nlayer %d" % EL])
ax.set(ylabel="$\\Delta U$   (R=15)",
       title="Fig 3.  A correct formula, paid for with an approximate input\n"
             "(position budgeting held at the exact hazard)")
ax.grid(alpha=.3, axis="y")
save(fig, "03_decomposition.png")

# ---------------------------------------------------------------- fig 4 -----
a = np.array([[d["approx"][k][0] for k in range(len(L))] for _, _, d in DATA])
b = np.array([[d["approx"][k][1] for k in range(len(L))] for _, _, d in DATA])
fig, ax = plt.subplots(figsize=(8.2, 5))
band(ax, L, a, RES, "ours   $[p^E-p^D]_+$")
band(ax, L, b, PRX, "Mirror-SD   $p^E$")
ax.axvline(EL, color="k", ls=":", lw=1.6)
ax.annotate("with the exact $p_T$ our form\nis 18x closer to the truth",
            (79, a.mean(0)[-1]), (46, 0.10),
            arrowprops=dict(arrowstyle="->"), fontsize=11)
ax.set(xlabel="early-exit layer (of 80)",
       ylabel="TVD to $r_{true}$   (lower is better)",
       title="Fig 4.  How close is each form to the distribution\nthe engine actually samples from?")
ax.legend(loc="lower left"); ax.grid(alpha=.3)
save(fig, "04_approximation.png")

# ---------------------------------------------------------------- fig 5 -----
hE = np.array([[d["approx"][k][5] for k in range(len(L))] for _, _, d in DATA])
hT = np.array([[d["approx"][k][6] for k in range(len(L))] for _, _, d in DATA])
hD = np.array([[d["approx"][k][7] for k in range(len(L))] for _, _, d in DATA])
fig, ax = plt.subplots(figsize=(8.2, 5))
band(ax, L, hE, "#b45309", "$p^E$  (early exit)")
ax.plot(L, hT.mean(0), color="#111827", lw=2.2, ls="--", label="$p_T$  (final head)")
ax.plot(L, hD.mean(0), color="#6b7280", lw=2.0, ls=":", label="$p^D$  (draft)")
ax.axvline(EL, color="k", ls=":", lw=1.6)
ax.set(xlabel="early-exit layer (of 80)", ylabel="entropy (nats)", ylim=(0, 6),
       title="Fig 5.  Why ours is the fragile form:\nthe early exit is FLATTER than the final head")
ax.legend(); ax.grid(alpha=.3)
save(fig, "05_entropy.png")

# ---------------------------------------------------------------- fig 6 -----
bs = np.stack([np.array(d["bins"]) for _, _, d in DATA])
mass = (bs[:, :, 0] / bs[:, :, 0].sum(1, keepdims=True)).mean(0)
gap = ((bs[:, :, 1] - bs[:, :, 2]) / np.maximum(bs[:, :, 0], 1e-12)).mean(0)
lab = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.bar(range(4), gap, .6, color=[RES if v > 0 else PRX for v in gap])
pad = 0.004
for i, (v, m) in enumerate(zip(gap, mass)):
    ax.text(i, v + (pad if v > 0 else -pad), f"{v:+.3f}", ha="center",
            va="bottom" if v > 0 else "top", fontsize=13, weight="bold",
            color=RES if v > 0 else PRX)
    ax.text(i, -pad if v > 0 else pad, f"{m*100:.0f}% of reject mass",
            ha="center", va="top" if v > 0 else "bottom",
            fontsize=10.5, color="#374151")
ax.axhline(0, color="k", lw=1.2)
ax.set_ylim(min(gap) - 0.014, max(gap) + 0.013)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=11)
ax.set(xlabel="how well the draft already mimics the target\noverlap $= 1-TVD(p_T, p^D)$",
       ylabel="alignment(ours) $-$ alignment(Mirror-SD)",
       title="Fig 6.  We only lose where the draft is already good")
ax.grid(alpha=.3, axis="y")
save(fig, "06_where_we_lose.png")

# ---------------------------------------------------------------- fig 7 -----
real = np.array([[d["hit"][L.tolist().index(EL)][H.index("hhat")][R.index("rE")][j]
                  - d["hit"][L.tolist().index(EL)][H.index("hhat")][R.index("pE")][j]
                  for j in range(len(B))] for _, _, d in DATA])
exact = np.array([[d["hit"][0][H.index("htrue")][R.index("rT")][j]
                   - d["hit"][0][H.index("htrue")][R.index("pT")][j]
                   for j in range(len(B))] for _, _, d in DATA])
fig, ax = plt.subplots(figsize=(8.2, 5))
band(ax, np.array(B), exact, "#16a34a", "with the exact $p_T$")
band(ax, np.array(B), real, "#111827", f"with $p^E$ at layer {EL}")
ax.axhline(0, color="k", lw=1.2)
ax.set_xscale("log", base=2); ax.set_xticks(B); ax.set_xticklabels(B)
ax.set(xlabel="cache budget R", ylabel="$U$(ours) $-$ $U$(Mirror-SD)",
       title="Fig 7.  The damage is in the tail:\nthe deeper we fill the cache, the worse ours gets")
ax.legend(); ax.grid(alpha=.3)
save(fig, "07_budget.png")

# ---------------------------------------------------------------- fig 8 -----
nt = np.stack([np.array(d["noise_tvd"]) for _, _, d in DATA]).mean(0)
nh = np.stack([np.array(d["noise_hit"]) for _, _, d in DATA]).mean(0)
tv = np.array([[d["diag"][k][0] for k in range(len(L))] for _, _, d in DATA]).mean(0)
gg = (curve("htrue", "rE") - curve("htrue", "pE")).mean(0)
fig, ax = plt.subplots(figsize=(8.2, 5))
ax.plot(tv, gg, "o", color="#9ca3af", ms=6, label="real early-exit layers")
ax.plot(np.r_[0, nt], np.r_[bias, nh[:, 0, bi] - nh[:, 1, bi]], "s-",
        color="#111827", lw=2.4, ms=7, label="synthetic noise on $p_T$")
ax.axhline(0, color="k", lw=1.2)
ax.axvspan(0, 0.36, color="#16a34a", alpha=.07)
ax.text(0.02, -0.155, "curves agree here", color="#166534", fontsize=11)
ax.set(xlabel="TVD$(\\hat p,\; p_T)$   — how wrong the proxy is",
       ylabel="$U$(ours) $-$ $U$(Mirror-SD)", xlim=(-0.02, 0.72),
       title="Fig 8.  Control: the effect is generic.\nRandom noise of the same size reproduces the curve")
ax.legend(loc="upper right"); ax.grid(alpha=.3)
save(fig, "08_noise_control.png")

# ---------------------------------------------------------------- fig 9 -----
fig, ax = plt.subplots(figsize=(8.2, 5))
names = ["exact hazard\n$h_{true}$", "estimated\n$\\hat h$  (DUET)", "no hazard\nround-robin"]
vals = [(col(EL, h, "rE") - col(EL, h, "pE")).mean() for h in ("htrue", "hhat", "unif")]
ax.bar(range(3), vals, .55, color=PRX)
for i, v in enumerate(vals):
    ax.text(i, v - 0.002, f"{v:+.3f}", ha="center", va="top",
            fontsize=12, weight="bold", color="white")
ax.axhline(0, color="k", lw=1.2)
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set(ylabel="$U$(ours) $-$ $U$(Mirror-SD)",
       title="Fig 9.  Control: not a budgeting artefact.\nOurs loses even when both share the same allocation")
ax.grid(alpha=.3, axis="y")
save(fig, "09_allocation_control.png")
print("\nfigures ->", FIG)
