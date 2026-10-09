"""CPU-only target/draft-distance diagnostics on saved direct-comparison runs.

Reuse stored GPU top-3 outcomes; no new top-k, policy selection, or inference.
Alignment is computed from full probability snapshots on the same contexts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DIRECT = HERE.parent / "direct_comparison"
EDGES = np.array([0., .25, .5, .75, 1.])
REPS = 4000
COLORS = ("#be5046", "#3479ae")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(group):
    records = []
    for path in sorted((DIRECT / group).glob("*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path) as z:
            rec = {k: z[k].copy() for k in
                   ["step", "seq_id", "row::z", "row::h",
                    "row::residual", "row::proxy"]}
        rec.update(dataset=meta["record"]["dataset"],
                   seed=meta["record"]["seed"], name=path.stem,
                   metric_file=path, manifest=ROOT / meta["manifest"])
        records.append(rec)
    return records


def alignment(rec, reuse):
    cache = HERE / "distance_alignment" / (rec["name"] + ".npz")
    cache.parent.mkdir(exist_ok=True)
    source_hash = digest(rec["metric_file"])
    manifest_hash = digest(rec["manifest"])
    if reuse and cache.exists():
        with np.load(cache) as z:
            assert str(z["metric_sha256"]) == source_hash
            assert str(z["manifest_sha256"]) == manifest_hash
            np.testing.assert_array_equal(z["step"], rec["step"])
            return z["alignment"].copy()
    meta = json.loads(rec["manifest"].read_text())
    assert meta["layers"] == [56, 79] and meta["temperature"] == 1.
    parts, steps, seqs, zparts = [], [], [], []
    for chunk in meta["chunks"]:
        with np.load(rec["manifest"].parent / chunk["file"]) as z:
            n = len(z["step"]) // 5
            np.testing.assert_array_equal(z["position"], np.tile(np.arange(5), n))
            p = z["p_T"].reshape(n, 5, -1)[:, :4].astype(np.float64)
            q = z["p_D"].reshape(n, 5, -1)[:, :4].astype(np.float64)
            e = z["p_E"][:, 0].reshape(n, 5, -1)[:, :4].astype(np.float64)
            y = z["y"].reshape(n, 5)[:, :4]
            for prob in [p, q, e]:
                assert np.isfinite(prob).all() and prob.min() >= 0
                np.testing.assert_allclose(prob.sum(-1), 1., atol=2e-6, rtol=0)
            a = np.maximum(p - q, 0)
            mass = a.sum(-1, keepdims=True)
            r = np.divide(a, mass, out=np.zeros_like(a), where=mass > 1e-10)
            residual = np.maximum(e - q, 0)
            np.put_along_axis(residual, y[..., None], 0, axis=-1)
            np.put_along_axis(e, y[..., None], 0, axis=-1)
            normalized = [s / np.maximum(s.sum(-1, keepdims=True), 1e-10)
                          for s in [residual, e]]
            parts.append(np.stack([(r * s).sum(-1) for s in normalized], axis=-1))
            zparts.append(mass[..., 0])
            steps.append(z["step"][::5])
            seqs.append(z["seq_id"][::5])
    np.testing.assert_array_equal(np.concatenate(steps), rec["step"])
    np.testing.assert_array_equal(np.concatenate(seqs), rec["seq_id"])
    np.testing.assert_allclose(np.concatenate(zparts), rec["row::z"][:, :4],
                               atol=2e-6, rtol=0)
    values = np.concatenate(parts)
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1 + 2e-6
    np.savez_compressed(cache, alignment=values, step=rec["step"],
                        metric_sha256=source_hash, manifest_sha256=manifest_hash)
    print("Full-snapshot alignment:", rec["name"], len(values), "steps", flush=True)
    return values


def clusters(records, mode, include_alignment):
    """Sufficient statistics per prompt, with all its seeds grouped together."""
    by_prompt = {}
    rows = np.zeros(4, dtype=int)
    metrics = ["coverage_residual", "coverage_proxy"]
    if include_alignment:
        metrics += ["alignment_residual", "alignment_proxy"]
    for rec in records:
        z = rec["row::z"][:, :4].astype(float).clip(0, 1)
        bins = np.digitize(z, EDGES[1:-1])
        h = rec["row::h"][:, :4].astype(float)
        values = np.stack([rec["row::residual"][:, :4],
                           rec["row::proxy"][:, :4]], -1).astype(float)
        if include_alignment:
            values = np.concatenate([values, rec["alignment"]], -1)
        if mode == "step_normalized":
            h = np.divide(h, h.sum(1, keepdims=True), out=np.zeros_like(h),
                          where=h.sum(1, keepdims=True) > 1e-10)
        for b in range(4):
            rows[b] += np.count_nonzero((bins == b) & (h > 0))
        for pid in np.unique(rec["seq_id"]):
            use = rec["seq_id"] == pid
            stats = np.zeros((4, 1 + len(metrics)))
            for b in range(4):
                w = h[use] * (bins[use] == b)
                stats[b, 0] = w.sum()
                stats[b, 1:] = (w[..., None] * values[use]).sum((0, 1))
            key = (rec["dataset"], int(pid))
            by_prompt[key] = by_prompt.get(key, np.zeros_like(stats)) + stats
    arrays = {}
    for ds in sorted({k[0] for k in by_prompt}):
        arrays[ds] = np.stack([v for k, v in sorted(by_prompt.items()) if k[0] == ds])
    return arrays, metrics, rows


def summarize(records, mode, include_alignment):
    groups, names, rows = clusters(records, mode, include_alignment)
    point = sum(x.sum(0) for x in groups.values())
    rng = np.random.default_rng(20261001)
    samples = sum(x[rng.integers(len(x), size=(REPS, len(x)))].sum(1)
                  for x in groups.values())
    assert np.all(samples[:, :, 0] > 0)
    out = dict(weighting=mode, metric_names=names, bins=[], by_dataset={})
    for b in range(4):
        mean = point[b, 1:] / point[b, 0]
        boot = samples[:, b, 1:] / samples[:, b, 0, None]
        item = dict(tv_low=float(EDGES[b]), tv_high=float(EDGES[b + 1]),
                    event_mass_fraction=float(point[b, 0] / point[:, 0].sum()),
                    positive_h_contexts=int(rows[b]),
                    contributing_prompts=int(sum((x[:, b, 0] > 0).sum()
                                                 for x in groups.values())))
        for i, name in enumerate(names):
            item[name] = dict(mean=float(mean[i]),
                              ci95=np.quantile(boot[:, i], [.025, .975]).tolist())
        for label, i in [("coverage_gap", 0)] + ([("alignment_gap", 2)] if include_alignment else []):
            item[label] = dict(mean=float(mean[i] - mean[i + 1]),
                ci95=np.quantile(boot[:, i] - boot[:, i + 1], [.025, .975]).tolist())
        out["bins"].append(item)
    for ds, x in groups.items():
        total = x.sum(0)
        out["by_dataset"][ds] = (total[:, 1:] / total[:, :1]).tolist()
    out["sampled_steps"] = sum(len(r["step"]) for r in records)
    out["unique_prompts"] = sum(map(len, groups.values()))
    out["generation_runs"] = len(records)
    return out


def bars(ax, items, key, scale, color):
    y = np.array([b[key]["mean"] for b in items]) * scale
    ci = np.array([b[key]["ci95"] for b in items]) * scale
    xx = np.arange(4)
    ax.bar(xx, y, .58, color=color)
    ax.errorbar(xx, y, yerr=[y - ci[:, 0], ci[:, 1] - y],
                fmt="none", ecolor="#333333", capsize=4, lw=1)
    ax.axhline(0, color="black", lw=1)
    for i, value in enumerate(y):
        ax.annotate(f"{value:+.2f}" if scale == 100 else f"{value:+.4f}",
                    (i, ci[i, 0] if value < 0 else ci[i, 1]),
                    xytext=(0, -14 if value < 0 else 7),
                    textcoords="offset points", ha="center", fontsize=10,
                    color="#333333")
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)
    ax.margins(y=.25)


def figures(result):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    main = result["confirmation"]["pooled_reject"]["bins"]
    labels = ["[0, .25)", "[.25, .50)", "[.50, .75)", "[.75, 1]"]
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
    xx = np.arange(4)
    for offset, name, color in [(-.18, "residual", COLORS[0]), (.18, "proxy", COLORS[1])]:
        key = "coverage_" + name
        yy = np.array([b[key]["mean"] for b in main]) * 100
        ci = np.array([b[key]["ci95"] for b in main]) * 100
        axs[0].bar(xx + offset, yy, .34, label=name.title(), color=color)
        axs[0].errorbar(xx + offset, yy, yerr=[yy - ci[:, 0], ci[:, 1] - yy],
                       fmt="none", ecolor="#333333", capsize=3, lw=.8)
        for i, y in enumerate(yy):
            axs[0].text(i + offset, ci[i, 1] + 2, f"{y:.1f}", ha="center", fontsize=9)
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("Correction coverage, top-3 (%)")
    axs[0].set_title("Both candidate policies")
    axs[0].legend(frameon=False)
    bars(axs[1], main, "coverage_gap", 100,
         [COLORS[0] if b["coverage_gap"]["mean"] > 0 else COLORS[1] for b in main])
    axs[1].set_ylabel("Residual - proxy coverage (pp)")
    axs[1].set_title("Paired difference")
    axs[1].set_ylim(min(b["coverage_gap"]["ci95"][0] for b in main) * 100 - 4, 3)
    for ax in axs:
        ax.set_xticks(xx, labels)
        ax.set_xlabel("Target-draft distance: TV(p_T, p_D)\ncloser on left; farther on right")
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    shares=" / ".join(f'{100*b["event_mass_fraction"]:.1f}%' for b in main)
    fig.suptitle("Target-draft distance vs root candidate coverage | exit 56, T = 1")
    fig.supxlabel("128 prompts (source rows 33-64 per task), 6,036 correction contexts | pointwise 95% prompt CI\n"
                  + "First-rejection probability weights; bin mass: " + shares
                  + " | offline coverage", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"04_tv_coverage.{ext}", dpi=180)
    plt.close(fig)

    # Same contexts, same bins, same weights: metric comparison with Fig.6 x-axis.
    ordered = list(reversed(main))
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.8), layout="constrained")
    for ax, key, scale, title in zip(
            axs, ["coverage_gap", "alignment_gap"], [100, 1],
            ["Top-3 coverage difference (pp)", "Full-vocabulary alignment difference"]):
        bars(ax, ordered, key, scale,
             [COLORS[0] if b[key]["mean"] > 0 else COLORS[1] for b in ordered])
        ax.set_xticks(xx, ["0-.25", ".25-.50", ".50-.75", ".75-1"])
        ax.set_xlabel("Target-draft overlap = 1 - TV(p_T, p_D)\nfarther on left; closer on right")
        ax.set_title(title)
        ax.set_ylabel("Residual - proxy")
    axs[0].set_ylim(top=3)
    fig.suptitle("Same 128 prompts and weighting: coverage vs alignment")
    fig.supxlabel("P1 off, exit 56, T = 1 | pooled first-rejection weights | pointwise 95% prompt CI\n"
                  "Alignment: dot product with true correction distribution; not top-k cache hit", fontsize=9)
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"05_overlap_coverage_alignment.{ext}", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-alignment", action="store_true")
    args = ap.parse_args()
    result = {}
    hashes = {}
    for group in ["development", "confirmation"]:
        records = load_records(group)
        for rec in records:
            hashes[str(rec["metric_file"].relative_to(ROOT))] = digest(rec["metric_file"])
            if group == "confirmation":
                rec["alignment"] = alignment(rec, args.reuse_alignment)
        result[group] = {mode: summarize(records, mode, group == "confirmation")
                         for mode in ["pooled_reject", "step_normalized"]}
    direct = json.loads((DIRECT / "confirmation_summary.json").read_text())
    main_group = result["confirmation"]["pooled_reject"]["bins"]
    recovered = sum(b["event_mass_fraction"] * b["coverage_gap"]["mean"] for b in main_group)
    old = direct["event_groups"]["all"]["means"]
    np.testing.assert_allclose(recovered, old["residual"] - old["proxy"], atol=1e-12, rtol=0)
    result["metadata"] = dict(
        source_sha256=hashes, bootstrap_replicates=REPS, bootstrap_seed=20261001,
        local_k=3, x_axis="TV(p_T,p_D) = sum positive(p_T-p_D), up to float32 roundoff",
        provenance="development: source rows 1-32, seeds42/123; confirmation: rows33-64, seed7",
        confidence_intervals="Pointwise 95% percentile paired prompt-cluster bootstrap, stratified by dataset; exploratory",
        coverage="Stored GPU top-3 outcomes; sampled draft token excluded; bonus excluded",
        alignment="CPU float64 dot products of true R with normalized residual/proxy after y exclusion",
        weighting="Primary pooled h (as prior conditional tables); sensitivity h normalized within each step. Neither is observed cache hit.",
        limitations=["Dataset subsets, not full datasets.", "Same 128 confirmation contexts for both metrics.",
                     "Initial Sep9 contexts are not reused; development is Sep10 replay.",
                     "Bin boundaries fixed at 0/.25/.5/.75/1; no threshold or policy fitted.",
                     "Not global15 coverage or P1-on readiness. No inference or GPU top-k rerun."])
    (HERE / "distance_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for group in ["development", "confirmation"]:
        for mode, info in result[group].items():
            for b in info["bins"]:
                for metric in info["metric_names"] + ["coverage_gap"] + (["alignment_gap"] if group == "confirmation" else []):
                    r = b[metric]
                    rows.append(dict(group=group, weighting=mode, tv_low=b["tv_low"], tv_high=b["tv_high"],
                                     metric=metric, mean=r["mean"], ci_low=r["ci95"][0], ci_high=r["ci95"][1],
                                     mass=b["event_mass_fraction"], contexts=b["positive_h_contexts"],
                                     prompts=b["contributing_prompts"]))
    with (HERE / "distance_tables.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(result)
    print("Saved distance figures and tables. Conditional coverage matches prior summary.", flush=True)
    for group in ["development", "confirmation"]:
        for b in result[group]["pooled_reject"]["bins"]:
            print(group, b["tv_low"], b["coverage_gap"],
                  b.get("alignment_gap", ""), flush=True)


if __name__ == "__main__":
    main()
