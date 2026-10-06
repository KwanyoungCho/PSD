"""Standalone paper-style figures. Run analyze_results.py first."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"figs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
COLORS = {"ar": "#777777", "ssd": "#D55E00", "duet-chain": "#0072B2"}


def save(fig, name):
    fig.savefig(OUT/(name+".png"), dpi=180, bbox_inches="tight")
    fig.savefig(OUT/(name+".pdf"), bbox_inches="tight")
    plt.close(fig)


def actual_tps(run, cell):
    count = (sum(len(o["token_ids"]) for o in cell["outputs"])
             if run["args"]["mode"] != "ar" else cell["metrics"]["decode_total_tokens"])
    return count/cell["metrics"]["decode_total_time"]


def main():
    groups = json.loads((ROOT/"FINAL_NUMBERS.json").read_text())["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, model, title in zip(axes, ["llama2", "llama3"], ["LayerSkip-Llama2-7B + AMD-135M", "LayerSkip-Llama3-8B + Qwama-0.5B"]):
        candidate = "width4" if model == "llama2" else "width1"
        keys = ["ar", "baseline", "fast", candidate, "ssd"]
        labels = ["AR*", "DUET\ncorrected base", "DUET\ncode only", "DUET\nF1="+("4" if model=="llama2" else "1"), "SSD\nK=3, F=2"]
        vals = [groups[model+"/"+k]["decode_tps"]["mean"] for k in keys]
        err = [groups[model+"/"+k]["decode_tps"]["sd"] or 0 for k in keys]
        bars = ax.bar(labels, vals, yerr=err, capsize=3, color=["#777777", "#88AACC", "#0072B2", "#009E73", "#D55E00"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+12, f"{v:.1f}", ha="center", fontsize=9)
        ax.set_ylim(0, max(vals)*1.16)
        ax.set_ylabel("Emitted decode tokens / second")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("B=8, T=0.7 | all 480 available first-turn inputs | max output 64", fontsize=12)
    fig.text(.5, -.04, "Mean ± sample SD over 3 fresh-process seeds. *AR: 1 run, 1 GPU; speculative modes: 2 GPUs.\nInput cap 512; EOS enabled; discarded tokens excluded. Width changes are measured candidates, not universal recommendations.", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "01_full_corpus_tps")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    padding = {}
    complete = True
    for r, model in enumerate(["llama2", "llama3"]):
        for mode in ["ar", "ssd", "duet-chain"]:
            path = ROOT/"validation"/(model+"_"+mode+"_scale.json")
            if not path.exists():
                complete = False
                continue
            run = json.loads(path.read_text())
            if run["status"] != "complete":
                complete = False
                continue
            for col, temp in enumerate([0., .7]):
                cells = sorted([c for c in run["cells"] if c["temperature"] == temp], key=lambda c:c["batch"])
                axes[r,col].plot([c["batch"] for c in cells], [actual_tps(run,c) for c in cells],
                                  "o-", color=COLORS[mode], label=mode.upper())
                if mode == "duet-chain" and temp == .7:
                    rows=[]
                    for c in cells:
                        evs=c["metrics"]["phase_events"]; steps={}
                        for e in evs:steps.setdefault(e["step_id"],[]).append(e)
                        rows.append((c["batch"], 100*(1-sum(e["valid_k"]+1 for e in evs)/sum(e["verify_width"]+1 for e in evs)),
                                     100*sum(any(not e["cache_hit"] for e in es) for es in steps.values())/len(steps)))
                    padding[model]=rows
        for col,temp in enumerate([0.,.7]):
            ax=axes[r,col];ax.set_title(model+f" | T={temp:g}");ax.set_xticks([1,2,4,8]);ax.grid(alpha=.2)
            ax.set_ylabel("Emitted decode tokens / second");ax.set_xlabel("Maximum active requests (B)")
            ax.legend(fontsize=8)
    if complete:
        fig.suptitle("Batch scaling | 48 inputs, max output 64 | general sampler", fontsize=12)
        fig.tight_layout();save(fig,"02_batch_scaling")
    else:
        plt.close(fig)
    if len(padding)==2:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for model,rows in padding.items():
            axes[0].plot([r[0] for r in rows],[r[1] for r in rows],"o-",label=model)
            axes[1].plot([r[0] for r in rows],[r[2] for r in rows],"o-",label=model)
        for ax in axes:
            ax.set_xticks([1,2,4,8]);ax.set_xlabel("B");ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel("Unused query positions (%)");axes[0].set_title("K1/K2 padding (excludes batch-bucket padding)")
        axes[1].set_ylabel("Steps containing a cache miss (%)");axes[1].set_title("One miss can stall the batch")
        fig.suptitle("DUET K1=4, K2=2, F1=2 | T=0.7 | same 48-input workload",fontsize=11)
        fig.tight_layout();save(fig,"03_batch_bottlenecks")


if __name__ == "__main__":
    main()
