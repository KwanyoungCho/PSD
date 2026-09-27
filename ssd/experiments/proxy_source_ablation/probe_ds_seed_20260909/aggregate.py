"""Aggregate the dataset x seed probe runs."""
import glob, json, os, statistics as st

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
runs = {}
for f in sorted(glob.glob(f"{OUT}/*.json")):
    tag = os.path.basename(f)[:-5]
    ds, seed = tag.rsplit("_seed", 1)
    runs.setdefault(ds, {})[int(seed)] = json.load(open(f))

any_d = next(iter(next(iter(runs.values())).values()))
B, L, H, R = any_d["budgets"], any_d["layers"], any_d["hpol"], any_d["rsrc"]
bi, EL = B.index(15), any_d["exit_layer"]


def cell(d, l, h, r, b=15):
    return d["hit"][L.index(l)][H.index(h)][R.index(r)][B.index(b)]


def show(title, fn, fmt="{:+.4f}"):
    print(f"\n=== {title} ===")
    print(f"{'dataset':>10s} " + " ".join(f"{s:>9d}" for s in sorted(
        next(iter(runs.values())).keys())) + f" {'mean':>9s} {'range':>8s}")
    for ds in runs:
        vs = [fn(runs[ds][s]) for s in sorted(runs[ds])]
        rng = (max(vs) - min(vs)) / 2 if len(vs) > 1 else 0.0
        print(f"{ds:>10s} " + " ".join(fmt.format(v) for v in vs)
              + f" {fmt.format(st.mean(vs)):>9s} {rng:8.4f}")


print(f"probe: exit_layer={EL}  R=15  layers={len(L)}  "
      f"steps/run={[d['n_steps'] for d in next(iter(runs.values())).values()]}")

show("수식의 이론적 이득  htrue x rT − htrue x pT  (참값, ε=0)",
     lambda d: cell(d, L[0], "htrue", "rT") - cell(d, L[0], "htrue", "pT"))
show(f"실제 격차  hhat x rE − hhat x pE  (layer {EL})",
     lambda d: cell(d, EL, "hhat", "rE") - cell(d, EL, "hhat", "pE"))
show(f"위치 배분 통제  unif x rE − unif x pE  (layer {EL})",
     lambda d: cell(d, EL, "unif", "rE") - cell(d, EL, "unif", "pE"))
show(f"근사 품질차  TVD(residual형) − TVD(proxy형)  (layer {EL}, +면 residual이 나쁨)",
     lambda d: d["approx"][L.index(EL)][0] - d["approx"][L.index(EL)][1])
show("교차 레이어 (rE가 pE를 역전하는 최소 층)",
     lambda d: next((l for l in L if cell(d, l, "hhat", "rE")
                     > cell(d, l, "hhat", "pE")), 99), "{:9.0f}")
show(f"p^E 엔트로피 − p_T 엔트로피 (layer {EL})",
     lambda d: d["approx"][L.index(EL)][5] - d["approx"][L.index(EL)][6])

print("\n=== 겹침 구간별 (residual − proxy 정렬도), 전 run 평균 ===")
lab = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
acc = [[0.0, 0.0, 0.0] for _ in range(4)]
for ds in runs:
    for s in runs[ds]:
        bs = runs[ds][s]["bins"]
        tot = sum(b[0] for b in bs)
        for k, b in enumerate(bs):
            if b[0] > 0:
                acc[k][0] += b[0] / tot
                acc[k][1] += b[1] / b[0]
                acc[k][2] += b[2] / b[0]
n = sum(len(v) for v in runs.values())
print(f"{'겹침':>12s} {'질량비중':>8s} {'residual':>9s} {'proxy':>8s} {'차이':>8s}")
for k in range(4):
    m, r, p = acc[k][0] / n, acc[k][1] / n, acc[k][2] / n
    print(f"{lab[k]:>12s} {m:8.3f} {r:9.4f} {p:8.4f} {r - p:+8.4f}")
