#!/usr/bin/env python3
"""23번 단계2 — 동적 실행 로그에서 대표 고정트리 1개 추출 (오프라인).

입력: SSD_TREE_TOPO_TRACE 접두어의 .draft.jsonl(rollout topology)
+ .walk.jsonl(서빙 root 보행). sweep이 아니라 로그 통계 기반 설계
(리뷰7: root 순위별 평균 예산 / 깊이별 선택 빈도 / 형제 수락 기여 /
빈발 부모·자식 구조).

출력: 대표 템플릿 JSON — root rank별 (par, sib) 고정 서브트리
(합 예산 ≤ F·W, root당 ≤ Nv) + 근거 통계.

사용: python design_static_tree.py <trace_prefix> [--out template.json]
"""
import argparse
import json
from collections import Counter, defaultdict


def canon_shape(par, sib, nodes):
    """root 서브트리의 정규형 문자열 (노드 리스트 → (par,sib) 서명)."""
    idx = {n: i for i, n in enumerate(nodes)}
    out = []
    for n in nodes:
        p = par[n]
        out.append((idx.get(p, -1), sib[n]))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-total", type=int, default=40)
    ap.add_argument("--cap", type=int, default=8)
    args = ap.parse_args()

    # ── draft 로그: root rank별 예산·형상 통계
    per_rank_budget = defaultdict(list)
    per_rank_shapes = defaultdict(Counter)
    per_rank_shape_nodes = defaultdict(dict)
    depth_hist = Counter()
    n_roll = 0
    with open(args.prefix + ".draft.jsonl") as f:
        for line in f:
            d = json.loads(line)
            n_roll += 1
            par, root, dep, sib = d["par"], d["root"], d["depth"], d["sib"]
            kids_of_rank = defaultdict(list)
            for i in range(len(par)):
                if par[i] >= 0:
                    kids_of_rank[root[i]].append(i)
                    depth_hist[dep[i]] += 1
            for r, nodes in kids_of_rank.items():
                per_rank_budget[r].append(len(nodes))
                sig = canon_shape(par, sib, nodes)
                per_rank_shapes[r][sig] += 1
                per_rank_shape_nodes[r].setdefault(sig, (par, sib, nodes))

    # ── walk 로그: 수락 기여 (깊이·형제 순서)
    acc_by_depth = Counter()
    acc_by_sib = Counter()
    walk_n = 0
    try:
        with open(args.prefix + ".walk.jsonl") as f:
            for line in f:
                w = json.loads(line)
                walk_n += 1
                par, sib, path = w["par"], w["sib"], w["path"]
                dep = {}
                for j in range(len(par)):
                    dep[j] = dep.get(par[j], 0) + 1
                for j in path:
                    acc_by_depth[dep[j]] += 1
                    acc_by_sib[sib[j]] += 1
    except FileNotFoundError:
        pass

    # ── rank별 대표 예산 (중앙값) → cap·총예산 안에서 조정
    ranks = sorted(per_rank_budget)
    med = {}
    for r in ranks:
        v = sorted(per_rank_budget[r])
        med[r] = min(args.cap, v[len(v) // 2])
    # 총합 초과 시 낮은 rank부터 감축 (관측상 낮은 rank 예산이 얇음)
    while sum(med.values()) > args.budget_total:
        r = max(ranks, key=lambda x: (med[x], x))
        med[r] -= 1

    # ── rank×예산 대표 형상: 해당 rank에서 그 예산으로 가장 빈발한 형상
    template = {}
    for r in ranks:
        want = med[r]
        best = None
        for sig, cnt in per_rank_shapes[r].most_common():
            if len(sig) == want:
                best = sig
                break
        if best is None and per_rank_shapes[r]:
            best = per_rank_shapes[r].most_common(1)[0][0]
        if best is None:
            continue
        template[str(r)] = {"par": [p for p, _s in best],
                            "sib": [s for _p, s in best]}

    report = {
        "rollouts": n_roll,
        "walks": walk_n,
        "per_rank_budget_median": {str(r): med[r] for r in ranks},
        "per_rank_budget_mean": {
            str(r): round(sum(v) / len(v), 2)
            for r, v in per_rank_budget.items()},
        "depth_hist": dict(sorted(depth_hist.items())),
        "accept_by_depth": dict(sorted(acc_by_depth.items())),
        "accept_by_sib": dict(sorted(acc_by_sib.items())),
        "shape_coverage": {
            str(r): round(per_rank_shapes[r].most_common(1)[0][1]
                          / max(1, sum(per_rank_shapes[r].values())), 3)
            for r in ranks if per_rank_shapes[r]},
        "template": template,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
