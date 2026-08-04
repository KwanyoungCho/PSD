#!/usr/bin/env python3
"""E0 체인 baseline: 예산 슬롯의 P_iv↔hit 분석 (docs/duet/17 §3.2 ⑥).

사용: SSD_HF_CACHE=... SSD_DATASET_DIR=... python e0_budget_piv_analysis.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import e1_arms
import statistics as st
from collections import defaultdict


def main():
    steps = e1_arms.load_steps()
    slots = []
    for s in steps:
        ret = sorted(s["retained"], key=lambda x: -x[2])
        out = s["outcome"]
        for i, (rk, seed, piv) in enumerate(ret):
            slots.append((piv, 1 if seed == out else 0, i))
    tot = sum(h for _, h, _ in slots)
    print(f"steps={len(steps)} slots={len(slots)} step당 hit={tot/len(steps):.3f}")
    print("\n[P_iv 구간별] 점유/실측hit/기여")
    for lo, hi in [(0.3, 1), (0.1, 0.3), (0.03, 0.1), (0.01, 0.03),
                   (0.003, 0.01), (0.001, 0.003), (0, 0.001)]:
        ss = [(p, h) for p, h, _ in slots if lo <= p < hi]
        if ss:
            print(f"[{lo:g},{hi:g}): {len(ss)/len(slots):.1%} "
                  f"{sum(h for _, h in ss)/len(ss):.4f} "
                  f"{sum(h for _, h in ss)/tot:.1%}")
    print("\n[예산 내 순위별] hit/piv중앙/누적커버")
    byr = defaultdict(list)
    for p, h, i in slots:
        byr[i].append((p, h))
    cum = 0
    for i in range(10):
        ps = [p for p, _ in byr[i]]
        hs = [h for _, h in byr[i]]
        cum += sum(hs)
        print(f"{i+1}: {st.mean(hs):.3f} {st.median(ps):.4f} {cum/tot:.1%}")
    print("\n[사망 슬롯]")
    for th in (0.03, 0.01, 0.003, 0.001):
        d = [(p, h) for p, h, _ in slots if p < th]
        print(f"P_iv<{th:g}: 점유 {len(d)/len(slots):.1%}, "
              f"기여 {sum(h for _, h in d)/tot:.1%}")


if __name__ == "__main__":
    main()
