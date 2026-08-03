"""Phase A6 — nsys 커널-수준 diff: 10행 vs 5행 verify 그래프의 행당 비용
을 커널 이름 단위로 분해 (19번 트랙; --cuda-graph-trace=node 리포트).

방법:
1. sqlite에서 device 0(rank0)의 커널 인스턴스 (이름, grid, 시각, 시간)
   로드.
2. **버킷 분류**: M-의존 grid를 가진 marker 커널(두 개의 grid 변형이
   뚜렷한 것)로 각 verify replay 창을 10행/5행으로 라벨.
3. 커널 이름별로 두 버킷의 replay당 시간 합 diff → "행당 µs"의 범인
   순위표. NCCL 커널의 시간에는 rank 대기가 포함되므로 straggler
   가설이 맞으면 nccl 항목이 상위로 나온다.

Run: cd ssd && python experiments/proxy_async_overlap/e2_micro/a6_nsys_kernel_diff.py <report.sqlite>
"""
import collections
import sqlite3
import sys


def main(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    ktab = next(t for t in tabs if "KERNEL" in t)
    print(f"[db] kernel table: {ktab}")
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({ktab})")]
    print(f"[db] cols: {cols}")
    namecol = "shortName" if "shortName" in cols else "demangledName"
    rows = cur.execute(
        f"SELECT start, end, {namecol}, gridX, gridY, gridZ, deviceId "
        f"FROM {ktab} WHERE deviceId=0 ORDER BY start").fetchall()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    print(f"[db] device0 kernels: {len(rows)}")

    ks = [(s, e, strings.get(n, str(n)), (gx, gy, gz))
          for s, e, n, gx, gy, gz, _ in rows]

    # verify replay 창 분할: 큰 시간 gap (> 2ms) 을 경계로 클러스터
    windows = []
    cur_w = [ks[0]]
    for prev, nxt in zip(ks, ks[1:]):
        if nxt[0] - prev[1] > 2_000_000:          # 2ms gap (ns)
            windows.append(cur_w)
            cur_w = []
        cur_w.append(nxt)
    windows.append(cur_w)
    big = [w for w in windows if len(w) > 300]     # 80층 그래프 창만
    print(f"[cluster] windows: {len(windows)}, 80층-급(>300커널): {len(big)}")

    # marker 자동 탐지: 창들 사이에서 grid 변형이 정확히 2종으로 갈리는
    # 커널 이름 찾기 (Marlin은 M-불변 grid라 부적합 — elementwise류가
    # M∝grid). 창별로 각 이름의 grid 집합을 만들고, 2-변형 이름 다수결.
    name_grid_by_win = []
    for w in big:
        d = collections.defaultdict(set)
        for _, _, nm, grid in w:
            d[nm].add(grid)
        name_grid_by_win.append(d)
    cand = collections.defaultdict(collections.Counter)
    for d in name_grid_by_win:
        for nm, gs in d.items():
            if len(gs) == 1:
                cand[nm][next(iter(gs))] += 1
    # spec-step 창만 남기기: silu(M∝grid) 최빈 grid 2종 = 10행/5행 버킷
    # (그 외 grid = prefill/캡처 창 → 제외)
    mk_name = "triton_poi_fused_mul_silu_0"
    if mk_name not in cand:
        mk_name = max(cand, key=lambda nm: sum(cand[nm].values()))
    top2 = [g for g, _ in cand[mk_name].most_common(2)]
    print(f"[marker] '{mk_name}' grid 분포: "
          f"{dict(cand[mk_name].most_common(6))} → 버킷 grid {top2}")
    g_a, g_b = top2
    wa = [w for w, d in zip(big, name_grid_by_win)
          if d.get(mk_name) == {g_a}]
    wb = [w for w, d in zip(big, name_grid_by_win)
          if d.get(mk_name) == {g_b}]
    print(f"[cluster] marker='{mk_name[:50]}' grids {g_a} vs {g_b} → "
          f"창 {len(wa)} vs {len(wb)}")
    if not wa or not wb:
        print("버킷 분리 실패")
        return

    def per_window_kernel_sums(ws):
        agg = collections.defaultdict(list)
        for w in ws:
            c = collections.defaultdict(int)
            for s, e, nm, _ in w:
                c[nm] += e - s
            for nm, tot in c.items():
                agg[nm].append(tot)
        return {nm: sum(v) / len(v) / 1000 for nm, v in agg.items()}  # µs

    A, B = per_window_kernel_sums(wa), per_window_kernel_sums(wb)
    dur_a = sum(A.values())
    dur_b = sum(B.values())
    span_a = sum(w[-1][1] - w[0][0] for w in wa) / len(wa) / 1000
    span_b = sum(w[-1][1] - w[0][0] for w in wb) / len(wb) / 1000
    hi, lo = (A, B) if dur_a > dur_b else (B, A)
    print(f"\n[창 요약] 그룹A: n={len(wa)} 커널합 {dur_a:.0f}µs 창길이 {span_a:.0f}µs")
    print(f"          그룹B: n={len(wb)} 커널합 {dur_b:.0f}µs 창길이 {span_b:.0f}µs")
    print(f"  창길이 차 {abs(span_a-span_b):.0f}µs vs 커널합 차 "
          f"{abs(dur_a-dur_b):.0f}µs (차이나면 idle gap 성분)")

    print("\n[커널별 diff 상위 15] (10행버킷 − 5행버킷, replay당 µs)")
    names = set(hi) | set(lo)
    diffs = sorted(((hi.get(nm, 0) - lo.get(nm, 0), nm) for nm in names),
                   reverse=True)[:15]
    for d, nm in diffs:
        print(f"  {d:+9.1f} µs  {nm[:90]}")


if __name__ == "__main__":
    main(sys.argv[1])
