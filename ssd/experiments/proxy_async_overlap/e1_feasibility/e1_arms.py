"""E1 feasibility — step 2: comparison arms (docs/duet/18 §2).

Per design v6 (§9 E1): feasibility 사전 점검 — 결과는 보고, 판정은
사용자. 모든 수치는 같은 timing 모델(같은-런 검증 −0.21%, 18번 §1)을
공유하는 **상대 비교**로 읽는다.

정확 성분 vs 모델 성분 (정직 표기):
- EXACT  : root 선택 효과 — 실제 outcome이 dedup-후 잔존 seed 상위
           R개 안에 있는지 (miss→P2hit / P2hit→miss 전이 포함, full
           step-status replay).
- MODEL  : counterfactual 트리의 수락 길이 — 실측 레벨별 수락률
           α_d (P2-hit step에서 적합)로 level 생존 1−(1−α_d)^f_d
           (형제 독립 근사 — 낙관 편향 가능; 민감도 병기).
- PLACEHOLDER: verify 행당 비용 1.9ms (체인 스윕 값 — E2①이 실측
           대체 예정; 1.0ms 민감도 병기).

Run:
  cd ssd && python experiments/proxy_async_overlap/e1_feasibility/e1_arms.py
"""
import glob
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E0 = os.path.join(ROOT, "e0_collect", "run1")
PAIRED = os.path.join(ROOT, "e0_collect", "run2_paired")
D_CAP = 4                       # K2 = 트리 깊이 상한 (설계 D7: D=4 우선)
ROW_MS_LIST = (1.9, 1.0)        # verify 행당 비용 (placeholder + 민감도)


def load_period():
    recs = json.load(open(glob.glob(
        PAIRED + "/duet_profile_target_rank0_*.json")[0]))
    start, status = {}, {}
    for r in recs:
        sid = r.get("step_id")
        if sid is None or r.get("start_ms") is None:
            continue
        if sid not in start or r["start_ms"] < start[sid]:
            start[sid] = r["start_ms"]
        if r.get("status") in ("miss", "hit_k1", "hit_k2"):
            status[sid] = r["status"]
    per = {}
    for sid in sorted(start):
        nxt, stt = start.get(sid + 1), status.get(sid)
        if nxt is None or stt is None or sid <= 10:
            continue
        d = nxt - start[sid]
        if 0 < d < 250:
            per.setdefault(stt, []).append(d)
    return {k: st.mean(v) for k, v in per.items()}


def load_slack():
    """짝런 draft profile의 draft_recv_cmd(작업후 대기) 분포 per status.
    delay 기대치 E[max(0, Δ−slack)]를 분포로 계산 (평균-slack만 쓰면
    볼록성 때문에 delay를 과소평가)."""
    recs = json.load(open(glob.glob(
        PAIRED + "/duet_profile_draft_*.json")[0]))
    by = {}
    for r in recs:
        if r.get("label") == "draft_recv_cmd" and r.get("status") \
                and r.get("step_id", 0) and r["step_id"] > 10:
            by.setdefault(r["status"], []).append(
                r["end_ms"] - r["start_ms"])
    return by


def delay_fn(slack_by, status, delta):
    v = slack_by.get(status)
    if not v:
        return delta
    return sum(max(0.0, delta - x) for x in v) / len(v)


def load_steps():
    """step → (phase, outcome(pos,tok), retained[(pos,tok)...] rank순)."""
    req, resp, sel = {}, {}, {}
    for line in open(glob.glob(E0 + "/e0_draft_*.jsonl")[0]):
        r = json.loads(line)
        if r["kind"] == "request":
            c = r["cache_keys"][0]
            req[r["step_id"]] = (c[0], c[1], c[2])
        elif r["kind"] == "response":
            resp[r["step_id"]] = r["phase_source"][0]
        elif r["kind"] == "selector":
            # retained seeds: fan_out[p]개씩 position-그룹 순서.
            fo = r["proxy_fan_out"][0]
            toks = r["proxy_forked"][0]
            seeds, i = [], 0
            for p, c in enumerate(fo):
                for _ in range(c):
                    if i < len(toks):
                        seeds.append((p, toks[i]))
                        i += 1
            # wire(rank) 순서로 재정렬: wire 기록과 join해야 하나,
            # 잔존 seed의 rank는 target wire에서 복원한다 (아래).
            sel[r["step_id"]] = seeds
    wires = {}
    for line in open(glob.glob(E0 + "/e0_target_*.jsonl")[0]):
        r = json.loads(line)
        if r["kind"] != "wire":
            continue
        wires[r["n"]] = (r["chosen_pos"][0], r["chosen_tok"][0],
                        r["piv"][0])
    steps = []
    for t, ph in resp.items():
        # 파이프라인 정렬 (핵심): step t의 응답은 t-1에 만든 캐시에서
        # 나오고, 그 조회 키(직전 outcome)는 req[t]에 실려 온다.
        # verify(t)의 커밋 결과는 req[t+1]에 실려 온다.
        key = req.get(t)                 # 캐시 조회 키 = (seq, fan, rec)
        nxt = req.get(t + 1)             # verify(t) 결과
        w = wires.get(t - 1)             # t-1의 wire (캐시의 원천)
        seeds = sel.get(t - 1)           # t-1의 잔존 seed
        if key is None or nxt is None or key[0] != nxt[0] or w is None \
                or seeds is None:
            continue
        committed = nxt[1] + 1
        lookup = (key[1], key[2])        # (pos, tok) — 캐시 키 좌표
        pos, tok, piv = w
        rank_of = {(p, tk): i for i, (p, tk) in enumerate(zip(pos, tok))}
        ranked = sorted(
            [(rank_of.get(s, 10**9), s) for s in seeds])
        retained = [(rk, s, piv[rk] if rk < len(piv) else 0.0)
                    for rk, s in ranked if rk < 10**9]
        steps.append(dict(phase=ph, committed=committed,
                          outcome=lookup, retained=retained,
                          wire=list(zip(pos, tok, piv))))
    return steps


def fit_alpha(steps):
    """P2-hit step의 committed(=1+수락 연속 L)에서 레벨별 수락률 적합:
    α_d = P(L≥d+1 | L≥d), d=0..D_CAP-1."""
    Ls = [s["committed"] - 1 for s in steps if s["phase"] == 2]
    alpha = []
    for d in range(D_CAP):
        ge_d = sum(1 for L in Ls if L >= d)
        ge_d1 = sum(1 for L in Ls if L >= d + 1)
        alpha.append(ge_d1 / ge_d if ge_d else 0.0)
    return alpha, st.mean(Ls)


def tree_L(budget, alpha, lam=1.0):
    """budget 노드를 D_CAP 레벨에 앞-우선 균등 배분한 트리의 기대 수락
    연속 길이. lam = 형제 상관 할인 (1=독립 근사(낙관), 0=완전 상관
    (형제 무가치, 체인과 동일 — 보수 하한)): 유효 fanout = 1+lam(f−1)."""
    base, rem = divmod(budget, D_CAP)
    fo = [base + (1 if i < rem else 0) for i in range(D_CAP)]
    L, surv = 0.0, 1.0
    for d in range(D_CAP):
        if fo[d] <= 0:
            break
        f_eff = 1.0 + lam * (fo[d] - 1)
        a = 1.0 - (1.0 - alpha[d]) ** f_eff
        surv *= a
        L += surv
    return L


def main():
    period = load_period()
    steps = load_steps()
    alpha, L_chain_obs = fit_alpha(steps)
    n = len(steps)
    print(f"[data] steps={n} | α(레벨별, P2-hit 적합)="
          f"{[round(a, 3) for a in alpha]} | 관측 체인 L_p2={L_chain_obs:.3f}"
          f" | 모델 재현 L(chain b=4)={tree_L(4, alpha):.3f} (자기검증)")
    print(f"[period(짝런)] { {k: round(v, 2) for k, v in period.items()} }")

    # 자기검증: R=10 판정이 실제 phase==2와 일치해야 함 (EXACT 성분 검증)
    agree = mism = 0
    for s_ in steps:
        if s_["phase"] == 1:
            continue
        m = any(seed == s_["outcome"] for _, seed, _ in s_["retained"][:10])
        if (s_["phase"] == 2) == m:
            agree += 1
        else:
            mism += 1
    print(f"[자기검증] R=10 판정 vs 실제 phase: 일치 {agree}, 불일치 {mism} "
          f"({mism/(agree+mism):.2%})")

    name = {0: "miss", 1: "hit_k1", 2: "hit_k2"}

    NODE_BUDGET = 40                  # 실제 P2 노드 예산 = 10행 × 깊이4

    def roots_for(s, R):
        """counterfactual root 목록 (rank, (pos,tok), piv) 상위 R.
        R ≤ len(retained): 현행 selector 의미론 그대로.
        R > len(retained): wire의 budget-cut 후보(rank > retained 마지막
        rank — dedup 아님이 보장되는 구간)로 확장. EXACT."""
        ret = s["retained"]
        if R <= len(ret):
            return ret[:R]
        out = list(ret)
        if ret:
            last_rk = ret[-1][0]
            for rk, (p, tk, pv) in enumerate(s["wire"]):
                if rk > last_rk and len(out) < R:
                    out.append((rk, (p, tk), pv))
        return out[:R]

    def run_arm(label, R, row_ms, nv_cap=8, tree=True, oracle_alloc=False):
        tot_tok = tot_ms = 0.0
        trans = {"miss->p2": 0, "p2->miss": 0}
        p2 = 0
        for s in steps:
            ph, committed = s["phase"], s["committed"]
            if ph == 1:
                tot_tok += committed
                tot_ms += period["hit_k1"]
                continue
            roots = roots_for(s, R)
            hit_rank = None
            for i, (rk, seed, piv) in enumerate(roots):
                if seed == s["outcome"]:
                    hit_rank = i
                    break
            if hit_rank is not None:
                p2 += 1
                if ph == 0:
                    trans["miss->p2"] += 1
                if not tree:                       # 현행: root당 깊이4 체인
                    L, nv = L_chain_obs, 4
                elif oracle_alloc:                 # 사후 최적: 예산 전부
                    L, nv = tree_L(nv_cap, alpha), nv_cap
                else:                              # 결정⑤v2: P_iv-비례
                    piv_sel = [max(pv, 1e-6) for _, _, pv in roots]
                    b = max(1, round(NODE_BUDGET * piv_sel[hit_rank]
                                     / sum(piv_sel)))
                    nv = min(b, nv_cap)
                    L = tree_L(nv, alpha)
                tot_tok += 1 + L
                tot_ms += period["hit_k2"] + max(0, nv - 4) * row_ms
            else:
                if ph == 2:
                    trans["p2->miss"] += 1
                tot_tok += committed if ph == 0 else 2.77
                tot_ms += period["miss"]
        tps = tot_tok / (tot_ms / 1000.0)
        print(f"  {label:40s} tok/step={tot_tok/n:.3f} TPS={tps:6.2f} "
              f"| P2hit {p2/n:.3f} | 전이 {trans}")
        return tps

    for row_ms in ROW_MS_LIST:
        print(f"\n=== 행당 비용 {row_ms}ms (PLACEHOLDER — E2① 실측 대체 예정) ===")
        base = run_arm("ⓐ 현행 체인 재현 (R=10, 깊이4)", 10, row_ms,
                       tree=False)
        for nv in (6, 8):
            for R in (8, 10, 14, 20):
                run_arm(f"트리 R={R}, N_v={nv}, budget∝P_iv", R, row_ms,
                        nv_cap=nv)
        run_arm("ⓔ oracle: R=20 + 사후최적 배분, N_v=8", 20, row_ms,
                nv_cap=8, oracle_alloc=True)
        print(f"  (기준: ⓐ={base:.2f} — 상대 비교로 읽을 것)")

    # ---------------- Step 3 ----------------
    slack_by = load_slack()
    print(f"\n[slack 분포] n per status: "
          f"{ {k: len(v) for k, v in slack_by.items()} }")

    def run_arm3(label, R, row_ms=1.9, nv_cap=8, lam=1.0, beta=1.0,
                 calib_posK=1.0, delta_draft=0.0):
        """Step-3 변형: lam(형제 상관), beta(배분 지수), calib_posK
        (pos=K 후보 P_iv 보정 배수 — root 정렬·배분 모두 적용),
        delta_draft(draft 증분 비용 ms — status별 slack 분포로 delay)."""
        tot_tok = tot_ms = 0.0
        for s in steps:
            ph, committed = s["phase"], s["committed"]
            if ph == 1:
                tot_tok += committed
                tot_ms += period["hit_k1"] + delay_fn(
                    slack_by, "hit_k1", delta_draft)
                continue
            roots = roots_for(s, R)
            if calib_posK != 1.0:
                K_step = max(p for p, _, in
                             [(seed[0], 0) for _, seed, _ in roots]) \
                    if roots else 0
                # pos=K(전부수락) 후보의 점수 보정 후 재정렬
                scored = [(pv * (calib_posK if seed[0] == K_step else 1.0),
                           rk, seed) for rk, seed, pv in roots]
                scored.sort(key=lambda x: -x[0])
                roots = [(rk, seed, sc) for sc, rk, seed in scored]
            hit_rank = None
            for i, (rk, seed, pv) in enumerate(roots):
                if seed == s["outcome"]:
                    hit_rank = i
                    break
            if hit_rank is not None:
                piv_sel = [max(pv, 1e-6) ** beta for _, _, pv in roots]
                b = max(1, round(NODE_BUDGET * piv_sel[hit_rank]
                                 / sum(piv_sel)))
                nv = min(b, nv_cap)
                L = tree_L(nv, alpha, lam)
                tot_tok += 1 + L
                tot_ms += period["hit_k2"] + max(0, nv - 4) * row_ms \
                    + delay_fn(slack_by, "hit_k2", delta_draft)
            else:
                tot_tok += committed if ph == 0 else 2.77
                tot_ms += period["miss"] + delay_fn(
                    slack_by, "miss", delta_draft)
        tps = tot_tok / (tot_ms / 1000.0)
        print(f"  {label:44s} tok/step={tot_tok/n:.3f} TPS={tps:6.2f}")
        return tps

    print("\n=== Step3-A: draft 증분 비용 Δ의 delay 전파 "
          "(최선 arm R=8/N_v=8, λ=1, 행당 1.9ms) ===")
    for d in (0.0, 0.5, 1.0, 2.0, 3.0):
        run_arm3(f"Δ_draft={d}ms", 8, delta_draft=d)

    print("\n=== Step3-B: 형제 상관 민감도 λ (R=8/N_v=8, Δ=1ms) ===")
    for lam in (1.0, 0.5, 0.25, 0.0):
        run_arm3(f"λ={lam} (1=독립 낙관, 0=체인과 동일 보수)", 8,
                 lam=lam, delta_draft=1.0)

    print("\n=== Step3-C: 배분 변형 (R=8/N_v=8, λ=1, Δ=0 — 순수 비교) ===")
    for beta in (0.0, 0.5, 1.0, 2.0):
        run_arm3(f"β={beta} (배분 ∝ P_iv^β)", 8, beta=beta)
    run_arm3("pos-K 보정 ×2 (E0 발견 반영, β=1)", 8, calib_posK=2.0)


if __name__ == "__main__":
    main()
