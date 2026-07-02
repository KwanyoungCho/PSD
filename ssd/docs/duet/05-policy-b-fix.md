# Policy B 재구현 + 통합 K+1 정형화

## 목적

DUET-SSD의 Phase 2 (proxy) 토큰 선택 로직 재설계. **Policy B를 단독
default**로 두고, all-accept 위치(pos=K)까지 통합한 단일 K+1 글로벌 ranking
으로 일원화한다. 현재 구현의 의미적 오류 및 dead code를 제거한다.

본 문서는 **구현되지 않은 변경에 대한 설계안**이다. 외부 검토자가 현재
코드를 읽고 본 문서의 변경안을 따라 구현 가능하도록 작성됐다.

**Scope 명시**: Policy A는 본 fix의 대상이 아니다. 코드에서 dead branch로
남겨두되 default가 아니며, 추후 정리 대상.

## 1. 현재 동작 (검토 기준)

### 1.1 데이터 흐름

```
[target rank 0]                               [draft]
  exit_logits → p_E (softmax)
  logits_q   → p_D (softmax)
  α_i = min(1, p_E(y_i)/p_D(y_i))
  residual = [p_E - p_D]_+ , scatter draft tok 제거
  topk_probs, topk_ids = residual.topk(top_k)  : [B,K,top_k]

  h[i] = (∏_{j<i} α_j) × (1-α_i)              : [K+1]
  h[K] = ∏α                                    (all-accept)

  Policy B:  P_iv = h[:K] × topk_probs[0]      ← pos==K 별도 처리
             chosen = argtopN(P_iv.flat, N=remaining)
             fan_out_list[i] = #chosen at pos i
             topk_ids reorder so chosen first
             fo_K = round(h[K]/h.sum() × total)

  wire send: fan_out_list, topk_ids, topk_probs   ─→ NCCL irecv
                                                    proxy_topk_ids[0,pos,:]
                                                    in_prev = self-dup
                                                    in_draft = vs draft_forked
                                                    valid = ~in_prev & ~in_draft
                                                    rank = valid.cumsum
                                                    take = first fo[pos] valid
                                                    pos==K: draft logits topk
```

### 1.2 Split-K1/K2 mode에서 두 차원 분리

**중요**: Policy B 설계에서 두 가지 길이는 **서로 다른 차원**이며 절대
혼동하면 안 된다.

| 차원 | 의미 | 값 |
|---|---|---|
| **ranking horizon** | Policy B가 평가할 position 수 | `valid_k + 1` (현재 verify row의 실제 길이) |
| **proxy forward depth** | Phase 2 proxy tree forward 단계 수 | **K2 (split-only 정의상 고정)** |

이 step의 verify는 valid_k+1 positions를 cover. Policy B는 그 길이를 따라
ranking. 반면 Phase 2 proxy tree는 split mode 정의상 항상 K2 deep forward.

| 케이스 | Phase 1 forward count<br>(`layout.K`) | Phase 1 position_count | Phase 2 forward count<br>(`layout.K`) | Phase 2 position_count<br>(현재) | ranking horizon<br>(target verify K+1) |
|---|---|---|---|---|---|
| **K1 hit** (long bucket) | **K1 (고정)** | K1+1 | **K2 (고정)** | K2+1 | K1+1 |
| **K2 hit** (short bucket) | **K1 (고정)** | K2+1 | **K2 (고정)** | K2+1 | K2+1 |
| **miss** | **K1 (고정)** | K1+1 | **K2 (고정)** | K2+1 | K1+1 (fallback) |

**중요**: Phase 1 forward count는 모든 케이스에서 **K1**. K2 hit이라고 K2번
forward하는 게 아님. forward 횟수와 position_count는 별개:
- forward count = `layout.K` = `for depth in range(K):` 루프 횟수
  (`draft_runner.py::_decode_tree` 1323).
- position_count = 트리에 존재하는 fork position 수 = `len(fan_out_list)`.

선례: `split_k1_short_layout` (`draft_runner.py` 324-329)이
`K=K1, position_count=K2+1`로 정확히 이 패턴.

Phase 2 position_count의 현재 값 (K2+1, 고정)은 **Section 3.6/3.7에서 Policy B
도입 시 변경 검토**.

→ **Policy B의 ranking 대상 position 수는 케이스별로 valid_k+1**:
- K1 hit / miss → K1+1
- K2 hit → K2+1

→ **proxy forward depth는 항상 K2** — layout의 K 인자는 K2.

K2 ≤ K1 invariant (`docs/duet/04-split-k1k2-design.md` 참조).

### 1.3 코드 위치

| 컴포넌트 | 파일 | 라인 |
|---|---|---|
| proxy 계산 + h + Policy A/B 분기 | `ssd/engine/verifier.py::_compute_and_send_proxy` | 227-380 |
| Policy B 예산 + 재정렬 | 동상 | 290-323 |
| wire pack (target → draft) | 동상 | 366-380 |
| wire unpack (draft) | `ssd/engine/draft_runner.py::_unpack_duet_proxy` | 1399-1420 |
| draft 토큰 selector | `ssd/engine/draft_runner.py::_select_proxy_sourced_tokens_policy_a` | 1462-1546 |
| `duet_proxy_top_k` auto-raise | `ssd/config.py::Config.__post_init__` | 177-187 |
| Policy default | `ssd/config.py` | `duet_policy: str = "a"` |

## 2. 식별된 문제

### Issue ① : 글로벌 ranking이 dedup 후 깨짐

**현상**: target은 `P_iv = h[:K] × residual` 글로벌 topN을 뽑아 chosen이 앞쪽
오게 재정렬해서 송신. draft는 per-position으로 첫 fo[pos]개의 valid 토큰을
픽함 (`_select_proxy_sourced_tokens_policy_a` 1502-1506). dedup으로 chosen
일부가 invalid이 되면 selector는 같은 pos의 **rest 영역(Policy B가 의도적으로
떨어뜨린 후순위)**에서 fall-through 토큰을 픽함.

**예시** (K=4, budget=10, dfo=1):

P_iv 글로벌 top-10 (target이 송신):
```
(0,A) 0.20  (0,B) 0.12  (1,D) 0.12  (2,G) 0.12  (1,E) 0.105
(0,C) 0.08  (1,F) 0.075 (2,A) 0.06  (3,I) 0.025 (3,D) 0.015
```
`fan_out_list = [3, 3, 2, 2, fo_K]`.

이제 draft 측 dedup. **draft가 이번 step의 새 트리에 후보로 둔 Phase 1
candidates** = `draft_forked = [B, E, A, D]` (각 position별 draft 모델
top-dfo 결과). 이 dedup은 **draft side에서만 일어남** — target은 Phase 1을
모름. (target은 별도로 verify-time 토큰 y_i를 residual.scatter로 미리
제외했음, `verifier.py` 257.)

dedup 후 픽 결과:
- pos 0 (fo=3): A ✓, B ✗ (Phase 1 중복), C ✓, **rank4** ← P_iv 글로벌 top-10 밖
- pos 1 (fo=3): D ✓, E ✗ (Phase 1 중복), F ✓, **rank4**
- pos 2 (fo=2): G ✓, A ✗ (Phase 1 중복), **rank3**
- pos 3 (fo=2): I ✓, D ✗ (Phase 1 중복), **rank3**

→ dedup 손실 4개를 글로벌 차순위 (예: pos=2의 H, pos=3의 J)가 아닌 local
rest로 채움. **Policy B의 글로벌 ranking 의도가 깨짐.**

### Issue ② : `duet_proxy_top_k` auto-raise가 split mode에선 over-provisioned

**현재** (`config.py` 178-181):
```python
K_plus_1 = self.speculate_k + 1                            # = K_long+1 = 17
max_possible_fo = self.duet_proxy_fan_out * K_plus_1       # = 1 × 17 = 17
required_top_k = max_possible_fo + self.duet_draft_fan_out + 2   # = 21
```

split mode에서 verify 깊이 K = K1 (worst case). 따라서 fan_out_list 길이는
K1+1. `K_long = K1+K2 = 16`은 **hybrid 시절 잔재**.

→ split mode 정확한 식: `pfo × (K1+1) + dfo + 2`. K1=8이면 9+2+2=13. 21 →
13 (38% 감소).

### Issue ③ : Policy B target side에 GPU sync (`.cpu().tolist()`)

**현재** (`verifier.py` 303-304):
```python
positions = (top_idx // top_k).cpu().tolist()    # ← GPU sync
ranks     = (top_idx % top_k).cpu().tolist()     # ← GPU sync
fan_out_list = [0] * (K + 1)
for pos in range(K):
    chosen = [r for p, r in zip(positions, ranks) if p == pos]
    ...
```

`top_idx`는 GPU 텐서. `.cpu()`는 GPU→Host 복사라 이전 GPU 작업이 완료될
때까지 host 멈춤. `_compute_and_send_proxy` hot path에 박혀 NCCL send를
지연시킴 → draft proxy_wait 증가.

**비용**: sync 10-50 μs + Python loop 30 μs ≈ 50-100 μs / step.

### Issue ④ : `topk_probs` wire 송신이 unused

**현재**: target → draft 송신:
- `fan_out_list` (K+1 ints) — draft 사용 ✓
- `topk_ids` ([B, K, top_k] ints) — draft 사용 ✓
- `topk_probs` ([B, K, top_k] floats) — **draft 미사용 ✗**

검증:
```
$ grep -n 'duet_proxy\["topk_probs"\]' ssd/engine/draft_runner.py
(0건)
```

→ `topk_probs` 1344 bytes/step 송수신, 사용 안 됨.

### Issue ⑤ : `in_prev` 마스크는 dead code

**현재** (`draft_runner.py` 1491-1496):
```python
proxy_exp  = proxy_topk_ids.unsqueeze(-1)               # [B, K, P, 1]
proxy_prev = proxy_topk_ids.unsqueeze(-2)               # [B, K, 1, P]
eq_prev    = (proxy_exp == proxy_prev)                  # [B, K, P, P]
in_prev    = (eq_prev & lower_triu).any(dim=-1)
```

`proxy_topk_ids[B, pos, :]`는 `residual.topk(top_k, dim=-1)` 결과로 dim=-1
따라 distinct V-indices 보장. 같은 pos 내 중복 불가능 → `in_prev`는 항상
전체 False.

**중요 — verify-time 토큰 (y_i) 제거와 무관**: target은 `verifier.py` 257
의 `residual.scatter_(2, gather_idx, 0.0)` 으로 **현재 verify 중인 draft
토큰 y_i**를 residual에서 사전 제거함. 따라서 draft가 받는 `proxy_topk_ids`
에는 y_i가 들어 있지 않음. y_i 중복 방지는 target side에서 이미 처리.
in_prev는 그것과 별개의 (그리고 죽은) 작업.

**비용**: `eq_prev` allocate `[1, 8, 21, 21]` ≈ 3.5 KB + 10 μs GPU work / step,
모두 무용.

## 3. 설계 결정

### 3.1 Policy B 단독 default — **지원 범위 한정**

```python
# ssd/config.py
duet_policy: str = "b"   # default; A path remains as dead branch (not exercised)
```

**중요 — Policy "b"가 적용되는 범위**:

| 구성 | Policy "b" 동작 |
|---|---|
| split-K1/K2 mode (`SSD_FORCE_SPLIT_K1K2=1`) + uniform Phase 1 | ✓ **fully supported** |
| split mode + **non-uniform** `duet_split_phase1_fan_out_list` | ✗ NotImplementedError (config 검증 시 raise) |
| 비분할 (legacy / hybrid) mode | ✗ 자동으로 `"a"`로 다운그레이드 + 경고 |

이유:
- Policy B의 unified selector는 `draft_forked: [B, P, dfo]` 3D를 가정 (`draft_forked[0, chosen_pos, :]` 인덱싱).
- non-uniform Phase 1의 `_select_draft_sourced_tokens_perpos`는 flat `[B, MQ_LEN]` 반환 → shape 불일치.
- 비분할 path의 `_build_tree_batch_duet()`는 `duet_proxy["fan_out_list"]` 직접 액세스 → Policy B wire schema에 그 키가 없어 KeyError.

→ 이 두 케이스는 별도 작업 (selector 확장 / legacy path 호환) 필요. 현재
fix는 split + uniform 범위만 커버하고, 외부 케이스는 fail-fast로 차단.

Policy A 코드는 즉시 제거하지 않음 (회귀 시 비교 + 비분할 fallback 가능).
향후 cleanup PR에서 정리.

### 3.2 All-accept (pos=K) 흡수 → 통합 K+1 정형화

기존: pos < K (residual) vs pos == K (별도 처리, draft logits topk) 분기.

신규: K+1 위치를 동등하게 P_iv 매트릭스에 포함.

| pos | reject prob | correction[v] |
|---|---|---|
| i < K | h[i] = (∏_{j<i} α_j) × (1−α_i) | residual = [p_E − p_D]_+ |
| K | h[K] = ∏α | p_E[K, :] (target only) |

`correction_topk_probs[B, K+1, top_k]`, `correction_topk_ids[B, K+1, top_k]`
를 한 번에 만들어 `P_iv = h.view(K+1, 1) × correction_topk_probs` `[K+1, top_k]`.

→ pos==K 분기 제거. `for pos in range(K)` 제거. `fo_K` 별도 계산 제거.

### 3.3 Dedup-aware 글로벌 fall-through

1. target이 **글로벌 top-(budget + buffer)** 만 뽑아 (pos, tok) 쌍으로
   송신 (점수 정렬 보존).
2. draft가 받아 in_draft (Phase 1 candidates 중복) dedup.
3. 살아남은 후보 중 **앞에서부터** budget개 픽.
4. 픽한 토큰들의 pos 카운트 → fan_out_list 동적 재구성.

→ dedup 손실 시 글로벌 차순위가 자동 fall-through. 글로벌 ranking 보존.

**Underfill 불가 (sum == total_budget 항상 성립)**:
- buffer = `(K_rank_max+1) × dfo + 2` — 각 position에서 in_draft로
  떨어질 수 있는 최대치 (dfo개) × position 수 + 안전 여유.
- wire_N = total_budget + buffer (Section 3.5).
- 따라서 dedup 후 valid 후보 수 ≥ total_budget **항상 보장**.
- → `sum(fan_out_list) == total_budget` invariant. underfill 안전장치 불필요.
- → **Phase 2 layout MQ_LEN (CG capture) = total_budget** (buffer 제외).
  buffer는 wire 송신용에 한정.

**Invariant: chosen_pos ∈ [0, K_rank]**:
- target이 P_iv shape `[K_rank+1, top_k]`에서 `flatten().topk(wire_N)` 호출.
- 후보 총수 `(K_rank+1) × top_k ≥ wire_N` 보장 (top_k auto-raise; Section 3.5
  `top_k_total` 항).
- `pos = idx // top_k` ∈ `[0, K_rank]`. **construction에 의해 자동 만족**.
- → 별도 in_range 체크 불필요 (debug assert로만 둠).

### 3.4 Wire 포맷 변경

기존: `fan_out_list` + `topk_ids` + `topk_probs`. ≈ 1.4 KB / step.
신규: `chosen_pos[N]` + `chosen_tok[N]` (score-sorted desc), N = budget + buffer.

→ topk_probs 제거 (Issue ④), fan_out_list 송신 불필요 (draft 동적 재구성).

### 3.5 K1 vs K2 hit 사이즈 결정 — 두 옵션 분석

Section 1.2에서 본 대로, Policy B의 ranking 대상 위치 수는 이 step verify의
깊이 K에 의해 결정됨:
- K1 hit / miss: K = K1 → K+1 = K1+1 positions
- K2 hit: K = K2 → K+1 = K2+1 positions

이로 인해 `top_k`, `buffer`, `MQ_LEN`, wire 사이즈 결정에 두 가지 선택지가
있음.

**주의**: Section 1.2에 따라 **proxy forward depth는 K2로 고정**.
사이징 결정은 **ranking horizon (= verify K + 1)**에만 영향. forward depth K2는
어느 옵션에서도 변하지 않는다.

#### 옵션 A: 항상 worst-case ranking horizon (= K1+1)

세 가지 양 (wire / Phase 2 tree / top_k) 모두 worst-case K_rank_max=K1 기준.

```python
K_rank_max    = K1                                       # invariant K2 ≤ K1
K_plus_1_max  = K_rank_max + 1                           # = K1+1

# (1) Phase 2 tree 크기 — CG capture MQ_LEN의 기준
total_budget  = pfo * K_plus_1_max                       # K1+1 positions 분량 fan_out 합
                                                          # = sum(fan_out_list) at runtime
                                                          # K1=K2일 때 변동 없음.
                                                          # K1 > K2일 때도 모든 step에 동일하게
                                                          # 고정 (Phase 2 work uniform).

# (2) wire 크기 — target → draft NCCL 송신 후보 수
buffer        = K_plus_1_max * dfo + 2                   # 최악 dedup 손실 + margin
wire_N        = total_budget + buffer                    # ★ MQ_LEN과 별개 ★

# (3) top_k auto-raise — P_iv 후보 수 ≥ wire_N 보장
# 두 제약의 max:
#   per-pos:  top_k ≥ max_fo + dfo + 2 = pfo*K_plus_1_max + dfo + 2  (한 pos에 budget 몰빵 시)
#   total:    top_k ≥ ceil(wire_N / (K_min+1))                       (short-hit candidate count)
# K_min = K2 (= K_short).
top_k_per_pos = pfo * K_plus_1_max + dfo + 2
top_k_total   = -(-wire_N // (K2 + 1))                   # ceil division
top_k         = max(top_k_per_pos, top_k_total)

# proxy forward depth는 별개로 항상 K2 (옵션 무관)
proxy_forward_depth = K2
```

**`total_budget`이 K1+1 고정인 의미** (Medium #3 명시):
- K2 hit step도 Phase 2 tree에 `pfo × (K1+1)` 개 토큰을 넣음 (`K2+1`이 아님).
- → K2 hit step의 Phase 2 forward 작업량은 K1 hit step과 **동일** (CG replay 균일).
- 의도: Phase 2 work를 step간 균일하게 유지하여 단일 CG capture로 처리.
- short-hit에서 Phase 2 work 줄이려면 옵션 B 필요.

**`wire_N` vs `MQ_LEN` 분리** (High #1 명시):
- `MQ_LEN` (CG capture, layout.MQ_LEN) = `total_budget`. selector는
  `sum(fan_out_list) == total_budget` 반환하므로 정확히 일치.
- `wire_N` = `total_budget + buffer`. NCCL pack 사이즈. 이는 layout/CG와
  무관, target → draft 데이터 전송량.
- buffer는 dedup 손실 대비 wire 추가 후보용. tree decode에 들어가는 query
  수가 아님. selector가 dedup 후 `total_budget`개를 추려서 tree 빌드.

**top_k 제약 (High #2 명시)**:
- 위 식의 `top_k_total` 항이 추가됐음. 기존 식(per-pos만)은 K1 ≫ K2일 때
  `(K_min+1) × top_k < wire_N` 가능 → topk(wire_N) 실패.
- 예: K1=8, K2=1, pfo=1, dfo=2 → per-pos=13, total=ceil(29/2)=15 → top_k=15.

K2 hit case에선 정작 ranking이 K2+1 positions에 한정되는데, wire/MQ_LEN
모두 K1+1 사이즈. (K1−K2) × (something) 슬롯이 비활성.

**장점**:
- 단일 wire schema, 단일 CG capture, prealloc 1세트.
- 분기 코드 없음.
- K1=K2일 땐 손해 0 (현재 실험 환경).

**단점**:
- K1 ≫ K2일 때 K2 hit step에서 over-provision (wire + Phase 2 work 둘 다).
- 예: K1=12, K2=4 → K2 hit step에서 Phase 2 forward 175% 낭비
  (= (13-5)/5 × 100). short-hit work 줄이려면 옵션 B 필요.

#### 옵션 B: 버킷별 ranking horizon 사이즈 분리

```python
# 두 종류 buffer/wire (forward depth는 양쪽 모두 K2로 동일)
N_long  = pfo * (K1+1) + (K1+1) * dfo + margin
N_short = pfo * (K2+1) + (K2+1) * dfo + margin

top_k_long  = pfo * (K1+1) + dfo + 2
top_k_short = pfo * (K2+1) + dfo + 2

# proxy forward depth는 양쪽 모두 K2 (불변)
proxy_forward_depth = K2
```

target은 verify 깊이 (= ranking horizon = K1 or K2)를 알고 그에 맞는 N으로
송신. draft는 들어오는 wire 사이즈로 어느 bucket인지 판별 → 해당 layout 사용.
**Phase 2 forward 자체는 양쪽 모두 K2 deep**.

**장점**:
- K2 hit 케이스에서 wire/메모리 절감.
- K1 ≫ K2 케이스에서 효과 큼.

**단점**:
- wire schema 분기 (long/short 두 종류).
- CG 추가 capture 필요 (verify_k2 케이스용 별도 layout 또는 mask 분기).
- prealloc buffer 두 세트.
- 구현 복잡도 + 디버그 면적 증가.

#### 권장: **옵션 A로 시작**, 옵션 B는 future work

근거:
1. 현재 실험 setup은 **K1 = K2 = 8** → 옵션 A 손해 0.
2. 일반적으로 split mode에서 K2 < K1로 운용해도 K1 ≤ 16 정도 (`speculate_k`
   상한과 버퍼 사이즈 비교에서). K1=16, K2=8 worst case에서 wire 50%
   낭비 — 절대량은 100여 ints 수준 (작음).
3. 옵션 B의 구현 비용 (CG 추가 capture, schema 분기)이 절감 효과보다 큼.
4. 본 fix의 critical-path 변경(Issue ①③④⑤) 없이는 옵션 B 구현해도
   효과 못 받음.

→ **단일 K_max 사이즈 (= K1+1) 사용**. 미래에 K1 ≫ K2 사용 패턴 생기면 별
도 PR로 옵션 B 도입.

### 3.6 Phase 2 layout — 두 차원 분리

신규 layout은 두 dimension을 명확히 구분:

| 인자 | 의미 | 값 |
|---|---|---|
| `K` | proxy tree forward depth (= 몇 단계 forward) | **K2 (고정)** |
| `position_count` | tree 내 position 갯수 (Policy B ranking horizon에 대응) | **valid_k + 1 (step별 동적)** |
| `fan_out_list` | position별 fan-out (dedup 후 동적 재구성) | sum = total_budget |

```python
# 매 step (Policy B path):
step_proxy_layout = create_tree_layout(
    name="split_k2_dyn",
    K=K2,                                        # forward depth = K2 (불변)
    position_count=_step_valid_k + 1,            # ranking horizon (K1 hit: K1+1, K2 hit: K2+1)
    fan_out_list=proxy_fan_out_list,             # draft가 dedup 후 결정
    fan_out_list_miss=proxy_fan_out_list,
    device=self.device,
)
```

**잘못된 예시** (혼동 사례):
```python
# WRONG: K=_step_valid_k 는 forward depth를 ranking horizon에 묶어버림.
#        split-only 정의상 forward depth는 K2여야 함.
step_proxy_layout = create_tree_layout(
    K=_step_valid_k,                # ✗ 잘못됨
    position_count=_step_valid_k+1,  # 이건 OK
    ...
)
```

`split_k1_short_layout` (`draft_runner.py` 324-329) 가 동일한 패턴 선례:
```python
# K=K1 (forward depth) ≠ position_count=K2+1 (positions used)
self.split_k1_short_layout = create_tree_layout(
    name="split_k1_short",
    K=K1, position_count=K2 + 1,                 # 두 차원 분리됨
    fan_out_list=_p1_fol_short, ...
)
```

신규 split_k2_dyn은 같은 패턴, 다만 K가 K2이고 position_count가 동적.

### 3.7 Phase 2 dynamic layout — CG / wrapper 계약 (옵션 A 기준)

리뷰어 지적 (High): 현재 코드는 `split_k2` 단일 capture만 존재 + 고정
`MQ_LEN = pfo × (K2+1)` 전제 (`model_runner.py` 314, `draft_runner.py` 2204,
`cudagraph_helpers.py` 174 — bucket name `split_k2` 하드코딩).

신규 path는 step별 position_count 변동 (K1 hit: K1+1, K2 hit: K2+1) — 기존
`split_k2`의 고정 가정과 충돌. **두 옵션 중 하나로 정해야 함**:

#### 옵션 A-1: 단일 `split_k2` capture를 worst-case로 재설계 (권장)

기존 `split_k2`의 `MQ_LEN`을 worst-case로 키우고, position_count / fan_out_list
는 mask 단에서 step별 update.

```python
# 변경: model_runner.py 314 부근
# OLD:  _p2_mq = _pfo * (K2_cfg_split + 1)
# NEW (옵션 A — worst-case sizing):
# layout MQ_LEN = total_budget (= sum(fan_out_list) at runtime)
# wire_N (= total_budget + buffer) 는 NCCL pack에서 별도 사이즈, layout과 분리.
K_rank_max = max(K1_cfg_split, K2_cfg_split)        # = K1 (K2 ≤ K1)
_p2_mq = _pfo * (K_rank_max + 1)                     # MQ_LEN = total_budget
_layout_specs.append(("split_k2", None, None, _p2_mq))   # 같은 이름 유지
```

- **CG capture**: `split_k2` 한 family로 유지. capture 시 `MQ_LEN = MQ_max`
  으로 buffer prealloc.
- **Wrapper**: `prefill_wrappers_by_layout["split_k2"]` 한 세트.
- **per-step**: `position_count` (= `K_rank+1`) 와 `fan_out_list` 만 mask
  buffer에 in-place write. CG replay 그대로.
- **bucket name `split_k2`** 그대로 — `cudagraph_helpers.py` 174 변경 불필요.
- **장점**: 기존 자산 최대 재사용. 신규 capture 추가 없음.
- **단점**: K2 hit step에서 mask buffer 일부 비활성 (옵션 A의 본래 trade-off).

#### 옵션 A-2: `split_k2_dyn` 신규 family 추가

별도 capture / wrapper / bucket name 신설.

- **CG capture**: `split_k2_dyn` 신규 family.
- **Wrapper**: `prefill_wrappers_by_layout["split_k2_dyn"]` 추가.
- **`cudagraph_helpers.py`**: bucket dispatch에 `split_k2_dyn` 인식 추가.
- **장점**: 기존 `split_k2`(static) 와 분리되어 회귀 안전.
- **단점**: 코드 복잡도 ↑. 옵션 A의 단순성 이점 일부 상실.

#### 권장: **옵션 A-1**

근거: 옵션 A는 처음부터 "단일 사이즈로 통합"이 목적. A-2처럼 family를 따로
두면 옵션 B(버킷 분리)와 변별력이 흐려짐. 기존 `split_k2`의 의미를
"position_count 동적, MQ_LEN worst-case 고정"으로 일반화하는 게 자연스러움.

**구현 작업 list (옵션 A-1)**:
1. `model_runner.py` 314 부근: `_p2_mq` 계산을 worst-case 식으로.
2. `draft_runner.py` 330-335 (`split_k2_layout` create): K=K2, position_count도
   K_max+1로 일단 잡고, 매 step rebuild로 position_count update.
3. `draft_runner.py` 2204 부근 (proxy decode 호출): step_proxy_layout을 매
   step rebuild — fan_out_list, position_count 둘 다 동적.
4. `cudagraph_helpers.py` 174: 변경 불필요 (bucket name 유지).

CG mask는 `duet_verify_*` capture 시 mutable buffer로 가정되므로 step별
mask rebuild는 외부에서 buffer write로 처리 (CG 재캡처 없음). 검증 필요
(Section 7).

## 4. 구현 단계

### Step 1 (config) — Policy default + top_k 수정

**파일**: `ssd/config.py`

```diff
-    duet_policy: str = "a"
+    duet_policy: str = "b"
```

```diff
 # __post_init__
-K_plus_1 = self.speculate_k + 1
-max_possible_fo = self.duet_proxy_fan_out * K_plus_1
-required_top_k = max_possible_fo + self.duet_draft_fan_out + 2
+# Issue ②: split mode worst-case = K_max = K1.
+# 두 제약 중 max:
+#   per-pos:  pfo*(K_max+1) + dfo + 2     (한 position에 budget 몰빵 시)
+#   total:    ceil(wire_N / (K_min+1))    (short-hit candidate count >= wire_N)
+# wire_N = total_budget + buffer = pfo*(K_max+1) + (K_max+1)*dfo + 2
+if self.duet_phase1_k is not None and \
+   os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1":
+    K_max = max(self.duet_phase1_k, self.duet_phase2_k)
+    K_min = min(self.duet_phase1_k, self.duet_phase2_k)
+else:
+    K_max = self.speculate_k
+    K_min = self.speculate_k
+pfo = self.duet_proxy_fan_out
+dfo = self.duet_draft_fan_out
+total_budget = pfo * (K_max + 1)
+buffer       = (K_max + 1) * dfo + 2
+wire_N       = total_budget + buffer
+per_pos_min  = total_budget + dfo + 2                       # = pfo*(K_max+1)+dfo+2
+total_min    = -(-wire_N // (K_min + 1))                    # ceil(wire_N / (K_min+1))
+required_top_k = max(per_pos_min, total_min)
```

신규 config 필드 (옵션 A buffer):
```python
@property
def duet_proxy_wire_N(self) -> int:
    """Total (chosen_pos, chosen_tok) entries on wire = budget + buffer."""
    K_plus_1 = max(self.duet_phase1_k, self.duet_phase2_k) + 1 \
               if self.duet_phase1_k is not None else self.speculate_k + 1
    total_budget = self.duet_proxy_fan_out * K_plus_1
    buffer = K_plus_1 * self.duet_draft_fan_out + 2
    return total_budget + buffer
```

### Step 2 (verifier) — Policy B 재작성 (K+1 통합 + GPU-only)

**파일**: `ssd/engine/verifier.py::_compute_and_send_proxy`

```diff
-# residual & topk
-residual = (p_E - p_D).clamp(min=0)
-residual.scatter_(2, gather_idx, 0.0)
-topk_probs, topk_ids = residual.topk(top_k, dim=-1)
-topk_probs = topk_probs / topk_probs.sum(...)
-
-# h
-cumprod = torch.cumprod(accept_probs[0], dim=0)
-h = torch.zeros(K + 1, ...)
-h[0]   = 1 - accept_probs[0, 0]
-h[1:K] = cumprod[:-1] * (1 - accept_probs[0, 1:])
-h[K]   = cumprod[-1]
-
-# Policy A/B 분기 (290-331)
-if config.duet_policy == "b":
-    fo_K = round(h[K] / h.sum() * proxy_fan_out_total)
-    ...
-    for pos in range(K):
-        chosen = [...]
-        ...
-else:
-    raw = (h / h.sum() * proxy_fan_out_total).floor().int()
-    ...
-
-# pad to K_long+1, send
-fan_out_tensor = ...
-if K < K_wire:
-    pad ...

+# === 통합 K+1 path (Policy B 전용) ===
+# 주의: 현재 코드 p_E = softmax(exit_logits[:, :K, :])는 pos 0..K-1만 포함.
+# 통합 path에선 pos==K도 필요 → exit_logits에서 한 칸 더 읽어옴.
+
+# pos < K: residual = [p_E - p_D]_+ (현재와 동일)
+p_E      = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)   # [B, K, V]
+residual_pos = (p_E - p_D).clamp(min=0)
+residual_pos.scatter_(2, gather_idx, 0.0)
+res_topk_probs, res_topk_ids = residual_pos.topk(top_k, dim=-1)
+res_topk_probs = res_topk_probs / res_topk_probs.sum(-1, keepdim=True).clamp(min=1e-10)
+
+# pos == K: target's full distribution at K. exit_logits[:, K, :]에서 따로 계산.
+pE_K = torch.softmax(exit_logits[:, K, :].float(), dim=-1)        # [B, V]
+pE_K_topk_probs, pE_K_topk_ids = pE_K.topk(top_k, dim=-1)         # [B, top_k]
+pE_K_topk_probs = pE_K_topk_probs / pE_K_topk_probs.sum(-1, keepdim=True)
+
+# concat to [B, K+1, top_k]
+correction_probs = torch.cat([res_topk_probs, pE_K_topk_probs.unsqueeze(1)], dim=1)
+correction_ids   = torch.cat([res_topk_ids,   pE_K_topk_ids.unsqueeze(1)],   dim=1)
+
+# h: [K+1] (unchanged 계산)
+cumprod = torch.cumprod(accept_probs[0], dim=0)
+h = torch.zeros(K + 1, ...)
+h[0]   = 1 - accept_probs[0, 0]
+h[1:K] = cumprod[:-1] * (1 - accept_probs[0, 1:])
+h[K]   = cumprod[-1]
+
+# 글로벌 P_iv top-N (옵션 A: K_max 기준 사이즈)
+N = config.duet_proxy_wire_N        # = total_budget + buffer
+P_iv = h.view(K + 1, 1) * correction_probs[0]              # [K+1, top_k]
+_, top_idx = P_iv.flatten().topk(N)                        # [N], score desc
+chosen_pos_global = top_idx // top_k                        # [N]
+chosen_tok_global = correction_ids[0].view(-1).gather(0, top_idx)  # [N]
+
+# K가 K_max보다 작은 경우 (K2 hit) — pos는 [0, K]에 들어있음. 옵션 A에서
+# wire는 K_max+1 사이즈로 padding 필요 없음 (chosen_pos 값이 작음).
+# 단 NCCL pack에서 N은 고정 (config.duet_proxy_wire_N).
+
+# wire send
+wire_send(chosen_pos_global, chosen_tok_global)
```

핵심: GPU 텐서 그대로 NCCL send → `.cpu().tolist()` 없음. Issue ③ 해소.

### Step 3 (draft) — wire unpack + 통합 selector

**파일**: `ssd/engine/draft_runner.py`

`_unpack_duet_proxy` 신규:
```python
def _unpack_duet_proxy(self, buf, B, K):
    N = self.config.duet_proxy_wire_N
    chosen_pos = buf[:N]                     # [N] int64
    chosen_tok = buf[N:2*N]                  # [N] int64
    return {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}
```

신규 selector `_select_proxy_sourced_tokens_unified`:
```python
def _select_proxy_sourced_tokens_unified(self, duet_proxy, draft_forked,
                                          K_rank, total_budget):
    """K+1 통합 unified Policy B selector.

    Args:
        duet_proxy: {"chosen_pos": [N], "chosen_tok": [N]}  score-sorted desc.
                    chosen_pos values ∈ [0, K_rank] (target-side construction
                    invariant; Section 3.3 참조).
        draft_forked: [B, K_rank+1, dfo]   Phase 1 candidates per ranking
                      position. **K_rank+1 폭이면 충분** (chosen_pos ≤ K_rank
                      invariant). 호출자는 split-only short-hit Phase 1이
                      자연스레 K_rank+1 폭 후보만 만들므로(`draft_runner.py`
                      2148, 2166), 그대로 전달.
                      (full-width K_long+1 draft_forked_full은 별도로 유지
                      되지만 본 selector는 K_rank+1 slice만 받음.)
        K_rank:  int  이 step의 ranking horizon (= valid_k).
                       K1 hit / miss: K1, K2 hit: K2.
                       Phase 2 forward depth (K2)와 별개임 — 본 selector는
                       ranking에만 관여하고 forward depth는 layout이 결정.
        total_budget: int  Phase 2 tree에 채울 토큰 수.

    Returns:
        result_tokens: [B, total_budget] int64  (pos 순 stable sort, score 보존)
        fan_out_list:  list[int] length K_rank+1, sum == total_budget.
                       layout의 position_count = K_rank+1과 일치.
                       buffer 사이징(Section 3.3)으로 underfill 불가.
    """
    chosen_pos = duet_proxy["chosen_pos"]    # [N]
    chosen_tok = duet_proxy["chosen_tok"]    # [N]
    N = chosen_pos.shape[0]
    B = draft_forked.shape[0]
    assert B == 1                            # DUET invariant
    # Invariant: chosen_pos ∈ [0, K_rank] by target-side construction.
    # debug-only sanity check (제거해도 안전):
    if __debug__:
        assert (chosen_pos <= K_rank).all(), \
            f"chosen_pos out of range [0, {K_rank}]: max={chosen_pos.max()}"

    # in_draft dedup against draft_forked[0, chosen_pos, :]
    df_per_cand = draft_forked[0, chosen_pos, :]              # [N, dfo]
    in_draft = (df_per_cand == chosen_tok.unsqueeze(-1)).any(-1)
    valid = ~in_draft                                          # [N]

    # 첫 total_budget개 valid 픽 (점수 정렬 보존)
    rank = valid.to(torch.int64).cumsum(0)
    take = valid & (rank <= total_budget)                     # [N]
    # buffer 사이징상 take.sum() == total_budget 항상 성립 (underfill 불가).

    taken_pos = chosen_pos[take]                              # [budget]
    taken_tok = chosen_tok[take]

    # fan_out_list 동적 재구성: pos별 카운트 (길이 = K_rank+1)
    K_plus_1 = K_rank + 1
    fan_out_tensor = torch.zeros(K_plus_1, dtype=torch.int64,
                                  device=chosen_pos.device)
    fan_out_tensor.scatter_add_(0, taken_pos, torch.ones_like(taken_pos))
    fan_out_list = fan_out_tensor.tolist()                    # 1회 sync / step

    # result tensor [B, MQ_LEN] in pos order (stable sort by pos, score 내부 보존)
    MQ_LEN = total_budget
    result = torch.zeros(B, MQ_LEN, dtype=torch.int64,
                          device=chosen_pos.device)
    order = taken_pos.argsort(stable=True)
    result[0, :taken_tok.shape[0]] = taken_tok[order]

    return result, fan_out_list
```

**Issue ⑤ 자연 제거**: in_prev 마스크 없음. 글로벌 ranking이 distinct
보장하므로 불필요.

### Step 4 (호출자) — Policy B 단독 path

**파일**: `ssd/engine/draft_runner.py` 1748 (split path), 2247 (hybrid path 동일).

```diff
-proxy_forked = self._select_proxy_sourced_tokens_policy_a(
-    glue_logits, gd_for_fork, duet_proxy, draft_forked,
-    proxy_fan_out_list, valid_k=_step_valid_k)
+# Policy B 단독 default. Policy A는 dead branch (config 검증으로 차단됨).
+proxy_forked, dyn_fan_out_list = self._select_proxy_sourced_tokens_unified(
+    duet_proxy, draft_forked,
+    K_rank=_step_valid_k, total_budget=total_budget)
+proxy_fan_out_list = dyn_fan_out_list
+# layout 매 step rebuild — 두 차원 분리 (Section 3.6 / 3.7 참조):
+#   K = K2                       (forward depth, 불변)
+#   position_count = K_rank+1    (ranking horizon, 동적)
+# 옵션 A-1 채택: bucket name "split_k2" 유지, MQ_LEN은 worst-case 고정.
+K2_cfg = self.config.duet_phase2_k
+step_proxy_layout = create_tree_layout(
+    name="split_k2",                          # 기존 bucket name 유지
+    K=K2_cfg,                                  # forward depth
+    position_count=_step_valid_k + 1,          # 동적
+    fan_out_list=proxy_fan_out_list,
+    fan_out_list_miss=proxy_fan_out_list,
+    device=self.device,
+)
```

`_select_proxy_sourced_tokens_policy_a` 함수와 그 진입로(`config.duet_policy
== "a"`)는 코드에 남겨두되 default 경로 아니므로 죽은 코드.

### Step 5 (wire helpers) — pack/unpack schema

**파일**: `ssd/engine/verifier.py` (`wire_send`) + `ssd/engine/draft_runner.py`
(`_unpack_duet_proxy`).

기존 `send_int64` 사용 패턴 유지. schema 변경:
```
[chosen_pos: N int64] [chosen_tok: N int64]    # N = config.duet_proxy_wire_N
```

prealloc buf 사이즈 = 2N int64. step마다 재사용. NCCL irecv은 buf의 첫
2N 슬롯만 사용.

### Step 6 (CG / wrapper 호환성 — 옵션 A-1 적용)

Section 3.7 (옵션 A-1)에 따라 `split_k2` family를 worst-case로 일반화.

**`model_runner.py` 수정 (line 314 부근)**:
```diff
-_p2_mq = _pfo * (K2_cfg_split + 1)
+# Phase 2 worst-case sizing for unified Policy B (옵션 A-1).
+# position_count는 step별 동적 (K1 hit: K1+1, K2 hit: K2+1) 이지만,
+# CG capture / wrapper의 MQ_LEN = sum(fan_out_list) at runtime = total_budget.
+# wire_N (= total_budget + buffer) 와 분리됨 — wire 사이즈는 NCCL pack에서
+# 별도 정해짐. layout MQ_LEN에는 buffer 포함 안 함.
+K_rank_max = max(K1_cfg_split, K2_cfg_split)             # = K1 (K2 ≤ K1)
+_p2_total_budget = _pfo * (K_rank_max + 1)               # Phase 2 tree size
+_p2_mq = _p2_total_budget                                # MQ_LEN = total_budget
+_layout_specs.append(("split_k2", None, None, _p2_mq))
```

**`draft_runner.py` 수정 (line 330 부근, `split_k2_layout` create)**:
```diff
 self.split_k2_layout = create_tree_layout(
     name="split_k2",
-    fan_out_list=[proxy_fo] * (K2 + 1),
-    fan_out_list_miss=[proxy_fo] * (K2 + 1),
-    K=K2, device=d, position_count=K2 + 1,
+    # Capture-time placeholder. Per-step rebuild updates
+    # fan_out_list / position_count (옵션 A-1).
+    fan_out_list=[proxy_fo] * (K_rank_max + 1),
+    fan_out_list_miss=[proxy_fo] * (K_rank_max + 1),
+    K=K2,
+    position_count=K_rank_max + 1,            # placeholder; updated per-step
+    device=d,
 )
```

**`cudagraph_helpers.py` 변경 불필요**: bucket name `split_k2` 유지.

**검증 (smoke)**:
```bash
SSD_TRACE_BUCKET=1 SSD_FORCE_SPLIT_K1K2=1 python -O bench/bench.py \
    --duet_policy b ... --duet_phase1_k 8 --duet_phase2_k 8 \
    --duet_exit_layer 57
# 매 step bucket dispatch 로그 확인 → split_k2 capture가 K1=K2=8 에서
# 정상 동작 (position_count=9 placeholder vs runtime 9 일치).

# 다른 K1 != K2 케이스도 테스트:
... --duet_phase1_k 12 --duet_phase2_k 4
# placeholder K_rank_max+1=13, runtime: K1 hit step=13, K2 hit step=5.
```

## 5. 검증 계획

### 5.1 단위 테스트 (`ssd/tests/test_policy_b_unified.py` 신규)

1. `test_correction_topk_probs_shape` — `[B, K+1, top_k]` 검증.
2. `test_global_topN_ordering` — P_iv 큰 토큰이 chosen_pos/tok 앞에.
3. `test_dedup_global_fallthrough` — draft_forked overlap 강제, 글로벌
   차순위가 들어오는지.
4. `test_fan_out_list_sum_exact` — `sum(fan_out_list) == total_budget`.
5. `test_chosen_pos_in_range` — invariant `chosen_pos ∈ [0, K_rank]` 검증.
6. `test_buffer_sufficient_no_underfill` — 모든 (in_draft 최악 시나리오)에서
   underfill 발생 안 함 검증.
7. `test_pE_K_indexing` — `exit_logits[:, K, :]` 별도 인덱싱이 정상 동작.

### 5.2 End-to-end 측정 (`exit_fine/exit_57` baseline)

host load < 10 시점 측정:

| metric | 기존 (A) | 신규 (B 통합) | 기대 |
|---|---|---|---|
| TPS | 32.5 | ? | 동등 또는 ↑ |
| accept rate | 0.30 | ? | ↑ (더 likely 토큰 선택) |
| Phase 2 hit rate | 0.07 | ? | **↑↑** (글로벌 ranking 보존) |
| accept_len_on_hit | 4.2 | ? | ↑ |
| graph_pre | 33 ms | ? | 동등 (target work 비슷) |
| proxy_compute_send | 1.5 ms | ? | ↓ (Issue ③ 해소) |
| draft_step | 70 ms | ? | 동등 또는 ↓ |

### 5.3 측정 지표 추가 (별도 commit)

`SSD_DUET_PROXY_STATS=1` 환경변수로 per-step 수집:
- `h` (예상 거부 분포)
- `chosen_pos` / `chosen_tok` 히스토그램
- in_draft drop 카운트 / step
- valid 카운트 / step
- fan_out_list 분포

상세는 별도 docs (예: `06-policy-b-stats.md`)에 정리.

## 6. 영향 받는 파일 요약

| 파일 | 변경 LOC (추정) |
|---|---|
| `ssd/config.py` | +12, -3 |
| `ssd/engine/verifier.py` | +50, -50 (Policy B path 재작성) |
| `ssd/engine/draft_runner.py` | +90, -10 (신규 unified selector + 호출 분기) |
| `ssd/tests/test_policy_b_unified.py` | +250 (신규) |
| `ssd/docs/duet/05-policy-b-fix.md` | 본 문서 |

총 신규/변경 ≈ 400 LOC.

## 7. 미결 / 추가 검토

1. **두 차원 분리 일관성**: ranking horizon (= valid_k+1) 와 proxy forward
   depth (= K2) 가 코드 전반에서 분리된 채 사용되는지. 호출 site 모두
   `K_rank` (ranking) vs `K2` (forward) 명명 분리 확인.
2. **Phase 2 cache 호환성**: 동적 fan_out_list가 다음 step의 cache lookup에
   영향 없는지. cache key는 `(seq_id, fan_idx, recovery_token)`, fan_idx는
   layout-기반. 동적 layout이라도 fan_idx 매핑 일관성 있으면 OK.
3. **CG mask in-place buffer**: `duet_verify_k1` / `duet_verify_k2` capture
   시 mask가 정적 가정인지 동적 가정인지 코드 검증 필요.
4. **selector 입력 폭 contract**: split-only short-hit Phase 1이 자연스레
   `K_rank+1` 폭 candidates 생성 (`draft_runner.py` 2148, 2166). 본 fix는
   selector 입력을 `[B, K_rank+1, dfo]`로 통일. full-width
   `draft_forked_full`은 caller가 슬라이싱 후 전달.
4. **B=1 invariant**: 현 코드 `B=1` 가정. 신규 selector도 같은 가정 유지.
5. **buffer 크기 검증**: `(K_rank_max+1) × dfo + 2` 가 underfill 방지에
   충분한지 단위 테스트 + 실측에서 검증. Underfill은 design상 불가
   (Section 3.3 invariant) — 발생 시 buffer 식 오류이므로 반드시 fail-fast.
7. **wire_N vs MQ_LEN 분리 일관성**: Phase 2 layout MQ_LEN = total_budget
   (buffer 제외). wire_N = total_budget + buffer는 NCCL pack에만 적용.
   model_runner의 `_p2_mq` (= layout MQ_LEN) 와 NCCL buf prealloc 사이즈
   (= wire_N) 는 서로 다른 값. 코드 분리 확인 필요.
8. **top_k auto-raise 식**: `max(per_pos_min, total_min)` 이 모든
   `(K1, K2, pfo, dfo)` 조합에서 `(K_min+1) × top_k ≥ wire_N` 만족하는지.
   특히 K1 ≫ K2 케이스 (K2=1, 2 등) 가 binding constraint.
9. **현재 fix 적용 범위 한정** (Section 3.1 참조):
   - 비분할 모드는 default "a" (자동 downgrade)
   - 분할 + non-uniform Phase 1은 NotImplementedError로 차단
   - 두 케이스 모두 selector / legacy path 확장이 필요한 별도 작업 — 본 fix
     scope 밖.
10. **prep sync 추가 최적화 보류** (commit `feb3691` 검증 결과):
    - `H_plan_event_sync` (`run_fi_tree_decode_cudagraph` 내 stream sync)가
      step-1+ prep 시간의 ~92%를 차지하지만, 이는 GPU-stream-drain wait —
      이전 `graph.replay()` 완료까지 CPU가 대기하는 직렬 구간.
    - skip_sync 가설 검증 (with `kv_lens_cpu` aligned to FlashInfer
      contract): 여전히 race → wrapper buffer lifetime 보호 목적 → 단순
      제거 불가.
    - K_loop = layout.K cleanup으로 step-0 prep CPU work 33-66% 감소했으나
      TPS는 variance 영역 (sync가 binding constraint이라 절약 시간 흡수).
    - 추가 최적화는 wrapper buffer ping-pong / FlashInfer plan() 우회 등
      구조 변경 필요 → **현재 PR scope 밖**, 별도 작업으로 분리.
    - 본 PR closure status: Policy B + split-K1/K2 correctness + K_loop
      cleanup. prep sync는 본질적 직렬 구간으로 acknowledged.
6. **옵션 B (버킷별 ranking horizon 사이즈) future work**: K1 ≫ K2 사용
   패턴 등장 시 별도 PR로 도입 검토. proxy forward depth는 어느 옵션에서도
   K2 불변.
