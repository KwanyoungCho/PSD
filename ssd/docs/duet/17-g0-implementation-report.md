# 17 — G0 준비 구현 리포트 (인자 정리 → P0 → E0)

**브랜치**: `feat/duet-p2tree-g0` (base: `feat/mesa-proxy-async-overlap`
@ e29c4b6 = 설계 v6). **목적**: 사용자가 나중에 판단할 수 있도록, 승인된
구현(16번 인자 정리, P0 기록 게이트, E0 수집)의 **모든 이슈·구현 내용·
결과를 이 문서에 기록**한다. 판정 원칙은 설계 v6과 동일 — 자동 문턱
없음, 종합 지표 보고 후 사용자 판단.

**기준 config (사용자 지정)**: layerskip-llama2-70B (AWQ W4A16 TP4,
`/data2/chokwans99/awq_calibrated/layerskip_llama2_70b`) + TinyLlama-1.1B
(AWQ, `.../tinyllama_1b`), champion **E9K24_jit** — K1=9 K2=4 (k=13),
exit=56, dfo=2 pfo=1 (f=3), P1 fan_out_list `2,2,2,2,2,2,1,1,1,1`,
jit-short ON, GPU 0-4 (target TP4 = 0-3, draft = 4), temp 0.7, seed 42.
모든 검증·스모크는 이 형상 기준.

**범위** (승인된 것만): ① 인자 정리 (16번 Tier 1→2→3, Tier별 커밋)
② P0 — E0 이중 trace 기록 게이트 ③ E0 수집 런. P2-tree 본체(T1~)는
이 브랜치 범위가 아니다 — 설계 승인 후 별도.

---

## 1. 인자 정리 (16번 계획 실행)

### 1.1 Tier 1 — CLI 별칭 (커밋 5d2c534)

**구현**: bench.py에 canonical 이름 신설 — `--duet_k1/--duet_k2`
(구 `--duet_phase1_k/phase2_k`), `--duet_p1_fanout`
(구 `--duet_draft_fan_out`), `--duet_p1_fanout_list`
(구 `--duet_split_phase1_fan_out_list`). 구 플래그는 그대로 동작하되
deprecation 한 줄 출력; 신·구를 다른 값으로 같이 주면 즉시 에러.
`--k` 생략 시 K1+K2로 유도 (명시하면 config가 일치 검증). 데드 플래그
2개 삭제: `--fl`(코드 전체에서 미사용 확인 — fan_out_list는
`--flh/--flm`이 담당), `--duet_split_phase2_fan_out_list`(넘기면
config가 NotImplementedError로 거부하던 인자).

**검증**: 신 명령 파싱 → k=13 유도·필드 정상 / 구 명령 → 동일 필드 +
deprecation 출력 / 충돌 명령 → 에러 확인 (CPU).

### 1.2 Tier 2 — env 처분 (커밋 d31db27 직전 커밋)

**구현**:
- `SSD_FORCE_SPLIT_K1K2` **은퇴**: split-K1/K2가 유일 경로이므로
  `--duet`이 곧 split — Config가 env를 자동 설정 (spawn 자식 프로세스가
  environ을 상속하므로 기존 11개 읽기 지점은 과도기 동안 무변경 동작).
  "env 없이 실행" 시 기존과 완전 동일함을 필드 전수 비교로 확인.
- `SSD_DUET_JIT_SHORT` → **config 필드 `duet_jit_short` 승격, 기본 ON**
  (champion 표준). 해제는 `--duet_no_jit_short`. env를 명시로 준 구
  스크립트는 env가 이기며(deprecation 출력) 확정값을 재-export해서
  draft 프로세스의 import-시점 읽기도 정합.

**주의 (동작 변경 1건, 계획된 것)**: 과거에 env 없이 DUET을 돌리면
jit-short OFF였으나 이제 기본 ON. 구 재현 스크립트가 env를 명시(0/1)
했다면 영향 없음; 명시 안 한 비-jit-short 재현은 `--duet_no_jit_short`
필요.

### 1.3 Tier 3 — config 단일화 + `--duet_p2_budget` (커밋 d31db27 + 후속)

**구현**:
- `duet_p2_budget` 필드/플래그 신설 — 미지정(None)이면 기존 산식
  `pfo×(K_max+1)` 그대로 (재현성 100%), 지정하면 그 값이 단일 소스.
  `--f` 생략 + budget 지정 시 f = p1_fanout + ⌈budget/(K1+1)⌉ 유도.
- budget 산식 증복 정리 (5곳): config wire_N·__post_init__ /
  scheduler 예약(:58) / model_runner CG 사이징 / draft_runner split_k2
  layout(합-보존 분배 리스트 — 기본값에서 `[pfo]*(K_max+1)`와 정확히
  동일) → 전부 `config.duet_proxy_total_budget` 단일 소스.
- fan_out_list 이중 유도 제거 (model_runner의 재유도는 config
  __post_init__가 항상 선행하므로 데드 — 삭제).
- K2≤K1 검증을 config `__post_init__` 명시 raise로 이동 (조기 실패;
  DraftRunner의 기존 검사는 방어로 유지).

**이슈 #1 — verifier의 산식은 같은 산식이 아니었다**: 16번 진단은
"budget 산식 5중복"으로 봤지만, verifier의 `pfo×(K+1)`은 **K가 스텝
가변**(그 step의 vk_max — P1-hit이면 9, P2-hit/miss면 4)이라 상수
budget과 다른 양이다. 상수로 치환하면 short step의 proxy 배분이
바뀐다 → 별도 헬퍼 `duet_p2_budget_at(K)`로 분리 (기본값에서
`pfo×(K+1)` 정확 재현; 직접 budget 시 위치수 비례 스케일). **단순
치환했으면 조용한 동작 변경이 났을 지점.**

**이슈 #2 — property의 호출-시점 env 결합 (유닛테스트가 검출)**:
`duet_proxy_total_budget`/`duet_proxy_wire_N`이 호출 시점의
`SSD_FORCE_SPLIT_K1K2`를 읽어서, env가 사라지면 같은 Config 객체가
다른 값(14 vs 10)을 돌려줬다. split 판정을 필드 기반(`duet_phase1_k
is not None`)으로 교체 — DUET 실행에서는 env가 필수였으므로 실사용
값 변화 없음, 파생값이 자기완결적이 됨.

### 1.4 동일성 검증

- **유닛테스트 신설** `tests/test_args_cleanup_equiv.py` (6개, champion
  70B+TinyLlama config 기준): env-free 신방식 ≡ env 구방식 (전 필드
  diff 없음) / budget=10 직접 ≡ 파생 (total·wire_N·top_k·
  budget_at(4/9)) / budget_at 기본 산식 / env=0 명시 우선 + 재-export /
  K2>K1 조기 raise / K1·K2 누락 raise → **6/6 통과**.
- **기존 CPU 테스트 회귀**: b_gt1 계열 (m1·m2·m3·m4·m6·jit_subset)
  **44/44 통과** (m6 스텁에 신설 헬퍼 1줄 추가 — 스텁 갱신, 동작 무관).
- **기존부터 깨져 있던 테스트 (제 변경과 무관, 미수정)**:
  `tests/test_policy_b_unified_padded.py` 4/4 에러 — base 커밋
  e29c4b6에서도 동일 재현. M1 배칭 이전의 1-D 형상을 넘기는 레거시
  테스트. 처분은 사용자 판단 대상 (갱신 vs 삭제).
- **champion GPU 스모크 (구 명령 vs 신 명령)** — 완료
  (`experiments/proxy_async_overlap/g0_args_cleanup/smoke_*.log`;
  numseqs 8 ×4셋, out 256, temp 0.7, seed 42 — 동등성 확인용 짧은 런,
  TPS 판정용 아님):

  | 런 | decode TPS | tok/step | T_full(ms) | T_verify(ms) | draft(ms) |
  |---|---:|---:|---:|---:|---:|
  | 구-방식 1회차 | 75.68 | 3.74 | 52.23 | 45.59 | 44.65 |
  | 구-방식 2회차 | 77.91 | 3.91 | 53.06 | 46.40 | 44.78 |
  | 신-방식 | 82.96 | 4.17 | 53.40 | 46.42 | 45.11 |

  **판독**: Config 출력은 세 런 모두 동일 (exit=56, top_k=14, dfo=2,
  pfo=1, K1=9, K2=4; 신-방식은 k=13 유도 포함). **시간 지표는 세 런이
  구간 내 동일**하고, TPS 차이는 전부 tok/step — temp 0.7 짧은 런의
  토큰 샘플링 운 (async 레이스로 같은 seed여도 hit 경로가 갈림; 구-
  방식끼리도 75.68 vs 77.91). 엔진 행동 차이의 증거 없음. 정밀 판정이
  필요하면 5-rep 인터리브로 재실측 가능 (사용자 요청 시).

## 2. P0 — E0 기록 게이트 (커밋 48c606f + 75f6aac)

**구현**: 독립 모듈 `ssd/engine/helpers/e0_trace.py` — 엔진에는 게이트
분기 4곳만 남기고 (verifier 1 + draft_runner 3) 전 로직을 모듈에 격리
(통째 제거 가능). 게이트 `SSD_DUET_E0_TRACE=1`, 기본 OFF.

- **target "wire"** (step당 1): 전체 wire 후보 (pos, tok) + **raw P_iv
  값** + 충분통계 — 위치별 y-logit(exit/draft), 정확한 lse@temp1,
  top-32 exit/draft logits, 후보별 exit logit. temp-정합 P_iv를
  오프라인에서 재계산 가능 (설계 v6 E0 ④; top-M 근사는 설계 허용).
- **draft**: "request"(직전 step 실제 outcome = cache key + temp),
  "response"(phase_source/valid_k/응답 토큰), "selector"(파싱된 wire =
  rank 순 + dedup 후 잔존 P2 seed + 위치별 fan-out — 원 rank 재구성
  가능).
- **오버헤드 설계**: pinned-host 비동기 복사 + CUDA event + 백그라운드
  스레드 직렬화 (임계경로 sync 0회), queue 넘침은 drop 카운트 (블로킹
  금지), SUBSAMPLE/DIR/TOPM env 옵션. **사용자 조건 이행**: TPS 측정
  런과 완전 분리 (전용 런 전용, ON 런 TPS는 보고 금지), OFF 비용 =
  모듈 bool 분기, 독립 모듈이라 통째 삭제 가능.

**검증**:
- 유닛테스트 `tests/test_e0_trace.py` 4/4 (기본 OFF / 스키마·값 정합
  (P_iv·후보 logit 수치 재계산 일치) / 서브샘플 / drop=0 summary) +
  기존 회귀 44/44 유지.
- GPU: P0 코드 포함 + 게이트 OFF 런 정상 (TPS 74.60 — 스모크 밴드 내,
  §1.4와 동일 노이즈 규모). 게이트 ON 검증 런에서 trace 산출 확인.

**이슈 #5 — 파일 버퍼 꼬리 유실 (ON 검증 1차에서 발견)**: 엔진이
hard-exit라 atexit이 안 돌고, 1MB 블록 버퍼에 남은 draft 꼬리 ~700
step이 유실됐다 (target 2134 vs draft 1438; draft step_id가 1..1438
연속 + 뒤가 통째로 없는 패턴 + K 분포 비율은 양쪽 일치 = 꼬리 절단의
전형). **수정**: 라인 버퍼링(쓰기는 백그라운드 스레드라 비용 무관) +
1000-레코드마다 heartbeat(drops 지속 기록 — 위생 "drop=0"을 비정상
종료에서도 검증 가능). **재검증: target 2188 == draft
request/response/selector 2188 완전 정합.**

## 3. E0 수집 런 — 완료 + 분석 (2026-08-03)

**런**: champion 형상 (numseqs 50×4셋=200 seq, out 512, temp 0.7,
seed 42), 전용 계측 (`e0_collect/run1/`, target 210MB + draft 18.5MB —
git 비추적). exit 0.

### 3.1 무결성 (전부 통과)

- 레코드 정합: target wire **25,062** == draft request/response/selector
  **25,062** (완전 일치). heartbeat drops **0/0** (양측).
- phase 분포: miss 0.179 / P1 0.571 / P2 0.250 — champion 공인
  (0.15-0.18 / ~0.53 / 0.24-0.27)과 일치.
- 수락 길이 (공인 정의 = recovery 제외): **L_p1 4.07 / L_p2 1.79 /
  L_miss 1.77**, 전체 tok/step(포함 기준) 4.086 — 공인 (L_p2
  1.76~1.81, tok/step 4.108)과 일치. **데이터셋이 champion 레짐을
  정확히 재현함을 확인.**

### 3.2 판정 지표 (설계 v6 E0 — 보고만, 판정은 사용자)

**① 위치별 P_iv 신뢰도 곡선** (raw P_iv 구간별 실측 hit 빈도;
24,862 step × 28 후보):

| P_iv 구간 | n | 구간 중앙 | 실측 hit |
|---|---:|---:|---:|
| [0.3, 1) | 20,422 | ≈0.66 | 0.488 |
| [0.1, 0.3) | 29,572 | ≈0.20 | 0.132 |
| [0.03, 0.1) | 69,951 | ≈0.065 | 0.040 |
| [0.01, 0.03) | 114,595 | ≈0.02 | 0.0133 |
| [0.003, 0.01) | 149,507 | ≈0.0065 | 0.0053 |
| ≤0.003 (4구간) | 312,089 | — | 예측보다 높음 (~2-4×) |

근사 ECE 0.012, Brier 0.019. **요약: raw P_iv는 4자리수 스케일에
걸쳐 순서·크기 모두 유의미** — 상단은 ~1.3-1.6× 과신, 하단은 2-4×
과소 (완만한 압축형 왜곡).

**② 위치-종류별 비균일 편향** (설계 §12 TODO 2의 발동 조건 관측):
거부-위치 후보(pos 0-2 vs 3+)끼리는 편향이 거의 같으나, **전부-수락
(pos=K) 후보는 같은 P_iv에서 실측 hit이 ~2배** (예: P_iv∈[0.3,1)에서
pos0-2 0.414 vs pos=K **0.705**; [0.1,0.3)에서 0.117 vs 0.212).
즉 raw P_iv는 bonus-후보를 계통적으로 ~2× 과소평가 — **비균일 편향
실재**. 처리는 사용자 유보 결정(보정계수 후속 도입 여부)의 입력.

**③ root coverage Recall@R** (dedup 전, wire 상위 R):
R=4 **0.632** / R=6 0.677 / R=8 0.704 / R=10 **0.721** / R=28(전체)
0.779. R 4→10 한계 +8.9pp, 10→28 +5.8pp.

**④ rank별 hit 확률** (wire 순위 = P_iv 내림차순): rank1 **0.416**,
rank2 0.121, rank3 0.059, rank4 0.036, … rank10 0.008 — **매우 가파른
순서 신호** (π̂=라이브 P_iv 결정 ③·root 예산 P_iv-비례 결정 ⑤v2를
데이터가 지지).

**⑤ 배분 분포 (현행 체인 baseline; 결정 ⑤v2 쏠림 모니터링용)**:
전 budget(10)이 한 위치에 몰리는 step이 존재 (pos0 몰빵 5.8%,
[10,0,...] 계열 상위 3패턴 합 ~11%) — 트리 전환 후 같은 지표로 재관측
예정.

**산출물**: 분석 스크립트는 이 문서의 §3 수치 재현 커맨드로 보존
(trace 파일 + 위 표). 다음 단계(E1 feasibility — offline 시뮬레이션,
엔진 무수정)는 이 데이터셋을 입력으로 하며 **사용자 승인 후 착수**.

## 이슈 로그 (전체)

| # | 발견 경위 | 내용 | 처리 |
|---|---|---|---|
| 1 | Tier-3 구현 중 코드 정독 | verifier의 budget 산식은 K가 스텝-가변이라 다른 4곳과 다른 양 — 상수 치환 시 조용한 동작 변경 | `duet_p2_budget_at(K)` 헬퍼 분리, 기본값 정확 재현 |
| 2 | 신설 유닛테스트 실패 | config property가 호출-시점 env에 결합 — env 부재 시 같은 객체가 다른 budget 반환 | split 판정을 필드 기반으로 교체 |
| 3 | 회귀 실행 | test_policy_b_unified_padded 4건 에러 — base 커밋에서도 동일 (기존 결함, 레거시 1-D 형상) | 미수정, 사용자 판단 대기 |
| 4 | Tier-2 설계 | jit-short 기본값 ON 전환은 "env 미명시 구 명령"의 동작을 바꿈 (계획된 변경) | env 명시 시 호환 우선 + deprecation 출력으로 완화 |
