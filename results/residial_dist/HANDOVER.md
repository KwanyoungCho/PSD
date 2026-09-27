# 인수인계 — DUET P2 후보 수식 연구

**2026-09-21 재개 시 먼저 읽을 문서:** [전체 연구 진행 기록](../DUET_PROGRESS.md).
아래 9/10 시점의 미착수 상태 중 temperature·calibration 등은 후속 연구로 갱신되었다.
사용자와 함께 재개할 설명은 [미완료 TODO](training_free/TODO.md)에 남겨 두었다.

**2026-09-12 후속 작업:** [학습 없는 개선 실험 인계](training_free/HANDOVER.md),
[최신 결과](training_free/REPORT.md). 2,640개 조합과 별도 프롬프트 검증,
위치 확률 개선, 실제 wire 동점 민감도, 단독 실행 속도 비교를 추가했다.
아래는 2026-09-10 시점의 기록이다.

**최종 갱신** 2026-09-10
**브랜치** `feat/duet-proxy-source-ablation` (base `a82f7d2`, **커밋 없음 — 전부 working tree**)
**서버** `eslab19`, RTX 4090 24GB × 8 (sm_89)

이 문서만 읽으면 이어서 작업할 수 있도록 썼습니다. 실험 결과 해석은
[REPORT.md](REPORT.md), 수치 원본은 [NUMBERS.txt](NUMBERS.txt)를 봅니다.

**공유 분석 검증 추가 완료:** [shared_review/REPORT.md](shared_review/REPORT.md).
4 dataset × 2 seed의 새 256 generation에서 전체 분포를 주기적으로 저장했다.
22,990 step 중 2,995 step / 14,975 context를 후보 replay에 사용했다.
Held-out coverage는 residual 0.6056, pE 0.6426; coverage로 고른 tau=0.9
residual은 0.6103이었다. Train KL의 연속 최적 tau=1.055193은 held-out KL을
0.4637→0.4556으로 낮추지만 residual coverage는 0.5965로 감소했다.
실제 residual 누락 질량은 reject-event 가중 22.04%이며, entropy를 유지한
rank 재배치·temperature surrogate 및 동일 DS 이후 추가 후보도 검증했다.
원시 분포는 `ssd/experiments/proxy_source_ablation/probe_replay_20260910/out/`,
재현 코드·CSV·그림·검증 기록은 `results/residial_dist/shared_review/`에 있다.
계측/코드/RNG/후보 함수 일치/원시 chunk SHA 검사를 완료했고, 실험 GPU는 모두 해제했다.
이 결과는 조건부 후보 coverage이며 보정 후 전체 P1/P2 TPS 측정은 아니다.

**2026-09-10 추가 완료 — 위치별 entropy 측정:** [entropy/REPORT.md](entropy/REPORT.md).
4 dataset × 3 seed, 각 32 prompt × 256 token, 34,702 step / 173,510 context를
동일한 only-proxy chain 조건에서 새로 측정했다. 실제 verify K는 전부 4였다.
Layer 56의 draft 위치 동일 가중 `H(pE)-H(pT)` 평균은 +0.1589 nats,
절대차 중앙값은 0.3042 nats. “항상 평평해진다”는 해석은 지지되지 않는다.

- 새 계측: `ssd/ssd/engine/helpers/entropy_probe.py`, `SSD_DUET_PROBE_KIND=entropy`.
  기존 coverage probe와 같은 tap을 쓰고, 후보 평가 대신 위치별 scalar를 저장한다.
- `llm_engine.py::exit()`에서 probe의 마지막 부분 chunk를 flush하도록 보강했다.
- 재실행: `ssd/experiments/proxy_source_ablation/probe_entropy_20260910/run.py`.
  원시 NPZ는 해당 실험 `out/<dataset>_seed<seed>/`, 완료·정상성 판정은
  `*.complete.json`. 코드·데이터 hash는 `CAMPAIGN.json`.
- 재집계: `OPENBLAS_NUM_THREADS=1 ssd/.venv/bin/python results/residial_dist/entropy/aggregate.py`.
  `summary.csv`, `validation.json`, PNG/PDF 2종을 생성한다.
- 합성 분포/RNG/가변 K/flush 검사 및 GPU smoke를 통과했고, 12런 모두 tap 일치,
  확률합, 모든 prompt/step 저장을 확인했다. 실제 TP 최종 head와 동일 replica
  최종 head를 각각 참조한 통제에서도 결론이 유지됐다. TPS 측정용 probe는 아니다.

---

## 0. 30초 요약

DUET의 P2 후보 점수 `P_iv = h_i · r_i(v)`, 여기서 `r_i = normalize([p^E − p^D]₊)`가
**Mirror-SD식 `p^E` 단독 랭킹보다 나은가**를 검증했습니다.

**결론: 수식은 옳지만 실전에서는 진다.** 참값을 넣으면 +0.028 낫지만(Bayes 최적),
`p^E`로 계산하면 배포 조건(exit 56)에서 −0.038이 됩니다. 이를 평탄화나 머리 순서라는
단일 원인으로 확정하면 안 됩니다. [공유 분석 추가 검토](shared_review/REPORT.md)에서
전체 분포 replay, support 누락, 순위/확률값 모양 통제, h 및 calibration 목적을 분리합니다.
기존 67%/27%는 특정 reject 가중 진단의 수치이며, 단순 context 비중이나 전체
top-R coverage 손실의 원인별 비중이 아닙니다.

**이 브랜치는 커밋되지 않았습니다.** 이어받는 사람이 먼저 커밋 여부를 판단해야
합니다(§6).

---

## 1. 이 브랜치가 무엇을 위한 것인가

`main`은 다른 서버(RTX PRO 6000, sm_120, `/home/eslab/`) 기준입니다. 이 브랜치는
**이 4090 서버에서의 실험용**이며 두 가지가 섞여 있습니다.

1. **서버 이식** — `env.sh` 경로, 데이터셋 준비 (§3)
2. **P2 후보 수식 ablation** — 엔진 스위치 + 측정 probe (§4)

`env.sh`는 `main`에 올리면 안 됩니다. `main`의 값은 다른 서버 것입니다.

---

## 2. 원래 하려던 실험 전체 계획

사용자가 제시한 DUET 추가 구현·실험 항목입니다. **이번 세션은 (5)의 선행
조사에 해당하는 부분만 진행했습니다.**

| # | 항목 | 상태 |
|---|---|---|
| 1 | **다양한 모델 지원** — LayerSkip-Llama2-70B+TinyLlama(현재), Llama2-7B+AMD-llama-135m, Llama3-8B+Qwama-0.5B | ❌ 미착수. 체크포인트 없음(§3.4), Llama3는 vocab 128k > 32768 제약 |
| 2 | **Calibration으로 DUET parameter 산출** — 시간 축(제약) × 품질 축(목적함수), `기대이득 = hit × AL_hit / timestep`으로 최적 exit layer | 🟡 **품질 축의 재료가 이번에 생김.** probe가 전 80 레이어의 hit을 한 런에서 냅니다. 시간 축은 하드웨어 의존이라 측정 서버 필요 |
| 3 | **B>1 지원** | ❌ 미착수. `docs/duet/13,14`에 설계·리뷰 있음 |
| 4 | **Temperature에 따른 cache hit 변화** | ❌ 미착수. temp=1.0만 측정. probe는 T 재정규화로 사후 계산 가능하나 미구현 |
| 5 | **Tree 구성 재설계** — "점수 수식이 결과를 반영 못 함" | 🟢 **진단 완료.** 수식이 틀린 게 아니라 부정확한 입력에 취약한 형태. 임계값 `TVD ≈ 0.12` 정량화 |

추가로 사용자가 별도 제기한 질문이 이번 실험의 직접 동기입니다.

> "Mirror-SD에서는 target의 분포만으로 sampling해서 토큰 예측을 하는데,
> 우리처럼 draft 분포를 함께 사용했을 때 정말로 더 잘 맞추는지, 그리고
> draft만으로 만들었을 때까지 테스트하려고 한다."

---

## 3. 서버 환경 — 이어받을 때 먼저 확인할 것

### 3.1 파이썬 환경

```bash
cd /home/chokwans99/PSD/ssd
source env.sh            # 경로 진실공급원. 이 브랜치 버전은 4090 서버용
.venv/bin/python ...     # uv sync 로 생성, Python 3.12, torch 2.8.0+cu128
```

⚠️ **`uv sync`를 다시 돌리면 수동 설치분이 전부 제거됩니다.** pyproject에 없는
것들이라 prune 대상입니다. 재설치 필요:

```bash
uv pip install torchao==0.12.0 accelerate matplotlib
```

`accelerate`는 pyproject에 누락돼 있는데 `scripts/awq_calibrate.py`가
`device_map`을 쓰므로 **필수**입니다.

### 3.2 GPU

**GPU 0·1은 다른 사용자가 상시 점유**합니다. `CUDA_VISIBLE_DEVICES=2,3,4,5,6`
(target TP4 + draft 1) 조합을 씁니다. 실행 전 `nvidia-smi`로 확인하고, 죽은
런의 좀비가 남으면 GPU를 잡습니다:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 4082979 / 4082980 은 남의 프로세스 — 절대 죽이지 말 것
```

### 3.3 모델 — **경로를 틀리면 조용히 망가집니다**

```bash
--model_path          /home/chokwans99/awq_calibrated/layerskip_llama2_70b   # 원본 아님!
--quant_awq_artifact  /home/chokwans99/awq_artifacts/layerskip70b_awq_tp4
--draft_path          /data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0
```

**AWQ fold가 선행 RMSNorm을 `1/s`로 수정해 캘리브레이션 디렉터리에 저장합니다.**
원본 모델(`/data/chokwans99/models/layerskip-llama2-70B`)을 가리키면 항등식
`Wx = (W·diag(s))(x/s)`가 깨져 **쓰레기가 생성됩니다.**

이 실패는 **모든 지표 검사를 통과합니다** — `EXIT:0`, Traceback 없음, hit/AL
finite, 오히려 cache hit이 0.85로 *높게* 나옵니다(반복 토큰은 예측이 자명하므로).
이것 때문에 반나절 분량의 실험이 무효화됐습니다.

> **실행 체크리스트에 "생성 텍스트 육안 확인"을 반드시 넣으십시오.**
> `bench.py --example` 이 생성물을 출력합니다. 정상이면 문법적으로 온전한
> 영어가, 비정상이면 `\\\\\\\\ the\\\\ the` 같은 반복이 나옵니다.

### 3.4 데이터셋

`ssd/ssd/paths.py`가 `<ds>_data_10000.jsonl`을 하드코딩하는데 이 서버엔 100행
파일만 있습니다. `/home/chokwans99/ssd_datasets/processed_datasets/`에 올바른
이름의 심볼릭 링크를 만들어 두었고 `env.sh`가 그쪽을 봅니다.

- **실제 프롬프트는 데이터셋당 100개** — `--numseqs`는 100 이하만 유효
- **c4만 실제 파일**(심볼릭 링크 아님) — 2048 컨텍스트에 256토큰 생성 여유가 없는
  5개를 제외한 95행. 원본 c4는 최대 6,003 토큰이라 스케줄러가
  `prompt leaves no room for generation`으로 죽습니다
- 자세한 내용은 `/home/chokwans99/ssd_datasets/README.txt`

**없는 체크포인트**: AMD-Llama-135m, Qwama-0.5B, Llama-2-70b-chat-hf,
sglang-EAGLE-llama2-chat-70B. 계획 (1)과 베이스라인 비교에 필요합니다.

### 3.5 디스크

`/` 여유 약 149GB(96% 사용). 보존 중인 것:

| 경로 | 크기 | 지울 수 있나 |
|---|---|---|
| `~/awq_artifacts/layerskip70b_awq_tp4.rank{0..3}.awq.pt` | 69G | ❌ 정본 아티팩트 |
| `~/awq_calibrated/layerskip_llama2_70b` | 35G | ❌ **수정된 norm이 여기 있음**, 런타임이 읽음 |

---

## 4. 이번에 구현한 것

### 4.1 엔진 스위치 — `--duet_proxy_source`

P2 후보의 **토큰 순위 점수원**을 바꿉니다.

```bash
--duet_proxy_source residual   # [p^E − p^D]₊  (기본값, DUET 현재)
--duet_proxy_source proxy      # p^E 단독      (Mirror-SD식)
--duet_proxy_source draft      # p^D 단독
```

**실제 구현 지점은 `ssd/engine/helpers/p2_tree.py::chain_proxy_candidates_fixed`**
입니다. `verifier.py::_compute_and_send_proxy`의 eager 블록에도 같은 분기가
있지만 **그쪽은 죽은 코드**입니다 — `SSD_CHAIN_PROXY_GRAPH`가 기본 `1`이라
chain proxy CUDA Graph가 항상 잡히고, `verifier.py:995`에서 early return 합니다.

> 이 사실을 모르고 eager 쪽만 고쳐서, 세 arm이 완전히 동일한 지표를 내는
> 현상을 한참 디버깅했습니다. **점수원을 바꿀 때는 반드시 `p2_tree.py` 쪽을
> 고쳐야 합니다.** 그래프 캡처 로그로 확인 가능:
> `[DUET] captured target chain proxy graphs (K=[4, 8], source=residual)`

분기는 **CUDA Graph 캡처 시점**에 해석되므로 replay 비용은 arm과 무관합니다.
tree 경로는 sibling residual ladder라는 다른 구조라 같은 스위치가 안 들어가고,
`proxy`/`draft` + tree on 조합은 `config.py`에서 명시적으로 거부합니다.

### 4.2 `--duet_only_proxy` CLI 노출

`duet_only_proxy`는 config에 있었으나 bench.py에 플래그가 없었습니다(문서의
`--only-proxy`는 이 서버에 없는 외부 러너 것). 추가했습니다.

논리적 `K1=0`으로 만들어 **P1이 후보를 하나도 만들지 않게** 합니다. 그래야 cache
전체가 P2 metric에서 나와 arm 비교가 의미를 갖습니다. P1이 켜져 있으면 P2는
cache hit의 4%만 담당해(P1 0.726 vs P2 0.037) 무엇을 바꿔도 거의 안 움직입니다.

### 4.3 전 레이어 early-exit probe — `ssd/engine/helpers/exit_probe.py` (신규)

**실제 DUET 런 안에서** 매 step마다 여러 후보 정책을 **동시에** 채점합니다.

```bash
SSD_DUET_EXIT_REPLICA=1 \        # 필수 (아래 설명)
SSD_DUET_PROBE_LAYERS=all \      # 또는 "40,56,72" 같은 목록
SSD_DUET_PROBE_TOPM=64 \
SSD_DUET_PROBE_OUT=/path/out.json \
python -O bench.py ...
```

측정 내용:

- **요인 격자** 3 배분(`hhat`/`htrue`/`unif`) × 5 순위(`rE`/`pE`/`pD`/`rT`/`pT`)
  × 7 예산(R=1~64) × 80 레이어
- **이론적 상한** `ceiling` (Bayes 최적, 측정 불변식)
- **근사 품질** 각 형태와 `r_true`의 TVD
- **`p^E` 성격** 엔트로피, `p^D`와의 거리, 순위 프로파일
- **통제** 인공 잡음 주입, `top_m` 스윕
- **조건부 분해** `overlap = 1 − TVD(p_T,p^D)` 구간별

구현상 주의점:

1. **레이어 tap은 CUDA Graph 캡처 전에 설정해야 합니다.** `model_runner.py::
   _setup_exit_probe()`가 `allocate_kv_cache()` 직후, 그래프 캡처 직전에
   버퍼를 붙입니다. 캡처 후에 설정하면 복사가 그래프에 안 들어갑니다.
2. **`SSD_DUET_EXIT_REPLICA=1`이 필수입니다.** probe는 rank0에서만 전 vocab
   logits을 계산하는데, `compute_logits`는 **TP collective**라 rank0 단독 호출 시
   다른 rank가 gather에서 600초 timeout으로 죽습니다. rank0 lm_head 복제본
   (`_duet_lm_head_replica`)을 써서 collective 없이 계산합니다.
   설정 안 하면 `_setup_exit_probe`가 명시적으로 raise 합니다.
3. **prefill 가드** — 레이어 loop은 prefill에서도 돌고 행 수가 훨씬 많습니다
   (예: 308행 vs verify 9행). 버퍼를 넘으면 건너뜁니다.
4. **자체 검증** — `cudagraph_helpers.py`가 첫 step에서 probe의 exit-layer tap이
   엔진 `exit_hidden`과 일치하는지 확인해 출력합니다:
   `[DUET probe] tap check @layer=56 max|probe-engine|=0.000e+00 OK`
   **MISMATCH가 뜨면 tap 규약이 어긋난 것이므로 모든 수치가 무효입니다.**
5. probe는 느립니다(80 레이어 기준 step당 수십 ms). 성능 측정과 섞지 마십시오.

### 4.4 변경 파일 목록

```
ssd/ssd/engine/helpers/exit_probe.py   신규 — 측정 로직 전체
ssd/ssd/engine/helpers/p2_tree.py      후보 점수원 분기 (실제 동작 지점)
ssd/ssd/engine/model_runner.py         probe 버퍼 setup, 그래프에 arm 전달
ssd/ssd/engine/verifier.py             probe 훅, eager 경로 분기
ssd/ssd/models/llama3.py               layer loop tap
ssd/ssd/engine/helpers/cudagraph_helpers.py   tap 자체 검증
ssd/ssd/config.py                      duet_proxy_source 필드 + 검증 + tree 가드
ssd/bench/bench.py                     --duet_proxy_source, --duet_only_proxy
ssd/env.sh                             4090 서버 경로  ← main 에 올리지 말 것
```

---

## 5. 실험 결과와 해석

상세는 [REPORT.md](REPORT.md). 여기서는 이어받는 데 필요한 것만 씁니다.

### 5.1 무엇을 쟀는가

엔진은 기각 시 recovery 토큰을 `normalize([p_T − p^D]₊)`에서 뽑습니다
(`ssd/utils/verify.py:174`). 예산 R개를 캐싱했을 때 **적중 확률의 기댓값**

```
U(C) = Σ_{(i,v) ∈ C}  h_true[i] · r_true[i][v]
```

를 씁니다. 확률의 기댓값이라 샘플링 잡음이 없고, 정책들이 같은 step을 보므로
**paired 측정**입니다.

`U`는 격자 위 top-R 선택이므로 최적해는 `h_true ⊗ r_true`, 즉 **DUET 수식에
참값을 넣은 것**입니다. 따라서 우리 수식은 이 목적함수의 Bayes 최적이며,
이를 `ceiling`으로 기록해 불변식으로 씁니다.

### 5.2 핵심 수치 (exit 56, R=15, 4 데이터셋 × 3 seed = 34,150 step)

```
수식의 이득 (참값)          +0.0280      12/12 양수
우리 형태의 ε 비용          −0.1588
Mirror-SD 형태의 ε 비용     +0.0717
─────────────────────────────────────
순 결과                     −0.0591      항등식 잔차 0.00e+00

배포 조건(ĥ 사용) 격차      −0.0383      12/12 음수
교차 레이어                 74 ~ 77      (93~96% 지점)
```

**같은 오차인데 우리 형태가 2.2배 더 손해**를 봅니다.

### 5.3 원인

`p^E`는 `p_T`보다 평평하고(엔트로피 +0.178), 무엇보다 **머리 순서가
불안정합니다** — 56층에서 `p_T`의 1위 토큰이 `p^E`에서도 1위인 경우는 **67%**.

`p^D`는 `p_T`의 머리와 상관되어 있어 빼면 머리가 깎이는데, `p^E`가 꼬리에 잘못
얹은 질량은 남아 순위 위로 올라옵니다. Mirror-SD식은 `p^E`의 순서를 그대로
쓰므로 이 함정을 피합니다.

### 5.4 어디서 지는가 — 가장 실행 가능한 발견

```
overlap = 1 − TVD(p_T, p^D)     질량    차이
0.00 - 0.25                     8.2%   +0.0177   이김
0.25 - 0.50                    22.1%   +0.0181   이김
0.50 - 0.75                    42.5%   +0.0094   이김
0.75 - 1.00                    27.2%   −0.0441   짐
```

**73% 구간에서 이기고, draft가 target을 잘 흉내내는 27%에서만 집니다.**
승패가 무작위가 아니라 **관측 가능한 축으로 갈립니다.**

### 5.5 통제 실험 (다른 설명 배제)

| 통제 | 결과 |
|---|---|
| 위치 배분 (`unif` 강제) | 여전히 짐 (−0.032) → 배분 탓 아님 |
| `top_m` 16/32/64 | 격차 불변 → 절단 아티팩트 아님 |
| 인공 잡음 주입 | TVD ≤ 0.36 에서 실제 곡선과 겹침 → early exit 특유 성질 아님 |

### 5.6 반드시 알아야 할 한계

1. **`only-proxy` 조건입니다.** champion이 아닙니다. P1이 켜지면 P2는 cache의
   일부만 담당해 효과가 크게 희석됩니다. **champion 조건 재확인 필요.**
2. **hit이 속도로 옮겨간다는 보장이 없습니다.** 엔진 실험에서 P2 hit이
   0.556~0.639로 갈렸는데 tokens/step은 2.87~2.88로 동일했습니다.
   `docs/duet/12-experiment-summary.md`의 첫 finding과 일치합니다.
3. **draft가 TinyLlama-1.1B 고정.** §5.4가 직접 예측하듯 **더 강한 draft면
   우리가 더 불리해집니다.** 계획 (1)의 다른 조합에서 재확인 필요.
4. TVD > 0.4 구간에서 인공 잡음 곡선과 실제가 벌어지는데 설명되지 않았습니다.

---

## 6. 이어받는 사람이 먼저 결정할 것

### 6.1 커밋 여부

**working tree에 미커밋 상태입니다.** 8개 파일 수정 + 2개 신규(213줄 추가).
`env.sh`는 서버 전용이라 분리해야 합니다. 권장:

```bash
# 실험 코드만 커밋, env.sh 는 제외
git add ssd/ssd/ ssd/bench/bench.py ssd/experiments/proxy_source_ablation results/
git commit -m "feat(duet): P2 candidate-source ablation + all-layer exit probe"
# env.sh 는 서버 로컬로 남겨두거나 별도 커밋
```

### 6.2 `.gitignore` 확인

실험 산출물이 큽니다. `probe_*/out/*.json`은 개당 200KB~1MB, log는 44KB 수준이라
현재는 커밋해도 괜찮지만, 대규모 스윕을 하면 커집니다. 저장소 관행상 raw dump는
gitignore 대상입니다(커밋 `912f0f2` 참조).

---

## 7. 다음에 할 수 있는 일 (우선순위 순)

### 7.1 champion 조건 재확인 — **가장 중요**

지금 결론은 `only-proxy`에서만 검증됐습니다. P1을 켠 champion에서 같은 방향인지
확인해야 논문에 쓸 수 있습니다. probe는 그대로 쓰면 되고 `--duet_only_proxy`만
빼면 됩니다.

### 7.2 조건부 전환 설계

§5.4가 27% 구간을 특정했습니다. `overlap`이 큰 위치에서만 Mirror-SD식으로
전환하면 양쪽 이점을 취할 수 있습니다.

⚠️ 런타임에는 `p_T`를 모르므로 `TVD(p^E, p^D)`로 overlap을 추정해야 하는데,
**그 추정도 같은 `p^E` 오차를 겪습니다.** 추정 overlap과 참 overlap의 상관을
먼저 재야 합니다. probe에 몇 줄 추가하면 됩니다.

### 7.3 ε에 강건한 점수 형태

뺄셈 대신 곱셈형(`p^E · (1−p^D)^λ` 등). probe 격자의 `RSRC`에 후보를 추가하면
**같은 런에서** 평가됩니다. `exit_probe.py`의 `cand` 딕셔너리에 항목을 넣고
`RSRC` 튜플에 이름을 추가하면 끝입니다.

### 7.4 계획 (2) 캘리브레이션 — 품질 축 완성

probe가 전 80 레이어의 hit을 한 런에서 냅니다. 시간 축(proxy 도착 시각, draft
forward 수)만 측정 서버에서 재면 `기대이득 = hit × AL_hit / timestep`을 완성할
수 있습니다.

### 7.5 계획 (4) temperature 스윕

probe는 `p_T`를 가지고 있으므로 온도 재정규화로 사후 계산이 가능합니다.
현재 미구현. `observe()`에 T 루프를 넣으면 한 런에서 전 온도가 나옵니다.

---

## 8. 실행 예시 (복붙용)

### 8.1 정상성 확인 (3분)

```bash
cd /home/chokwans99/PSD/ssd/bench && source ../env.sh
CUDA_VISIBLE_DEVICES=2,3,4,5 ../.venv/bin/python -O bench.py \
  --llama --size 70 --gpus 4 \
  --model_path /home/chokwans99/awq_calibrated/layerskip_llama2_70b \
  --draft_path /data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0 \
  --quant_awq --quant_awq_artifact /home/chokwans99/awq_artifacts/layerskip70b_awq_tp4 \
  --b 1 --temp 0 --numseqs 2 --output_len 48 --example
# GENERATIONS 절의 텍스트가 온전한 영어인지 눈으로 확인할 것
```

### 8.2 3-arm 엔진 실험

```bash
cd /home/chokwans99/PSD/ssd
ONLY_PROXY=1 SEEDS=42 bash experiments/proxy_source_ablation/temp1_3arm_20260908/run.sh
.venv/bin/python experiments/proxy_source_ablation/temp1_3arm_20260908/aggregate.py
```

### 8.3 전 레이어 probe

```bash
cd /home/chokwans99/PSD/ssd
bash experiments/proxy_source_ablation/probe_ds_seed_20260909/run.sh
.venv/bin/python experiments/proxy_source_ablation/probe_ds_seed_20260909/aggregate.py
```

### 8.4 보고서 재생성

```bash
cd /home/chokwans99/PSD/results/residial_dist
/home/chokwans99/PSD/ssd/.venv/bin/python make_numbers.py > NUMBERS.txt
/home/chokwans99/PSD/ssd/.venv/bin/python make_figs.py
/home/chokwans99/PSD/ssd/.venv/bin/python make_figs_rank.py
```

---

## 9. 이 세션에서 밟은 함정 (반복 방지)

| 함정 | 증상 | 교훈 |
|---|---|---|
| `--base-model`을 원본 모델로 지정 | 모든 지표 통과, cache hit 오히려 상승, 생성물은 쓰레기 | **텍스트를 눈으로 볼 것** |
| eager 경로만 수정 | 세 arm 지표가 완전히 동일 | 실제 동작은 CUDA Graph 안 (`p2_tree.py`) |
| rank0에서 `compute_logits` 호출 | 600초 NCCL GATHER timeout | TP collective — replica 사용 |
| `bench.py` 종료 코드로 성공 판정 | 예외 후에도 exit 0 → c4 3런이 조용히 실패 | **산출물 존재로 판정** |
| `torch.bucketize` 경계 | "1위 일치율 0.0%" | 명시적 비교로 대체 |
| probe의 `randn_like` | 전역 RNG 오염 → 궤적 변화 | 별도 generator 사용 |
| `numbers.py` 파일명 | numpy import 실패 | stdlib 모듈명 회피 |
| `uv sync` 재실행 | torchao/accelerate/matplotlib 제거 | 매번 재설치 |

---

## 10. 참고 문서

| 문서 | 내용 |
|---|---|
| [REPORT.md](REPORT.md) | 이번 실험의 전체 분석 (그래프 10종 포함) |
| [NUMBERS.txt](NUMBERS.txt) | 인용된 모든 수치, 데이터셋 × seed 원값 |
| `ssd/MESA-SSD.md` | DUET 방법 명세 — `P_iv = h_i·r_i(v)` 정의는 §4.2 |
| `ssd/docs/duet/00-server-setup.md` | 서버 셋업. **단, 구 서버(sm_120) 기준** |
| `ssd/docs/duet/12-experiment-summary.md` | 과거 실험 색인. 첫 finding("hit이 토큰으로 안 옮겨간다")이 §5.6-2와 직결 |
| `ssd/docs/quantization/03-final-report.md` | AWQ 경로. §4.2에 70B 캘리브레이션 이력 |
