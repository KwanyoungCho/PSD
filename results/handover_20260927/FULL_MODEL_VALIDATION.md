# 실제 실험 서버에서의 재개 절차

이 문서는 실행 순서와 판정 기준이다. 새 서버의 GPU/모델 경로는 아직 정해지지
않았으므로 **검증하지 않은 full-model 실행 명령을 완성본처럼 제공하지 않는다.**
기존 명령은 `results/duet_tree_al_full/runs/s1_q_path/command.json`에 정확히 보존했다.
아래 이식 작업 후 새 campaign에 명령·env·model manifest를 기록한다.

Full model은 우선 dense 원본 LayerSkip Llama2-70B BF16/FP16으로 해석한다.
사용자가 다른 뜻으로 정하면 checkpoint/precision을 먼저 고정한다. 기존 full480
평가를 다시 하는 것과 target 정밀도를 바꾸는 것은 서로 다른 실험 축이다.

## 0. 수령 및 새 campaign 생성

```bash
git clone --branch feat/duet-proxy-source-ablation https://github.com/KwanyoungCho/PSD.git
cd PSD
git rev-parse HEAD
python3 results/handover_20260927/manage_artifacts.py verify --group git
```

원시 archive를 받은 뒤에는 [ARTIFACTS](ARTIFACTS.md)의 restore/verify를 수행한다.
Git만 받은 상태에서도 보고서, 핵심 수치, full online records는 읽을 수 있다.
Full-pqe 재계산/사후 branch 분석에는 별도 archive가 필요하다.

- 새 결과는 예를 들어 `results/duet_fullmodel_<date>_<target>_<dtype>/`에 기록한다.
  기존 frozen directory나 `runs/smoke_q_path`에 이어 쓰지 않는다.
- `results/duet_tree_followup/STOP`은 역사적 캠페인에 남긴다. 새 runner는 새 위치의
  STOP/lock/status를 사용한다. 기존 queue를 재개할 목적의 삭제는 하지 않는다.
- Run ID마다 source commit, 모델/토크나이저 hash, dtype, TP, GPU UUID, driver,
  CUDA/package versions, attention backend, env, command, seed, plan hash를 기록한다.
- 실행 실패는 기존 attempt를 보존하고 다음 `attemptN`으로 남긴다. 실패한 결과를
  통계의 성공 row에 넣거나 완료로 표시하지 않는다.

## 1. 환경과 메모리의 feasibility

1. `ssd/pyproject.toml`/`uv.lock` 및
   `ssd/docs/duet/00-server-setup.md`를 읽고 새 venv를 만든다. 이전 `.venv`는
   `/home/chokwans99/anaconda3/bin/python3`에 연결돼 있으므로 복사해 쓰지 않는다.
   Python은 프로젝트 범위(3.11 이상, 3.13 미만)를 따른다. 기존 package snapshot과
   차이를 기록하고 커널을 해당 GPU arch로 빌드한다.
2. `ssd/env.sh`는 old eslab19 전용 export다. 새 local env를 만들고 실제 모델,
   dataset, `SSD_CUDA_ARCH`, backend를 지정한다. 환경변수 이름은 CODE_MAP 참조.
3. 스케줄러 또는 장비 할당으로 GPU를 확보한다. `nvidia-smi` idle 확인만으로
   예약됐다고 간주하지 않는다. 9/23 OOM은 바로 이 경쟁 상황에서 발생했다.
4. Dense70B BF16은 가중치만 대략 140GB(decimal) 규모다. 실제 TP 배치는 KV,
   head replica, temporary load tensors, executor/verify workspace, 모든 graph
   bucket을 포함해 측정한다. 특정 GPU 개수면 반드시 들어간다고 가정하지 않는다.
5. DUET `--gpus N`은 통상 target TP=N−1, 마지막 한 장 draft 배치다. 새로운 TP가
   attention/KV head 분할, custom kernel, GPU topology 제약을 만족하는지 확인한다.
6. Dense 실험에서는 AWQ model/artifact 경로와 `--quant_awq`,
   `--quant_awq_artifact` 등을 제거하고 실제 dense checkpoint를 지정한다.
   Flag를 지우는 것만으로 원본 가중치가 보장되지는 않으므로 로딩 dtype와 key,
   model config를 로그에 남긴다. Draft도 quantized로 바꾸면 별도 축이다.
7. 먼저 target-only → draft-only → 기존 chain → G=M tree baseline 순으로
   로딩/prefill/decode/capture를 확인한다. 최대 prompt, 모든 P1 root bucket,
   P2/verify rows, workspace 및 peak allocated/reserved VRAM을 기록한다.

Gate: 모델 종류/정밀도에 모호함이 없고, longest-context + graph warmup에서
OOM/NCCL hang 없이 baseline이 실행돼야 한다. `--max_model_len`을 줄이거나
입력을 잘라서 기존 full corpus를 “통과”시키지 않는다.

현재 wire vocab cap은 32,768이다. 128k vocab 모델로 바꿀 때는 tokenizer만 바꾸면
되지 않는다. B=1과 Llama2/TinyLlama의 검증 범위를 벗어난 모델/B>1은 별도 port다.

## 2. 올바른 실행과 lossless 조건을 먼저 확인

### CPU 및 작은 GPU fixture

CPU smoke는 원본 결과를 덮어쓰지 않는 인계 도구로 실행한다.

```bash
ssd/.venv/bin/python results/handover_20260927/validate_handoff.py --output /tmp/duet_handoff_cpu_check
```

이 도구는 연구 코드 문법, entropy unit check, CPU C=3/배분/분포 보존/selector/mask
검사를 수행하고 별도 output에 결과를 쓴다. Torch/Numpy 환경이 필요하다.
Git/원시 데이터 검증은 별도 `manage_artifacts.py`가 담당한다.

한 GPU에서 새 campaign용 복사본의 `check_cuda.py`와 `check_executor.py <policy>`를
여섯 정책 각각 **새 프로세스**에서 실행한다. 원래 script는 자기 디렉터리에
결과를 쓰므로 frozen 결과 위치에서 직접 다시 실행하지 않는다. Arch/cache env를
명시한다. 통과 기준은 CPU/eager/capture allocation 동일, token/parent/sibling/
q-reference/valid mask 동일, logits/raw-q 오차가 정한 범위 내인 것이다.

### 실제 full-model smoke

여섯 정책 모두 원래 계획의 task별 대표 prompt + 최장 prompt로 짧게 실행한다.
`DUET_FULL_SMOKE=1`, `DUET_FOLLOWUP_PARITY=1`의 hook 의미를 보존한 fresh runner가
필요하다. `run_full.py`의 tokenizer 절대경로, fixed temperature, plan 경로를 먼저
고친다. Hook이 spawned worker에도 capture 전에 설치되는지 로그로 확인한다.

다음 항목을 각각 기록한다.

- P1/P2 **양쪽** actual-model eager vs graph, 동일 고정 sampling noise 비교.
- P1 roots9/21/27, P2 roots15, 여러 prefix page bucket, bucket 왕복 replay.
- `tok`, `parent_local`, `sib_order`, `parent_q_ref`, `parent_q_cells`, validity,
  realized fanout, graph/eager proposal q의 대응.
- Mixed-depth frontier의 position/RoPE, prefix 및 ancestor-only attention mask,
  uninitialized/padding lane 배제, 과거 round의 logits/q reference 유지.
- Root hit 후 root 자체를 다시 proposal로 검사하지 않는지, descendants는 실제
  parent q와 WOR exclusion으로 검증하는지, target residual을 올바르게 갱신하는지.
- G=M, sibling prefix/ancestor closure, reserve/quota/node budget, zero fanout,
  ties, proposal zero mass, residual zero mass, EOS/cap, immediate EOS.
- Prompt/history hash, tokenizer special token 처리와 실제 이전 turn 출력 사용.
- Probe 켰을 때 exit hidden self-check와 final layer p가 맞는지. Probe off 성능
  캠페인에서는 저장·진단 flag가 모두 꺼졌는지.

Gate: 두 phase의 parity 기록과 구조 audit가 있고 모든 필수 shape가 통과해야 한다.
실패하면 full480을 먼저 돌려 수치를 얻지 말고 원인을 고친다. Tiny fixture의
6×8 통과는 이 gate를 대체하지 않는다. Monte Carlo output frequency가 p와
가깝다는 것은 보조 검사다. Proposal/sampling 식의 검증과 implementation 대응도
필요하며, 같은 seed의 서로 다른 sampler 출력이 token별 동일할 필요는 없다.

**G>M은 별도 문제로 보류한다.** 현재 사후 점수 pruning에는 정확한 편향 반례가
있다. 새 pruning/proposal correction을 유도하고 검증하기 전까지 G=M으로 비교한다.

## 3. Dense 모델에서 proxy 및 calibration 다시 측정

기존 AWQ 수치를 새 모델의 정답으로 사용하지 않는다. 다만 frozen AWQ 보정표를
그대로 쓰는 transfer arm과 새 dense calibration arm을 구분하면 이식성을 분석할
수 있다. 주결과의 arm을 validation 결과를 본 뒤 유리하게 선택하지 않는다.

### Root score의 질문

Disjoint calibration/validation prompt ID와 내용 hash를 고정한다. T={1,.7,.5},
주 exit56과 앞/뒤 layer(예:40/72), 실제 같은 context의 p/q/e를 수집한다.
Layer별 entropy delta, TV(p,e), true residual mass, proxy residual mass, support
누락, rank deletion, 실제 top-k margin을 확인한다. p와 q는 실제 sampler의
temperature와 logits processing을 반영해야 한다. 기존 T=1 proxy 후보 관례를
유지하는 control과 temperature를 맞춘 변경 arm을 구분한다.

최소 ablation:

| Arm | Token score | 위치 배분/정규화 | 답하려는 질문 |
|---|---|---|---|
| A | legacy residual | legacy | 기존 DUET 기준 |
| B | e | legacy | 단순 proxy 기준 |
| C | e | full norm + hazard mix | 배분 개선의 효과 |
| D | e(1-q) 또는 고정 β 변형 | C와 같음 | Token score만의 추가 효과 |

사용할 β/λ/ω/floor는 calibration에서 고르고 고정한다. Validation에서 온도별로
best만 뽑아 평균을 내지 않는다. Actual wire top17→15, BF16 동점, P1 dedup,
root ready time, 최대 root budget까지 재현해야 실제 cache hit와 연결된다.
Offline coverage와 online cache hit를 둘 다 기록한다. `--duet_proxy_source`는
chain 전용이므로 이 switch를 tree에 붙여 D가 실행됐다고 주장하지 않는다.
Tree root 정책 port는 별도 구현/검증 후 별도 factorial arm으로 평가한다.

### 시간/파라미터 calibration의 질문

1. 새 hardware에서 target prefill/verify, exit→proxy send, draft forward,
   proxy ready 전후 구간, roots/nodes/row 수에 따른 latency를 적은 anchor로 측정한다.
2. K1은 단순 `exit_time / constant_latency`로만 결정되지 않을 수 있다. Comm,
   overlap, early arrival, dynamic K, padding/capture shape를 실제 timeline에 넣는다.
3. Chain의 모델을 tree에 그대로 적용하지 않는다. Tree의 generation/verification
   node 수, per-round width, bucket별 비용을 따로 구한다.
4. 예산/시간 제약으로 shortlist를 만든 후 소수 실제 비교로 reward(roots면 hit,
   tree면 AL)를 확인한다. 첫 추천→shortlist refinement→독립 confirmation을 구분한다.
5. Predict/measure 오차, 후보 수 감소, profiling 비용, 측정한 후보 중 best 대비
   regret를 기록한다. 전수 grid를 하지 않았다면 “global optimum”이라고 쓰지 않는다.

Tree 주비교에는 같은 budgets/root 정책을 유지한다. Calibration으로 더 큰 tree를
허용한 결과는 별도 시스템 설정 비교이며 score 자체의 효과와 분리한다.

## 4. 가장 먼저 완료할 본 실험: 여섯 tree 정책 AL

이 실험이 기존 작업의 직접적인 미완료 부분이다. 전체 정책은
`q_path`, `reach`, `q_gain`, `reach_gain`, `reach_frontier`, `reach_gain_frontier`.
구성과 수식은 [실험 이력 §9](EXPERIMENTS.md)에 있다.

### Freeze할 설정

- Target/draft/checkpoint/tokenizer/precision/TP/backend 고정.
- Exit56, K1/K2=4/2, C=3, P1 G=M=8, P2 G=M=6, P2 roots15,
  P1 위치당 roots3, 동일 root 정책/threshold/reserve/physical round widths.
- Reach/gain calibration은 test prompts와 분리해서 고정; table과 질문 hash 저장.
  추가 neural head 없음. 새 dense calibration이 없으면 transfer arm으로 명시.
- 기존 `questions.json`:480문항, 6tasks×80, 560turns,
  SHA256 `aec560cbe7bee67ca764c211c822cad8ff26e1a84834b86b86f35870c0048af4`.
- Seeds1,42,123, 정책 순서 interleave, 총18 runs/10,080 turns.
- 1차 재현은 natural EOS, cap128, native2048, raw prompt+실제 이전 응답을 유지.
  모든 flag/model/plan 변경을 기록하고 source hash를 freeze.
- 성능 캠페인은 full-distribution observer/profiler/trace off. 별도 diagnostic runs 사용.

### 반드시 비교할 contrast

1. reach − q_path: reach score만 바꾼 효과.
2. q_gain − q_path: q score 아래 gain 배분 효과.
3. reach_gain − reach: reach score 아래 gain 배분 효과.
4. reach_frontier − reach: 이전 depth의 미확장 frontier 허용 효과.
5. reach_gain_frontier − reach: 배분+frontier 결합 효과.
6. reach_gain_frontier − reach_gain: gain 정책에 frontier를 더한 효과.

주지표는 **각 task의 nonfinal hit-conditional descendant AL을 seeds에 걸쳐 pool한
뒤 여섯 task를 동일 가중 평균**한 값이다. Root/correction 자체는 AL에서 제외한다.
원래 question 단위 paired, task-stratified bootstrap 5,000회; 동일 question의
두 turn 및 모든 seed를 함께 resample한다. 위 사전 contrast의 simultaneous CI를
계산한다. Pointwise CI만 골라 winner를 선언하지 않는다. CI가 0을 포함하면
미확정으로 보고하고, 효과 크기와 task/seed 차이를 제시한다.

함께 남길 것: P1/P2 AL, equal-phase 진단, 깊이별 acceptance tail/histogram,
all-event AL, emitted AL, 실제 hit rate, no-hit question 수, generated/verified
node 수, executed/padded forward rows, per-root budget 사용, wall/TPS, peak VRAM.
Root 식이 같아도 trajectory와 hit subset이 달라질 수 있으므로 conditional AL만
보고 coverage 변화를 숨기지 않는다. Zero-hit question의 conditional AL을 0으로
임의 보간하지 않는다. Input 누락/절단, early failure, EOS/cap을 audit한다.

완료 조건: 18개의 valid run과 corpus/seed/policy coverage audit, 새
`online_analysis.json`, `audit.json`, per-task/seed 표, frozen manifest. 효과가 좋아
보인다는 이유로 조기 중단하거나 test corpus에서 다시 tune하지 않는다.

## 5. 원인을 판별할 추가 계측과 causal replay

별도 두 정책 full frontier trace(원래 reach와 reach_gain_frontier)를 수행한다.
각 선택 시점에 eligible node 전체, 불선택 이유(threshold/depth/expanded/quota/
physical width/reserve), score, γ, 할당 fanout, prefix hash, parent-q reference를
기록한다. Timing을 본 실험 성능과 섞지 않는다.

기존 trace는 hit 뒤 관측한 일부 tree이므로 다음을 단독으로 답하지 못한다.
“다른 가지를 택했다면 다음 p/q와 다중 round AL이 어떻게 됐는가?”
이를 위해 same-engine/KV 상태에서 branched replay 또는 별도 super-tree를 구축한다.
현재 full frontier trace를 실행하는 것만으로 이 기능이 구현되지는 않는다.

Replay 구현 시 gate:

- 알려진 parent부터 같은 prefix/token position/KV state/temperature/backend의 q를
  재생성한다. 기존 기준 median TV<.005, p95<.02를 baseline 재현 gate로 유지한다.
  Same-engine 경로의 더 엄격한 기준이 필요하면 효과 분석 전에 정한다.
- 실패하면 prefix serialization, actual generated history, special tokens, RoPE,
  padding/mask, KV truncation, fused kernel/precision/wire rounding을 하나씩 확인한다.
  효과가 좋아질 때까지 threshold를 느슨하게 바꾸지 않는다.
- 성공한 context만을 선택해 전체 개선으로 일반화하지 않는다. 누락/실패율을 제시한다.
- 미확장 child의 p/q를 관측하고, policy는 각 시점에 당시 볼 수 있었던 정보만
  사용한다. Oracle의 미래 p를 정책 입력으로 누설하지 않는다.
- Coin을 적분한 expected AL, 분포 calibration 오차, 표현력 부족, quota/threshold/
  frontier/배분 제한, 순수 coin noise를 분리한다.
- One-round γ 배분이 여러 round의 continuation value를 놓치는 정도를 평가한다.
  이 결과가 확보되기 전 “tree 구성이 최적”이라고 결론내리지 않는다.

## 6. 논문용 범위 확장

Full480/cap128 재현이 완료되면 별도 campaign으로 논문 cap1024와 chat format,
추가 모델/온도/예산을 평가한다. TinyLlama native2048에서 긴 prompt+출력1024가
넘칠 수 있다. Longer-context compatible draft, 명시적 RoPE 변경, 문맥 정책 중
어느 것을 사용할지 사전에 고정한다. RoPE extension은 모델 동작 변경이므로
기존 native run과 같은 조건이라고 하지 않는다. 묵시적 입력 절단은 하지 않는다.

Mirror-SD 전체 비교는 아직 없다. 실제 baseline 구현과 checkpoint/exit head,
temperature, precision, GPU 배치, cache/root budget, verifier, 출력 조건을 맞춰
별도 실행해야 한다. 지금의 proxy score e baseline은 그 실험을 대신하지 않는다.

## 최종 제출 묶음 및 판단

| 질문 | 필요한 결과 | 현재 상태 |
|---|---|---|
| Dense에서 정확한 실행인가? | Actual-model parity + q-reference/verification audit | 미실행 |
| 개선 tree가 AL을 올리는가? | 여섯 정책 full480/3seed + simultaneous CI | 미실행 |
| 왜 좋아지거나 실패하는가? | Frontier trace, 손실 분해, same-KV causal replay | Offline 일부 완료, 새 trace/replay 남음 |
| 새 score가 e보다 좋은가? | 같은 배분의 A/B/C/D + actual-wire/live hit | AWQ에서 score 추가 우위 미확정 |
| Calibration으로 sweep을 줄이는가? | 새 장비 shortlist/독립 confirmation 및 regret | AWQ chain의 제한적 feasibility만 완료 |
| 논문 조건에서도 유지되는가? | Cap1024/context 해결 + 실제 Mirror-SD baseline | 미실행 |

개선이 불확실하거나 없어도 모든 정책/seed를 보고한다. 성공의 기준은 원하는
결론을 만드는 것이 아니라, 고정한 조건에서 채택 여부를 판단할 근거를 완성하는 것이다.
