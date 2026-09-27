# 코드·구현 지도와 이식 주의점

기준: `a82f7d2` 이후 이 브랜치의 변경. 원본 engine 대부분은 이전 commit에서
이미 구현되어 있었다. 이번 인계에서는 기존 연구 코드를 보존하고 문서·검증·이관
도구를 추가한다. 새 정책을 production 기본값으로 채택한 것은 아니다.

## Production 쪽의 실제 변경

| 파일(저장소 루트 기준) | 이번 변경 | 확인해야 할 한계 |
|---|---|---|
| `ssd/bench/bench.py` | `--duet_proxy_source residual/proxy/draft`, `--duet_only_proxy` CLI 전달 | Only-proxy 기능 자체는 기존 config에 있었으며 이번에 CLI로 노출 |
| `ssd/ssd/config.py` | Source 기본 residual, 값 검사, tree와 nonresidual source 조합 거부 | Tree source는 sibling ladder라 chain score switch를 적용하지 않음 |
| `ssd/ssd/engine/helpers/p2_tree.py` | Chain 후보 source 선택과 captured chain proxy graph 전달 | `[e-q]+`, e, q; draft 실현 token 제외 후 top-M; 기존 정규화 관례 유지 |
| `ssd/ssd/engine/verifier.py` | Eager chain source 분기, `_run_exit_probe`와 entropy/distribution 진단 | B=1 chain 지원; tree/sampler_x 등 미지원 경로를 일반화하지 않음 |
| `ssd/ssd/engine/model_runner.py` | Capture 전 post-layer hidden/residual tap 및 probe layer 설정 | Rank0 full head replica 필요; 모델/shape에 따른 메모리 증가 |
| `ssd/ssd/models/llama3.py` | Preallocated tap으로 post-layer hidden/residual 복사 | Prefill의 큰 row 수는 건너뛰고 지원 row 범위만 저장 |
| `ssd/ssd/engine/helpers/cudagraph_helpers.py` | Tap이 실제 exit hidden과 일치하는지 최초 self-check | `MISMATCH`/`UNVERIFIED` 로그를 무시하면 layer 해석이 틀릴 수 있음 |
| `ssd/ssd/engine/llm_engine.py` | 종료 시 collector 최종 flush | 마지막 partial chunk 누락 방지 |
| `ssd/ssd/engine/helpers/{exit_probe,entropy_probe,distribution_probe}.py` | 신규 coverage/entropy/full-pqe 수집 | 진단 목적, 저장/계산 overhead 있음 |
| `ssd/tests/test_entropy_probe.py` | 알려진 분포, variable K, hazard, RNG 불변, flush 검증 | 실제 TP/CUDA correctness 전체를 대신하지 않음 |
| `ssd/env.sh` | 이전 eslab19 경로/4090 sm89 환경으로 설정 | 새 서버에서 그대로 source하면 사용자의 env를 덮어씀 |
| `ssd/docs/duet/TREE_IMPLEMENTATION.md` | Tree audit/AL 목적 및 G>M 편향 주의 추가 | 기존 문서의 TPS champion과 최신 AL 목적을 구분 |

Source switch는 chain CUDA Graph를 만들 때 결정된다. 다른 정책을 비교할 때는
새 프로세스에서 config/env를 고정하고 graph를 새로 capture한다. 실행 중에
환경변수만 바꾸어 정책이 바뀌었다고 가정하지 않는다.

Layer 번호는 **0-based post-layer index**다. Llama2-70B의 index56은 57번째
layer 이후이며 마지막은79다. Probe는 `hidden+residual → final norm → lm_head`
경로를 사용한다. `SSD_DUET_PROBE_LAYERS=all` 또는 index CSV가 사용되며,
rank0 진단에는 `SSD_DUET_EXIT_REPLICA=1`이 필요하다. Rank0에서 TP collective
head를 호출하는 변경은 다른 rank와 통신 순서가 맞지 않아 hang을 일으킬 수 있다.
일부 self-check 경로는 exit layer가 probe set에 있다고 가정하므로 새 subset을
쓸 때 exit index를 포함한다. 부분 subset의 모든 경로가 검증된 것은 아니다.

## 서로 다른 “only-proxy”와 확률의 의미

| 이름 | 정확한 의미 |
|---|---|
| `--duet_proxy_source proxy` | Chain P2 token score를 e로 설정 |
| `--duet_only_proxy` | P1 후보 생성을 끄고 logical K1=0, P2만 cache root 구성 |
| `SSD_TF_POLICY` | Training-free 연구 hook의 score/normalization/hazard 조합 |
| `DUET_TREE_SCORE_MODE` | Tree continuation의 q-path 또는 calibrated reach priority |
| `DUET_TREE_POLICY` | 후속 여섯 정책의 reach/fanout/frontier 조합 |

`p`는 해당 run의 target 최종 분포, `q`는 실제 draft proposal, `e`는 proxy다.
AWQ target을 실행한 run의 p는 dense 원본 p와 다를 수 있다. Root candidate score는
cache 준비를 결정하며 최종 token sampling 분포 자체가 아니다. Hit root 이후
자식 검증에서는 해당 parent의 q row와 WOR sibling 순서를 사용한다.
`q`의 original row, 앞 형제를 제외한 `D_j`, target residual `R_j`를 구분한다.

## 연구용 코드가 실행에 연결되는 위치

| 연구 디렉터리 | 주요 파일 | 읽기/실행 목적 |
|---|---|---|
| `results/residial_dist/entropy/` | `aggregate.py` | Probe chunk 집계, layer별 entropy |
| `results/residial_dist/shared_review/` | 해당 REPORT의 실행 스크립트 목록 | Full pqe와 temperature/support 검토 |
| `results/residial_dist/direct_comparison/` | `THEORY.md`, 분석 스크립트 | Residual/proxy 비교, 삭제/순위 손실 분해 |
| `results/residial_dist/training_free/` | `runtime_policy.py`, `PROTOCOL.md` | Chain 후보 hook, actual wire/top-k 검증, live 비교 |
| `results/duet_calibration/` | `HANDOVER.md`, 각 campaign 스크립트 | Anchor profiling, 비용/보상 model, recommendation, confirmation |
| `results/duet_tree_analysis/` | `core.py`, `audit_math.py`, `score_hook.py` | Exact ladder/subset 진단, bias 반례, frozen reach hook |
| `results/duet_tree_al_full/` | `run_full.py`, `runtime.py`, `analyze.py`, `validate.py` | 이전 두 정책 full480 생성/지표/검증 |
| `results/duet_tree_posthoc/` | `observe.py`, `features.py`, `analyze.py`, `fanout.py` | Hit tree/full-pq 수집, OOF 보정, gain 진단 |
| `results/duet_tree_followup/` | `gain.py`, `calibrate_gain.py`, `allocation.py` | C=3 gain 및 사전 정수 fanout 배분 |
| 같은 디렉터리 | `policy_hook.py`, `runtime.py`, `trace_hook.py` | 여섯 정책/계측, worker 시작 시 hook 설치 |
| 같은 디렉터리 | `check_math.py`, `check_frontier.py`, `check_cuda.py`, `check_executor.py` | 단계별 수학/selector/GPU 검사 |
| 같은 디렉터리 | `run_full.py`, `launch.py`, `freeze_submission.py`, `analyze_online.py`, `validate_online.py`, `analyze_trace.py` | 미완료 real-model campaign 실행 및 분석 |
| 같은 디렉터리 | `shadow_q.py`, `diagnose_c3.py`, `robustness.py` | 실패한 shadow q gate, C3 손실/민감도 진단 |

후속 `runtime.install()`은 이전 full-run metric hook, score hook, 새 policy hook,
선택적 trace를 연결한다. **Spawn된 worker에서도 engine 생성/capture 전에 설치**돼야
한다. Hook은 process-local monkeypatch이며 별도의 production CLI 정책 구현이 아니다.
Fresh process 없이 hook을 겹쳐 설치한 비교는 지원하지 않는다.

`score_hook.py`는 `DUET_TREE_SCORE_CALIBRATION`의 JSON을 읽는다. Frozen table은
`results/duet_tree_analysis/calibration_frozen.json`이고 SHA256은
`afcb6510bfa6e64c8c1bad11ec49cbefd9c76bee790dcad3e65a076ec2537d39`다.
후속 `policy_hook.py`는 자기 디렉터리의 `gain_calibration.json`을 사용한다.
이를 새 dense calibration으로 바꾸면 새 policy version 및 source/model/calibration
manifest를 생성해야 한다.

## 데이터 계약과 재분석

- Entropy: metadata JSON → chunk NPZ. `entropy_proxy`, `entropy_target`, hazard,
  sequence/context 구분을 유지한다. 모든 context를 독립 표본으로 취급하지 않는다.
- Full distribution: JSON metadata와 NPZ의 저장 layer/온도/position/step을 함께
  사용한다. Sampling interval/저장 precision과 실제 wire proposal precision을 기록한다.
- Full online: `runs/<seed_policy>/records.jsonl`, 한 원래 question/turn당 한 row.
  UID, seed, group, input/output hashes, output IDs, stop reason, wall time,
  `metrics.phase_events`가 포함된다. 마지막 event를 제외하는 AL과 emitted AL은 다르다.
- Posthoc: `trees.jsonl`의 tree metadata와 sampled NPZ의 p/q가 짝을 이룬다.
  Parent indices, sibling order, parent q references, ancestor closure를 보존한다.
  Hit-only trace에 없는 branch의 분포를 생성해 넣지 않는다.
- Followup: `c3_contexts.jsonl`, `c3_allocations.jsonl`은 동일 관측 context의
  국소 배분 진단이다. Online `records.jsonl`과 섞지 않는다.
- 기존 `final_manifest.json`은 역사적 파일의 hash다. `execution_manifest.json`은
  실행 source와 외부 model의 size/mtime까지 고정한 제출물이다. 새 서버에서
  경로가 바뀐다는 이유로 검사를 제거하지 말고 새 manifest를 만든다.

## 새 서버에서 그대로 실행할 수 없는 부분

| 위치 | 이식 작업 |
|---|---|
| `ssd/env.sh` | 새 local env 작성; 경로/arch/backend를 실제 장비에 맞춤 |
| Followup `launch.py` | 기존 AWQ `command.json` 복사, GPU5개, old cwd/Python 경로를 새 config로 변경 |
| Followup `run_full.py` | Tokenizer AWQ 절대경로를 actual model로; plan/question 위치와 T=.7 상수 일치 확인 |
| 같은 `run_full.py` | AL depth4/2 검사 및 context+128 guard는 실험 예산 변경 시 명시적으로 조정 |
| `check_executor.py` | Local cache/arch defaults를 명시 override; tiny fixture 결과를 full model과 구분 |
| `freeze_submission.py` | 새 source set, 모델 metadata/hash, plan, gain/reach files로 새 manifest |
| `calibrate_gain.py` | 원래8 AWQ prompts의 입력을 그대로 dense calibration으로 오인하지 않음 |
| `policy_hook.py`, `runtime.py` | HERE/import 경로를 fresh campaign에 맞춰 old/new 결과가 섞이지 않게 함 |
| `validate_online.py`, `analyze_online.py`, `analyze_trace.py` | 새 policy/plan/records 경로와 고정 contrast가 일치하는지 확인 |
| `make_report.py`들 | 일부 역사적 수치/경로가 상수임; 새 보고서를 old 숫자로 채우지 않음 |
| Old paper baseline driver | `<HOME>/baseline/runners/`, `DUET_PAPER_RESULTS/scripts` 등은 repo 밖; 필요하면 별도 전송 |

새 full480 research runner 자체는 이 저장소 안에 있다. 외부 paper driver가 없다는
이유로 이 runner까지 없다고 생각하지 않는다. 반대로 기존 raw continuation 평가를
그대로 paper chat/1024 평가라고 보고해서도 안 된다.

완료한 연구 코드를 보존하는 것이 이번 인계의 목적이므로 이식용 대규모 리팩터링은
수행하지 않았다. 실제 서버의 모델/GPU가 확정되면 위 상수를 새 campaign config로
옮기고 smoke를 통과시킨 뒤 본 실험을 실행한다.
