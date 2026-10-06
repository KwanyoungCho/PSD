# Full-model MLSys systems 실험 인수인계

## 먼저 읽을 파일

1. [REPORT.md](REPORT.md): 결론, 실제 성능, B>1 병목, 정확성 한계.
2. [FINAL_NUMBERS.json](FINAL_NUMBERS.json) / [GREEDY_NUMBERS.json](GREEDY_NUMBERS.json): 최종 표를 재생성한 값.
3. [README.md](README.md): 모델 revision·환경·측정 규칙.
4. `*_plan.json`: 당시 실행한 모든 CLI 인자와 환경 변수. 각 결과 폴더의 `campaign.json`: 실제 명령·GPU UUID·종료 상태·외부 GPU PID.

## Branch와 원래 연구의 관계

- 정확한 출발점: `origin/feat/duet-p2tree-g0@a82f7d2`.
- 작업 branch: `feat/duet-mlsys-coverage`, `/home/chokwans99/PSD-mlsys-coverage`.
- 원래 `/home/chokwans99/PSD`의 `feat/duet-proxy-source-ablation`과 미추적 root 연구 보고서는 건드리지 않았다.
- Root 후보 선정 수식, 위치 분배 수식, tree score의 새 연구 변경을 합치지 않았다.
- 이번 B>1/T=0 최적화 대상은 paper guide의 throughput 기준 chain 설정이다. Dynamic P2 tree는 B=1/T>0만 실행 검증했다.
- 실험 결과를 핑계로 원래 논문의 throughput 우위를 주장하지 않는다. 새 GPU 구성, 자체 harness, 짧은 출력 조건의 측정이다.

## 코드 작업 기록

| Commit | 변경 |
|---|---|
| `503fa96` | Ragged greedy 복구 위치, tokenizer/model pair 계약, token wire 32-bit, 초기 검증 harness |
| `317741d` | Paged prefix prefill, batch admission 제한, verification 동기화 감소 |
| `78d28d8` | Fully cached prompt의 query 보존, padding 계측 |
| `f50e74a` | Batched proxy CUDA Graph, B>1 exit-replica 경로 |
| `aabc246` | Stochastic fast verifier, GPU 격리·telemetry campaign, 파라미터 탐색 |
| `13f42d8` | Colocated target/draft의 RMSNorm 상수 capture, batch별 수치 차이 진단 |
| `7260f28` | 실제 출력 TPS 계수, preemption 시 출력 budget 보존 |
| `7d3e52b` | T=0 전용 sampler 및 temperature readback 감소 |
| `e2dd89f` | 생성 중 block hash가 해당 physical page를 가리키도록 수정; preemption prefix 저장/HF 검증; campaign resume 시 manifest 보존 |

주요 위치: `ssd/ssd/engine/{scheduler,verifier,sequence,step}.py`, `ssd/ssd/utils/{verify,verify_fast}.py`, `ssd/ssd/engine/helpers/batched_proxy.py`, `ssd/ssd/layers/{attention,sampler,layernorm}.py`.

## 실험 목록과 사용 목적

| 결과 폴더 | 목적 / 해석 |
|---|---|
| 루트 `llama*_ar/duet/tree.json` | 초기 dense-model B/T correctness, dynamic tree 실제 실행 |
| `screen/` | 48개 입력; eager/graph/replica, K1/K2, SSD/AR, profile. 최종 통계 아님 |
| `tune/`, `refine/` | Fast verifier, draft 폭, exit=26, K 변경 탐색. 불리한 결과도 보존 |
| `baselines/`, `baselines_remaining/` | SSD K=2/3/4 및 colocated synchronous SD. 실패한 첫 SD 로그도 포함 |
| `full_llama2/` | 480개 × 3 seed; 공통 오류 수정 baseline / 코드 개선 / fan-out=4 |
| `full_llama2_references/` | AR 1회, SSD K=3/F=2 3 seed |
| `full_llama3/` | 480개 × 3 seed; baseline / fan-out=1 후보. 후자는 full에서 채택 실패 |
| `full_llama3_fast/` | 동일 파라미터 코드 개선 3 seed, AR 1회, SSD 3 seed |
| `stress/` | B=3/8/16, T=0/0.7, 서로 다른 output cap, shrink/refill, HF 수치 진단 |
| `validation/` | 두 pair AR/SSD/DUET의 B=1/2/4/8 scaling; 1536-token cap 긴 입력; fast profile; SD K=4 |
| `greedy_tune/` | T=0 sampler 특화 및 K1=4/6/8 탐색 |
| `full_greedy/` | 전체 480개, 일반 sampler/특화 sampler. 같은 설정 출력 960/960 동일 |
| `greedy_repeat/` | Llama2 K1=4/6 별도 반복, K 변화에 따른 26개 first-divergence HF 진단 |
| `preemption/`, `preemption_progress/` | 저메모리 preemption, 짧은 출력 budget 점검 |
| `closing/` | 전체 greedy K1=6, 긴 출력 320-token preemption + HF audit |
| `profiles/` | 구간별 CPU dispatch/CUDA event 기록. 중첩 가능; 합으로 wall time 추정 금지 |

중간 실패/오염 파일은 README의 제외 규칙을 따른다. 최종 숫자는 명시된 완료·비오염 실행만 이용한다. `summary.decode_tps`가 옛 counter를 사용한 파일도 있으므로 최종 표에는 재계산 값을 사용한다.

## 새 서버에서 실행

- 실제 GPU에 맞게 `SSD_CUDA_ARCH` 변경. 아래 8.9는 RTX 4090이다.
- 모델 경로, Python 환경, prompt 경로를 plan JSON에서 변경한다. 기존 plan에는 당시 절대 경로가 보존되어 있다.
- Target/draft vocab 매핑 일치 필요. 두 모델 조합 모두 dense 가중치로 target 1 GPU + draft 1 GPU에 올렸다.
- GPU와 통신 port를 실험별로 분리한다. Campaign은 이미 사용 중인 GPU에서는 시작하지 않는다. 실행 도중 새 외부 PID가 들어온 결과도 성능 표에서 제외한다.
- `uv sync`로 공유 환경을 무심코 변경하지 말고, 새 서버에서는 별도 venv를 준비한다.

```bash
cd /home/chokwans99/PSD-mlsys-coverage
PY=/home/chokwans99/PSD/ssd/.venv/bin/python
$PY ssd/bench/mlsys_campaign.py \
  --plan results/mlsys_coverage/full_llama2_plan.json \
  --directory results/mlsys_coverage/server2_llama2 \
  --gpus 0,1 --port 29500

CUDA_VISIBLE_DEVICES=0 MLSYS_PYTHON="$PY" \
  bash results/mlsys_coverage/run_regressions.sh

$PY results/mlsys_coverage/analyze_results.py
$PY results/mlsys_coverage/make_figs.py
```

`analyze_results.py`는 **보존된 이번 최종 폴더 목록**을 읽는다. 새 서버 결과를 비교하려면 폴더 목록을 명시적으로 갱신하거나 `ssd/bench/mlsys_summarize.py --runs <run JSON들> --output <summary.json>`를 사용한다. 기존 결과를 덮어쓰지 않는다.

- Stochastic: `SSD_FAST_VERIFY=1`, `SSD_BATCHED_PROXY_GRAPH=1`이 기본. 두 값을 0으로 두면 correctness 수정은 유지한 baseline.
- Greedy: `--temperatures 0 --greedy-only`. Graph capture 전에 특화되며 T>0 요청은 거부한다.
- TP=1 권장 시작: `SSD_DUET_EXIT_REPLICA=0`, `SSD_ASYNC_PROXY_SEND=1`, `SSD_PROXY_STREAM=0`.
- `--mode duet-chain`이 B>1/T=0 경로. `duet-tree`는 B=1/T>0만 허용한다.
- 최종 관련 검사 184개 통과. 실험 종료 후 GPU worker는 모두 종료했다.

## 다음 개선에서 검증해야 할 것

### 우선순위 1: 배치 동기화 비용

실측 B=8에서 query padding 21~24%, 한 요청 이상 miss인 step 76~84%. Proxy 계산 자체보다 target graph, draft 응답 대기가 크다.

- 후보 A: hit 요청을 먼저 검증하고 miss 요청은 JIT 완료 후 합류시키는 scheduler. 장점은 기다림 감소지만 작은 GPU batch와 추가 graph launch 비용이 생긴다.
- 후보 B: 실제 k별 target verification bucket 또는 packed ragged verification. Query 낭비를 줄이지만 graph 수·KV 인덱싱·kernel 효율 비용을 함께 측정해야 한다.
- 후보 C: draft 응답을 요청/부분 batch 단위로 전달. 기존 wire, cache 수명, 순서, RNG 의미를 함께 보존해야 한다.
- 위는 **실측에서 도출한 후속 가설이며 구현·효과가 검증된 결과가 아니다**. AL/hit 증가만으로 채택하지 말고 TPS와 대기 시간을 비교한다.

### 우선순위 2: 논문 실험 프로토콜과 자원 공정성

- 실제 paper runner의 chat template, 입력/출력 길이, EOS, prompt corpus로 재현.
- 이번 출력 64-token 결과와 256/512-token decode 결과를 구분. 긴 요청에서 KV pressure·preemption과 이득이 달라질 수 있다.
- 전체 corpus를 이용한 B별 반복. 이번 전체 480문항 반복의 중심은 B=8이며, 다른 B scaling은 48문항 보조 측정이다.
- 다른 GPU/TP=2,4, NVLink 여부. TP=1 결과로 exit replica의 원래 TP overlap 가치를 판단하지 않는다.
- AR 1 GPU vs speculative 2 GPU 수치 외에 같은 총 GPU 예산의 처리량, target-only 복제 serving, 메모리 점유를 비교.
- Baseline SSD도 같은 탐색 예산으로 튜닝. Full 측정에서 DUET의 SSD 대비 확실한 우위는 아직 없다.

### 우선순위 3: lossless/greedy와 tree 범위

- 같은 verifier·같은 설정의 최적화 출력 일치는 확인했다. AR와 모든 batch/K에서 bitwise 같은 출력은 보장되지 않는다.
- Near tie의 원인을 고정하려면 attention/GEMM reduction, dtype, TF32/precision 정책을 구분하고 그 비용도 측정한다. 단순히 허용 오차를 두고 strict greedy라고 부르지 않는다.
- Dynamic tree B>1/T=0은 별도 미지원 범위. Chain 검증 결과로 이를 완료 처리하지 않는다.
- Root 후보/위치 수식 연구 branch를 합칠 때 correctness와 cache hit를 먼저 비교한 뒤, 이 branch의 chain TPS가 유지되는지 확인한다.

## 실험에서 채택하지 않은 주장

- Llama3 fan-out=1: 작은 표본의 이득이 전체 입력에서 재현되지 않음.
- T=0 sampler 특화: 연산은 줄었지만 TPS 차이 0.5% 미만, 유의미한 throughput 개선 주장 금지.
- Exit replica: TP=1에서 일관된 개선 없음.
- Equal K: padding 제거만으로 throughput이 좋아지지 않음.
- Greedy K 변화: 두 paired run 모두 ~4% 개선이지만 출력 26/480 차이; near-tie 진단과 함께 보고.
