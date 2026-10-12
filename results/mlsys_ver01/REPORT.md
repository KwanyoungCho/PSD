# MLSys-ver0.1 논문 브랜치 verification 변경 리뷰

2026-10-13. `feat/duet-p2tree-g0@a82f7d24fb36827a9a81a3567f344dccb71f193e`에서 `MLSys-ver0.1`을 만들었다. 기준 revision은 작업 시작 시 원격 논문 브랜치와도 일치했다. 구현 commit은 `be61186`이다. 작업 폴더는 `/home/chokwans99/PSD-MLSys-ver0.1`이다.

**구현과 계획한 검증을 완료했다.** 실제 논문 원본과 동일한 480개 입력을 비교했을 때, tree TPS는 두 반복에서 각각 **+1.78%, +2.61%**, chain은 **+2.85%** 개선됐다. 각 대조의 출력·AL·cache hit는 같았다. Target forward 시간은 거의 유지됐고, 수락 처리 비용이 감소했다. T0의 AR 출력 차이26건도 실제 logits로 모두 추적했다. 아래 결과는 7B·FP16·RTX4090·B1·G=M 조건이며, 모든 장비와 설정에 대한 무결성 보증은 아니다.

먼저 [전체 성능·구간 그림 PDF](figs/REVIEW.pdf), 아래 성능표와 구간표를 확인한 뒤 파일별 변경을 검토하면 된다. 수치 원본은 [PERFORMANCE.csv](PERFORMANCE.csv), 전체 파라미터는 [PARAMETERS.json](PARAMETERS.json)이다.

## 이번 브랜치의 범위

기존 논문 실행기를 유지하면서 chain 간소화, tree CPU 정보 재사용, 온도별 계산 간소화, T=0 tree 지원을 적용했다. Tree의 수락 경로는 **CPU에서 선택하는 한 가지 구현**이다. Target·draft forward와 vocabulary 분포 연산은 GPU에서 실행한다. CPU/GPU 수락 경로 선택 옵션은 없다.

이전에 준비한 verification A+B+C 패치에 해당한다. 새로운 root·위치·tree 선정 정책, GPU tree 수락 graph, B>1 통합 tree 실행기, KV 관리 변경은 포함하지 않는다. G>M proposal 수정, 실제 온도를 이용한 root 후보 점수, 집계·scheduler·wire의 production 이식도 이번 verification 변경과는 별도다. 이번 tree 실험은 G=M으로 수행한다.

## 원본과 달라진 동작

| 변경 | 논문 원본의 처리 | MLSys-ver0.1의 처리 | 검토할 파일 |
|---|---|---|---|
| Chain 확률 검증 간소화 | 양의 온도 요청에도 greedy argmax 준비, GPU 조건 확인, 작은 결과의 여러 번 readback | CPU의 요청 온도로 확률 검증 경로를 결정. JIT proposal의 p/q 검증·residual recovery와 난수 순서는 유지하고 필요한 최종 결과를 묶어서 반환 | `ssd/ssd/utils/verify.py`, `verify_fast.py`, `engine/verifier.py` |
| Tree 정보 재사용 | Target 입력 준비 때 확인한 topology를 수락 때 다시 GPU에서 읽고 파싱. 온도도 GPU scalar에서 읽음 | 준비 시 확인한 CPU topology와 float32로 반올림한 요청 온도를 재사용 | `ssd/ssd/engine/verifier.py` |
| 불필요한 온도 tensor 생략 | CPU tree walk와 순수 greedy chain에도 수락용 GPU 온도 tensor 생성 | 해당 경로에서는 생성하지 않음. E0 진단처럼 실제 tensor를 사용하는 경우는 유지 | `ssd/ssd/engine/verifier.py` |
| Sampler 온도별 계산 | T>0에도 사용하지 않는 logits argmax, T=0에도 확률·난수 준비 | 양의 온도는 확률 샘플링만 수행. T=0은 argmax만 수행 | `ssd/ssd/layers/sampler.py`, `config.py`, `engine/model_runner.py`, `engine/llm_engine.py` |
| T=0 tree 생성과 검증 | Legacy tree가 T=0을 차단하거나 chain으로 처리 | Draft 첫 자식은 argmax, 추가 형제는 중복되지 않는 상위 후보. Target argmax와 일치하는 자식으로 이동하고 없으면 그 argmax를 반환 | `ssd/ssd/engine/helpers/p2_tree.py`, `p2_tree_executor.py`, `engine/draft_runner.py`, `verifier.py` |
| 기존 benchmark 자동 연결 | Sampler의 온도 전용 경로를 선택하지 않음 | 기존 `--temp`, `--dtemp`로 capture 전에 자동 결정. 새 CLI 선택지는 없음 | `ssd/bench/bench.py` |

`greedy_only`와 `stochastic_only`는 CUDA graph를 만들기 전에 알려야 하는 내부 온도 계약이다. 기존 benchmark가 자동 설정한다. 일반 Python API에서는 초기화 시 계약을 지정할 수 있고, 계약에 맞지 않는 요청은 거부한다. 아무 계약도 지정하지 않는 API의 기존 호환 경로는 유지한다. 확률 샘플링에 필요한 random-score argmax는 제거 대상이 아니다.

### 확률적 tree에서 유지한 법칙

형제 후보는 순서가 있는 비복원 추출이다. 이전 형제를 제외한 proposal과 이전 거절을 반영한 target residual로 각 자식의 수락 확률을 계산한다. 이번 CPU 경로는 기존 `tree_verify_walk_tensor`를 그대로 사용한다. Tree의 방문 순서와 확률적 의미를 바꾸는 GPU 구현을 이식하지 않았다.

T=0에서는 p/q 수락 검증을 사용하지 않는다. 예산 배분·후보 순위에 사용하는 soft score와 target argmax 검증은 역할이 다르다. T=0의 soft score는 untempered draft 분포를 사용하며, 이를 확률 proposal이라고 간주하지 않는다.

## 실제 모델 대조 조건

| 항목 | 설정 |
|---|---|
| 데이터 | MT-Bench, translation, summarization, QA, math reasoning, RAG에서 저장한 각80개, 총480개 첫 turn 입력 전부. 원 데이터셋 전체나 새로운 holdout은 아님 |
| 모델 | LayerSkip-Llama2-7B + AMD-Llama-135M, full dense FP16 |
| 장비 | RTX4090 8개. 각 speculative job은 target TP1과 draft 1GPU 사용 |
| 주 비교 | B=1, T=1. Tree 2회 독립 process 대조, chain 1회 대조 |
| Greedy | B=1, T=0 tree 전체 480개와 원본 AR 대조. Chain T0 24개 추가 확인 |
| Batch 회귀 | B=4 chain 원본/변경본 24개 확인. B>1 tree 지원 실험은 아님 |
| DUET | Exit=21, K1=4, K2=2, speculate K=6, async fan-out=3, draft fan-out=2 |
| Tree | P1 G=M=8, P2 G=M=4, 형제 capacity=3, 위치별 P1 root=2 |
| 선정 정책 | 원본 residual·위치·tree 정책 유지 |
| 길이 | 입력 최대512, 출력 최대96, EOS 존중, model length=2048 |
| 메모리 | GPU memory utilization=0.45, KV block=256 |
| 전체 입력 워밍업 | 32개 입력 후 seed를 다시 설정하고 480개 측정 |
| 구간 진단 | 별도 24개 균등 선택 입력, 워밍업16개, 상세 프로파일 활성화 |

| 대조 | GPU | CPU affinity | 순서 | Target seed / draft seed |
|---|---|---|---|---|
| Tree 반복0 | 2,3 (PIX) | 8–15,40–47 | 원본 → 변경본 | 10801 / 231 |
| Tree 반복1 | 6,7 (PIX) | 24–31,56–63 | 변경본 → 원본 | 10802 / 232 |
| Chain | 4,5 (PIX) | 16–23,48–55 | 원본 → 변경본 | 10801 / 231 |
| Greedy | 0,1 (NODE) | 0–7,32–39 | Tree → AR | 10801 / 231 |

각 쌍은 같은 GPU·CPU affinity를 사용하며, 네 대조의 CPU 코어는 겹치지 않는다. 실행 전 GPU 점유를 확인하고 실행 중 외부 GPU process와 clock·온도·전력을 기록한다. Tree의 두 반복은 순서를 반대로 하여 순서에 따른 편향을 확인한다.

### 공통 benchmark 보정과 집계

실제 원본 checkout과 새 브랜치 코드를 실행한다. 새 통합 실행기의 graph OFF를 원본으로 대신하지 않는다. 두 방식에 동일한 AMD dtype 호환, 고유 shared-memory/port, 안전한 종료 adapter를 적용한다. Prefix lookup은 동일하게 끄고 decode의 기존 KV/glue 방식을 유지한다. 이 adapter를 verification의 성능 기여로 계산하지 않는다.

실제 반환 토큰을 측정한다. 출력 상한에 도달하거나 EOS 등으로 결과가 잘린 step, 활성 요청이 지정 B보다 적은 step은 토큰·시간 양쪽에서 제외한다. AL·hit도 같은 step 집합을 사용한다. AL은 recovery를 포함한 step당 반환 토큰 수다. TPS는 포함된 decode 토큰 합 / decode 시간 합이며 prefill을 포함한 전체 wall TPS와 구분한다.

프로파일 실행의 성능을 주 TPS 결과로 사용하지 않는다. Target exit 이전/이후 forward, 수락 처리, draft glue, P1/P2 graph, P1/P2 전체 구간을 따로 비교한다. CUDA event 구간에는 CPU launch gap이 들어갈 수 있으며, 서로 겹치거나 포함 관계인 구간의 평균·중앙값을 합산하지 않는다.

## 전체 입력에서 확인한 성능

각 행은 480개 입력 전체의 비계측 실행이다. Tree 두 반복은 seed와 GPU 쌍이 다르므로 **같은 반복 안에서** 원본과 변경본을 비교한다.

Tree와 chain도 서로 다른 GPU 쌍을 사용했다. 이 표의 목적은 각 구조에서 원본 대비 변경 효과를 확인하는 것이며, 구조 간 최종 우열을 확정하는 대조는 아니다.

| 구조와 반복 | 버전 | B | AL | Cache hit | Verification ms | TPS |
|---|---|---:|---:|---:|---:|---:|
| Tree 반복0 | 논문 원본 | 1 | 1.7780 | 64.35% | 17.3007 | 90.6187 |
| Tree 반복0 | MLSys-ver0.1 | 1 | 1.7780 | 64.35% | 16.8634 | 92.2345 |
| Tree 반복1 | 논문 원본 | 1 | 1.7776 | 64.26% | 17.2209 | 91.3444 |
| Tree 반복1 | MLSys-ver0.1 | 1 | 1.7776 | 64.26% | 16.7591 | 93.7290 |
| Chain | 논문 원본 | 1 | 1.7057 | 66.38% | 17.5661 | 88.3442 |
| Chain | MLSys-ver0.1 | 1 | 1.7057 | 66.38% | 17.0087 | 90.8610 |

**Tree의 TPS는 +1.78%, +2.61%, chain은 +2.85% 개선됐다.** 각 대조에서 480개 출력, AL, cache hit, 포함된 step 수가 모두 같았다. 이번 개선은 후보 선정 품질이나 더 긴 수락 길이로 얻은 것이 아니라, 같은 결과를 반환하는 실행 비용이 줄어든 결과다.

Verification은 tree 반복0에서0.4372ms, 반복1에서0.4618ms, chain에서0.5574ms 감소했다. 이 수치는 전체 verification 구간의 감소량이며 target 모델 forward 자체의 감소량과 동일시하지 않는다. 아래 구간 진단으로 실제 변화를 구분한다.

| 대조 | TPS 변화 | 입력 단위 paired bootstrap 95% 구간 | Verification 변화 |
|---|---:|---:|---:|
| Tree 반복0 | +1.78% | +1.60% ~ +1.99% | −2.53% |
| Tree 반복1 | +2.61% | +2.44% ~ +2.76% | −2.68% |
| Chain | +2.85% | +2.68% ~ +3.02% | −3.17% |

각 구간은 같은 process 대조 안에서 입력을 쌍으로 재표집한4000회 bootstrap이다. Process를4000번 독립 실행한 신뢰구간이 아니다. Tree는 실제 두 process 대조에서 모두 개선됐고 chain은 한 process 대조이므로, chain의 실행 간 변동까지 같은 강도로 검증했다고 하지 않는다.

![전체 입력 원본 대조](figs/01_full_comparison.png)

## 어느 구간이 빨라졌는가

각 값은 별도24개 입력 프로파일의 유효 구간 중앙값(ms)이다. 앞19 step, cap/clip/tail, capture가 있는 step과 그 다음 step을 동일하게 제외했다. 단독 kernel 실행시간이나 비계측 TPS와 같지 않다.

| 구간 | Tree 반복0 원본 → 변경본 | Tree 반복1 원본 → 변경본 | 해석 |
|---|---:|---:|---|
| Target exit 이전 forward | 10.2339 → 10.2369 | 10.2420 → 10.2410 | 거의 동일 |
| Target exit 이후 forward | 4.6561 → 4.6582 | 4.6592 → 4.6592 | 거의 동일 |
| Tree 수락 경로 처리 | 0.9191 → 0.5802 | 0.8968 → 0.5724 | 약36~37% 감소 |
| Tree 실행 중 miss의 chain 수락 | 1.4600 → 0.7699 | 1.4138 → 0.7670 | Chain 간소화가 tree miss에도 적용 |
| Draft P1 graph replay | 3.8431 → 3.8574 | 3.8984 → 3.9107 | 감소하지 않음 |
| Draft P2 graph replay | 1.6916 → 1.6937 | 1.6937 → 1.6988 | 감소하지 않음 |
| Draft P1 전체 구간 | 5.2344 → 5.1914 | 5.1973 → 5.2422 | 반복별 방향이 다름 |
| Draft P2 전체 구간 | 4.1660 → 3.9922 | 3.9941 → 4.0039 | 일관된 개선으로 판단하지 않음 |

Chain 전용 대조에서도 target pre/post는10.3895→10.3864ms / 4.7268→4.7268ms로 거의 같고, 수락 처리만1.3658→0.7403ms로 감소했다. 고정 GPU tensor를 사용하는 별도 chain 함수 대조도 같은 방향이다. V=32000·B1·T1에서 **chain proposal 길이2**는1.0942→0.5173ms, **길이4**는1.0974→0.5117ms였다. 여기의 길이는 DUET phase 예산 K1/K2와 구분한다.

따라서 이번 묶음의 TPS 개선은 **동일한 모델 계산과 출력에서 수락 처리의 불필요한 작업·동기화를 줄인 효과**로 설명할 수 있다. Draft 모델 forward가 빨라졌다는 근거는 없다. 묶음 대조이므로 각 변경의 독립 TPS 기여를 더해서 주장하지 않는다.

![Tree 구간 대조](figs/02_components.png)

![Chain 구간 대조](figs/04_chain_components.png)

[실제로 관측한 두 GPU timeline](figs/03_timelines.png)도 제공한다. 각 process에서 관측한 대표적인 P2-hit step이다. 이 그림 하나를 고정 tensor replay의 인과 증명으로 사용하지 않는다.

## 추가 기능과 회귀 결과

| 조건 | 버전 | 입력 수 | AL | Cache hit | TPS | 판정 |
|---|---|---:|---:|---:|---:|---|
| B4 chain, T1 | 논문 원본 | 24 | 1.6603 | 66.67% | 294.7446 | 회귀 대조 |
| B4 chain, T1 | MLSys-ver0.1 | 24 | 1.6603 | 66.67% | 310.1797 | 출력24개 모두 같음, 관측 TPS +5.24% |
| B1 tree, T0 | MLSys-ver0.1 | 480 | 2.5447 | 86.40% | 136.9019 | Tree T0 실제 실행 완료 |
| B1 AR, T0 | 논문 원본 | 480 | — | — | 65.7473 | Greedy 정확성 대조 기준 |

B4는24개 입력의 회귀 검사이며 전체 데이터셋의 B>1 최적화 결과로 일반화하지 않는다. T0 AR은 원본 sampler를 사용하는 정확성 기준이다. 이 표를 별도로 최적화한 AR·SSD와의 공정한 최종 경쟁 실험으로 제시하지 않는다. Tree T0의 source별 포함 step은 miss2372, P1 hit7693, P2 hit7380으로 두 phase가 실제로 실행됐다.

### T0에서 AR과 달랐던26개 출력

480개 중454개는 AR과 같고26개는 달랐다. 26개 모두 첫 차이 직전의 **동일 token prefix**에서 실제 AR/tree logits를 저장해 다시 확인했다. 진단 실행은 원래 전체 실행의 토큰열도 모두 재현했다.

- 26개 모두 AR은 자기 target logits의 argmax, tree도 자기 target logits의 argmax를 반환했다.
- AR의 top1–top2 차이는 최대0.015625였고, 두 실행의 vocabulary 전체 최대 logit 차이는0.029296875였다.
- 24건은 tree-hit verification, 2건은 tree 실행 중 miss-chain에서 발생했다.
- 확인한 직접 원인은 FP16의 근접값·동점에서 query shape별 logits가 조금 달라지는 것이었다. 잘못된 자식을 수락하거나 argmax가 아닌 token을 반환한 사례는 없었다.

이로써 target argmax 규칙을 따르는 것은 확인했지만, AR과 비트 단위로 같은 토큰열을 보장한다고 표현하지 않는다. 원시 logits와 진단 결과는 `RAW_RESULTS.tar.gz` 및 [GREEDY_DIAGNOSIS.json](GREEDY_DIAGNOSIS.json)에 있다.

## 정확성 검사와 발견한 문제

- 원본 함수와 고정 입력 대조: chain 384건, tree 288건에서 반환 결과와 RNG 상태 일치. Tree의 terminal sampling 분포도 tensor 값이 일치했다.
- 온도 0.3/0.7/1과 FP16/BF16의 q 분포 12조건, sampler 12조건 대조 통과.
- T0 branching/tie/중복 형제, ragged chain의 실제 길이, 온도 계약, sampler RNG 검사 5개 통과.
- P1 기존 검사 28개 통과.
- P2 기존 검사108개를 GPU에서도 실행해 **skip 없이 모두 통과**했다. P1 검사28개도 GPU가 보이는 환경에서 통과했다.
- GPU chain 고정 tensor 대조: V=32000/128256, B=1/2/4/8, proposal 길이2/4, T=0.7/1의32조건×16 seed, **512개 출력 모두 원본과 일치**했다. 이는 함수 단위 검사이며 큰 vocabulary 모델의 통신·전체 serving 지원까지 검증한 것은 아니다.
- **발견한 실패는 기존 P2 warmup 테스트의 mock 누락이었다.** 가짜 executor에 `total_cells`가 없었고 page bucket의 기대 개수도 이전 값이었다. 수정 전 테스트를 원본 runtime에서도 실행해 동일 실패를 재현했다. Mock에 실제 footprint 30을 지정하고 기대 bucket을1–8로 바로잡았다. Production warmup 함수는 원본과 동일하다. 실패 기록도 보존했다.

고정 입력의 확률 검증 동등성과 end-to-end 토큰열 일치는 구분한다. 비동기 phase의 timing에 따라 hit 경로가 달라질 수 있다. T0도 FP16에서 query shape별 target logits의 미세 차이로 AR과 토큰열이 달라질 수 있으므로, 발생한 첫 차이는 동일 prefix의 실제 logits로 추적한다.

계획한 모델 실행25개(전체480개 실행8개 포함)와 T0 추가 진단2개, 총27개 실행이 정상 종료했다. 각 실행의 GPU 점유 기록에 외부 process 간섭은 없었다. 발견했던 테스트 mock 오류 외에 이번 변경으로 인한 실행 실패나 검증 불일치는 발견하지 못했다. 기존 prefix 재사용·G>M 등 범위 밖의 조건까지 오류가 없다고 주장하는 결과는 아니다.

## 리뷰와 재현

1. `git diff a82f7d2..MLSys-ver0.1 -- ssd/ssd`로 runtime 변경을 확인한다. `engine/verifier.py`와 `utils/verify_fast.py`, `helpers/p2_tree.py` 순서가 이해하기 쉽다.
2. `ssd/bench/bench.py`에서 기존 온도 인자로 sampler를 자동 연결한 부분을 확인한다.
3. `PLAN.json`, `*_plan.json`에서 기준 checkout, 모델·예산·GPU·seed를 확인한다.
4. `ANALYSIS.json`의 전체 입력 결과와 `COMPONENTS.json`의 구간별 결과를 함께 본다. 프로파일 수치를 비계측 TPS와 섞지 않는다.
5. `PAPER_PARITY.json`, 테스트 로그, `GREEDY_DIAGNOSIS.json`으로 정확성 근거를 확인한다.

환경은 기존 `/home/chokwans99/PSD/ssd/.venv/bin/python`을 사용한다. 다른 서버에서는 모델 경로와 baseline checkout 경로, GPU 번호와 CPU affinity를 해당 장비에 맞춘다. Runtime의 CPU/GPU 경로 선택 옵션을 추가할 필요는 없다.

원시 자료는 [RAW_RESULTS.tar.gz](RAW_RESULTS.tar.gz)에 보존한다. [RAW_MANIFEST.json](RAW_MANIFEST.json)에 파일별 SHA256과 전체 archive SHA256을 기록하며, archive를 다시 읽어 원본과 대조했다. 이 폴더에서 `tar -xzf RAW_RESULTS.tar.gz`로 풀고 `analyze.py`, `analyze_components.py`, `make_figures.py`를 실행하면 표와 그림을 재생성할 수 있다. [RUNTIME.patch](RUNTIME.patch)는 논문 원본 기준 runtime과 기존 benchmark 온도 연결 변경만 담고, 결과 archive나 진단 harness는 제외한다. 소스·모델 config·질문 집합 해시는 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)을 따른다.

실행 예시는 다음과 같다. 결과가 이미 존재하는 폴더에 덮어쓰지 않도록 새 결과 폴더와 plan 경로를 사용한다.

```bash
python results/mlsys_ver01/run_pair.py \
  --plan results/mlsys_ver01/tree_r0_plan.json \
  --gpus 2,3 --port 30020 \
  --directory results/mlsys_ver01/tree_r0
```

여기서 검증하는 것은 원 논문 **코드 기준의 7B·4090 결과**다. 70B·다른 GPU·TP2·큰 tree에서의 개선율은 그 서버에서 같은 기준으로 재측정해야 한다. Prefix 재사용, G>M 및 B>1 tree까지 이번 결과가 검증한 것으로 확대 해석하지 않는다.
