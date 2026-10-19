# ver1.4 Tree 예산 배분 실험 재현 코드

브랜치: `exp/duet-v14-tree-figures-repro`.
기준 엔진: `MLSys-ver1.4`의 **dedc85ccb299df2d76fcf34d50f05f2d8747a32a**.

이 브랜치는 위 엔진에서 분기해 **이 디렉터리만 추가**한다. 엔진 런타임은 수정하지 않는다.
비교 코드·입력 검증·집계·그림 생성 코드와 지침을 포함한다. 데이터 본문, 모델 가중치,
실험 결과·PDF·로그·원시 logits는 추가하지 않는다.

## 재현할 실험

실제 문장은 원래 ver1.4가 생성한다. 같은 문맥에서 실제 hit 후보의 후속 구조를
Chain / 균등 예산 Tree / DUET Tree로 각각 추가 생성하여 계산 배분과 기대 AL을 비교한다.
원래 ordered WOR와 recursive residual 검증을 유지한다. ver1.5 target matching,
calibration 수락확률 표, 생성 후 점수순 top-M pruning은 사용하지 않는다.
세 방식으로 각각 전체 후속 문장을 생성한 온라인 TPS 비교는 아니다.

| 구분 | 실제 문장 생성 | 추가 비교 계산(P1·P2 공통) |
|---|---|---|
| P1 K / W / root | 10 / 56 / 위치당 4개 | 10 / 10 / 총 10개 |
| P2 K / W / root | 5 / 28 / 총 24개 | 10 / 10 / 총 10개 |
| P1 G=M | 23 | 300, 제한 미도달 검사 |
| P2 G=M | 15 | 300, 제한 미도달 검사 |
| C | 3 | Tree 3 / Chain 1 |
| Exit | 21, 0-based | 실제 생성에서 얻은 같은 문맥 |

공통 조건은 LayerSkip-Llama2-7B + JackFram/llama-68m, B1, BF16, target TP1,
target/draft GPU 각각 1장, 양쪽 T=0.7, nucleus 및 sampler rescaling 없음이다.
`SSD_ASYNC_PROXY_SEND=1`, `SSD_DUET_EXIT_REPLICA=0`, `SSD_PROXY_STREAM=0`,
cold prefix, 모든 P1/P2 page bucket prewarm을 사용한다.
Target memory utilization은 0.45, draft free-memory fraction은 0.35다.
공통 capacity adapter는 KV 용량을 확인하고 예기치 않은 preemption이면 실패시킨다.

### 입력·관측·집계 규칙

- 원본 Spec-Bench 480문항·560턴, seed 42·123·2026 각각 전체 1회.
- raw prompt·순서·토큰 ID를 고정한다. 임의 chat template·truncation·생성 대화 이력을 추가하지 않는다.
  MT raw turn들을 각각 입력하되 집계 때 같은 문항으로 묶는다.
- 문맥 2048, 출력 상한 `min(1024, 2048-입력 길이)`, 자연 EOS 허용, 기존 context guard 유지.
- 처음 3턴(`81_t0`, `81_t1`, `82_t0`)은 560턴 안의 warmup으로 집계에서 제외한다.
- 비교 계산은 후보 **생성 step** 1·9·17…에서만 수행한다. hit/AL을 보고 관측 step을 고르지 않는다.
- P1 후보 10개는 native pool에서 위치 quota를 최대한 균등하게 배분한다. 위치 안에서는 높은 점수 우선.
  위치 순환 시작점은 private seed `seed*1000003+seq*503+generation_step*101`로 정한다.
- P2 후보 10개는 native DS 전체 pool과 중복 제거한 native PS pool의 점수 상위 10개다.
- 이 10개는 hit 전에 고정하며 같은 phase의 세 방식에서 공통 사용한다. 실제 hit를 뒤늦게 추가하지 않는다.
- Chain은 root당 매 round 입력 1개·자식 1개다. 균등 Tree는 root당 매 round 입력 1개이며
  그 root의 마지막 깊이에서 최고 q-path 노드를 확장해 자식 3개를 ordered WOR로 샘플링한다.
- DUET은 마지막 깊이의 노드들이 root score × 누적 draft 경로 확률로 입력 10자리를 경쟁한다.
  과거 깊이 전체 frontier 경쟁이 아니다. 자식 샘플링은 ordered WOR 3개다.
- 모두 **유효 입력 10개 × forward 10회 = 100개**다. Tree 후속 노드는 총 300개, Chain은 100개다.
  같은 draft 입력 수 비교이며 verification 노드 수·wall latency까지 같은 비교는 아니다.
- G=M=300은 비활성 안전 용량이다. 생성수 `C*K*W`와 padding 없는 100입력을 검사한다.
- 실제 hit root가 고정 10개 안에 있으면 세 구조의 union을 target에서 평가해 실제 p를 얻는다.
  각각의 실제 q로 기존 residual walk의 수락 난수를 적분한 기대 AL을 계산한다.
  `E[AL]=1+Σ(d=1..10)P(accepted draft length>=d)`이며 자연 EOS 이후는 더하지 않는다.
- 그림 1은 준비된 forest의 score rank별 입력 budget, 그림 2는 실제 hit root 대 나머지 9개 평균,
  그림 3은 동일 hit 문맥·root에서 세 구조의 조건부 기대 AL이다.
- 그림 2/3은 warmup·cap/clip·출력 잔여11토큰 미만·고정 pool 밖 hit·공통 target KV 공간 부족을 제외한다.
  이유와 분모를 저장한다. 포함률을 cache hit rate로 해석하지 않는다.
- 그림 1은 hit나 AL을 기준으로 사후 선별하지 않는다. 그림 1과 2/3의 모집단은 다르다.
- 이 품질 분석에는 latency outlier 필터를 추가하지 않는다. 추가 계산이 있으므로 TPS를 인용하지 않는다.
- step → turn 평균 → 문항 평균 → 동일 가중치 seed 평균. 95% CI는 문항 bootstrap 1000회이며
  같은 문항 재표집을 seed 사이 공유한다. 생성 seed 모집단 전체의 CI는 아니다.

## 다른 서버에서 받기

```bash
git fetch origin exp/duet-v14-tree-figures-repro
git switch --track origin/exp/duet-v14-tree-figures-repro
git worktree add --detach ../PSD-v14-tree-engine dedc85ccb299df2d76fcf34d50f05f2d8747a32a
```

이미 작업 중인 checkout을 바꾸기 어렵다면 도구 브랜치도 별도 worktree로 받는다.
실행 코드 경로와 검증할 엔진 경로는 구분한다. `--checkout`에는 정확한 기준 commit의
**clean 엔진 worktree**를 지정한다. 새 도구 브랜치 HEAD를 엔진 기준으로 지정하면 commit 검사가 실패한다.
결과를 엔진 worktree 안에 쓰지 않는다.

모델은 두 checkpoint의 로컬 snapshot을 준비한다. 원본 환경은 Python 3.12.7,
torch 2.8.0, Triton 3.4.0, Transformers 4.57.1, flashinfer-python 0.5.2,
NumPy 2.3.3, RTX 4090이었다. 엔진 `ssd/pyproject.toml`을 기준으로 준비하고 matplotlib도 설치한다.
이 코드는 엔진/가중치/CUDA 라이브러리를 설치하거나 GPU 작업을 자동 종료하지 않는다.

## 입력 데이터는 별도 지정

이 코드 전용 브랜치에는 prompt·토큰 ID 본문을 넣지 않았다. 이전 재현 묶음의
`payload/INPUTS_FULL.json` 또는 동일한 원본 파일을 별도로 준비한다.
원본 서버의 해당 파일은 다음 경로에 있다.

```text
/data/chokwans99/duet-v14-tree-figures-20261019/INPUTS_FULL.json
```

`init --inputs-full`로 지정한다. `INPUT_SPEC.json`의 semantic SHA256으로 순서·prompt·토큰 ID를
포함한 내용 일치를 검사한다. JSON 들여쓰기 차이는 허용하지만 임의 재토큰화·문항 변경은 허용하지 않는다.
Smoke 입력 9턴은 이 파일 앞 9턴에서 자동 생성하고 별도 hash로 확인한다.
입력 파일이 없으면 원본 실험 담당자에게 받아야 하며, Git fetch만으로 데이터까지 생기지는 않는다.

## 실행 명령

아래 경로와 GPU는 예시다. 해당 서버에서 비어 있는 GPU와 `nvidia-smi topo -m` 결과에 맞춰 지정한다.
각 쌍의 첫 GPU가 target, 둘째가 draft다. 1~3쌍을 지정할 수 있고 쌍은 겹치면 안 된다.
두 장이면 세 seed를 순차, 여섯 장이면 세 seed를 병렬 실행한다.
CPU affinity가 필요하면 `--cpu-sets '0-3' '4-7' '8-11'`처럼 쌍별로 하나씩 지정한다.

```bash
cd tools/tree_figures_v14
python -B reproduce.py init \
  --checkout /absolute/path/PSD-v14-tree-engine \
  --target /absolute/path/layerskip-llama2-7b \
  --draft /absolute/path/llama-68m \
  --inputs-full /absolute/path/INPUTS_FULL.json \
  --work /absolute/path/tree-figures-v14-run \
  --cuda-arch 8.9 \
  --pairs 0,1 2,3 4,5

python -B reproduce.py check --work /absolute/path/tree-figures-v14-run
python -B reproduce.py smoke --work /absolute/path/tree-figures-v14-run
```

`check`는 GPU forward 없이 입력·코드·CPU 기대 AL을 검사한다.
`smoke`는 첫 쌍에서 seed42·9턴·출력96의 원본/관측 실행을 각각 수행한다.
9/9 출력과 모든 source/AL event 일치, P1·P2×세 방식의 실제 p/q 기대값 검사를 요구한다.
Query 폭 변경에 따른 BF16 q 차이는 가능하므로 각 구조는 자신이 실제 사용한 q로 평가한다.
마스크·문맥과 같은 native 폭의 replay는 일치해야 한다. 관측 중 private RNG와 KV 복원도 검사한다.

Smoke 통과 후:

```bash
nohup python -B reproduce.py full \
  --work /absolute/path/tree-figures-v14-run \
  > /absolute/path/tree-figures-v14-run/campaign.log 2>&1 &
```

각 seed에서 **해당 서버의 원본 full → 관측 full**을 순차 실행한다. GPU 쌍끼리는 병렬이다.
다른 서버의 BF16 차이를 원본 Orion 출력과 직접 비교해 관측 버그로 오인하지 않기 위함이다.
총 3개의 원본과 3개의 관측 실행이며, seed를 무한히 추가하지 않는다.
과거 PROTOCOL의 연속 seed 관련 필드는 이 실행기를 제어하지 않는다.

원본 helper는 slot별 포트 35440·35441·35442와 tag 기반 SHM 이름을 사용한다.
같은 호스트에서 이 캠페인 두 개를 동시에 실행하지 않는다. GPU/포트가 사용 중이면 충돌을 해소한 뒤 실행한다.
busy GPU의 프로세스를 종료하지 않는다. 실패 run을 덮어쓰거나 삭제하지 않는다.
완료·source_unchanged=true인 run은 재실행 명령에서 재사용하며 실패 시 로그를 보존한 새 work를 만든다.
CONFIG는 hash로 고정한다. 다른 설정은 새 작업 디렉터리로 시작한다.

완료 자료를 GPU 실행 없이 다시 분석할 때:

```bash
python -B reproduce.py analyze --work /absolute/path/tree-figures-v14-run
```

## 완료 확인과 보존 자료

- 모든 원본/관측 run: exit_code=0, complete, source_unchanged=true.
- 엔진 commit=dedc85…, dirty 없음, 560 입력 ID/순서 일치, B1/BF16/T0.7.
- 동일 서버·동일 seed의 전체 출력·source·AL event 일치.
- 100개 유효 입력·`C*100` 생성수·RNG/KV 복원 검사 통과.
- `report/STATUS.json`이 complete이고 세 seed가 모두 포함되어야 최종 완료다.
- 그림: `report/paper/FIGURES_1_2_3.pdf`, 수치: `report/paper/NUMBERS.json`, 설명: `report/REPORT.md`.
- CONFIG, 각 run의 PROVENANCE/status/run.json, allocation/choices/paired.jsonl.gz,
  보고 JSON/PDF와 사용한 코드 commit을 보존한다.
- full q는 bounded `/dev/shm` sidecar로 전달 후 삭제한다. 일부 진단 tensor만 남을 수 있다.
  이 기록만으로 새로운 C나 생성 알고리즘을 전부 replay할 수 있다고 가정하지 않는다.

## 코드와 검증 범위

`payload/figure_hooks.py`, `figure_math.py`와 원본 harness 파일은 성공한 최종 v7의 원본을
그대로 보존했다. `ORIGINAL_PAYLOAD_SHA256.json`에 파일 hash를 기록했다.
사용하지 않는 다른 benchmark·중복 엔진 snapshot은 제외했고, 필요한 harness 의존성만 넣었다.
일부 원본 helper의 주석/환경 변수명에 ver1.5/VER15 이름이 남아 있으나 실제 엔진 기준은 위 ver1.4다.

새 실행기는 경로·GPU 설정, 외부 입력 hash 검사, 원본 대 관측 비교, 실행/분석 순서만 담당한다.
`SHA256.json`은 배포된 코드/설정 무결성을 검사한다. 실행 중 소스를 수정해 검증을 무력화하지 않는다.

코드 전용 배포에서 CPU 검사와 원본 자료의 재집계를 확인했다. 원래 고정 observer는
3개 seed full 및 native parity를 통과했지만 **이식 실행기로 GPU smoke/full을 다시 돌린 것은 아니다.**
수신 서버에서 smoke를 통과시킨 뒤 full을 실행한다.

로컬 CPU 회귀 검사(실제 입력 hash 통합 검사까지 포함하려면 환경 변수를 지정):

```bash
PORT_TEST_INPUTS=/absolute/path/INPUTS_FULL.json python -B -m unittest -v test_portability
```
