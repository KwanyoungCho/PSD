# MLSys ver1.10 실행과 인수인계

ver1.10은 **MT-Bench 두 번째 턴에서 첫 질문과 실제 생성 답변이 빠지던 측정 오류**를
수정한다. 기준은 `MLSys-ver1.9@a12ebc25dab2cddf1e3a5bd65938c74396c858f5`,
배포 브랜치는 `MLSys-ver1.10`이다. 변경 대상은 benchmark 입력과 기록이며,
DUET의 모델 실행·tree 확장·residual stochastic 검증 코드는 ver1.9와 동일하다.

## ver1.9에서 바뀐 내용

| 항목 | 이전 | ver1.10 |
|---|---|---|
| 기본 입력 모드 | 560개 독립 raw turn | `base-history`: 문항 내 실제 답변 이력 연결 |
| MT 두 번째 입력 | 두 번째 질문만 | 첫 입력＋실제 첫 답변＋구분자＋두 번째 질문 |
| 출력 상한 계산 | 독립 질문 길이 기준 | 실제 누적 입력 길이 기준 |
| Warmup | 데이터 첫 3턴 성능 제외 | 동일하며, warmup 답변도 다음 턴에 연결 |
| 문맥 초과 | 독립 입력에서 주로 검증 | 입력 초과·출력 제한·engine guard를 구분해 기록 |
| 부분 재측정 | 별도 입력 파일 준비 | `--task mt_bench`로 80대화/160턴 선택 |
| 검증 자료 | 정적 입력 ID 검사 | 입력·답변·연결 해시와 실제 엔진 입력 길이 대조 |

`--input-protocol independent-raw`를 명시하면 이전 입력 방식을 재현한다.
기존 결과는 그대로 보존하며 이력 포함 MT 결과로 바꿔 표기하지 않는다.

## 모델과 입력 규칙

현재 LayerSkip-Llama2-7B는 Llama-2 **base**를 계속 사전학습한 모델이다.
로컬 tokenizer에 chat template이 없으며, Vicuna system prompt나 Llama-2-Chat의
`[INST]`를 추가하지 않는다. 이는 **Spec-Bench 데이터에 대한 base-model 성능 측정 규칙**이고,
공식 MT-Bench chat-quality 평가 포맷 또는 모델의 공식 멀티턴 템플릿이라는 뜻은 아니다.

첫 입력은 제공된 `SPEC_BENCH_FULL.json`의 `input_ids`를 그대로 유지한다.
모델과 vocabulary ID 대응이 같고 decode한 내용이 raw prompt와 정확히 같은지 먼저 검사한다.
입력 파일에 ID가 없을 때만 target tokenizer로 BOS를 포함해 토큰화한다.
과거 고정 tokenizer와 로컬 기본 tokenizer는 같은 문자열을 다른 ID열로 만들 수 있으므로,
기존 결과와 비교할 때는 제공된 고정 입력 파일을 사용한다.

두 번째 입력은 다음 **토큰 배열**이다.

```python
input_ids_2 = (
    input_ids_1
    + output_ids_1_without_terminal_eos
    + tokenizer.encode("\n\n" + raw_question_2, add_special_tokens=False)
)
```

- 답변은 해당 기법·해당 seed가 실제 생성한 전체 토큰이다. 정답 또는 다른 기법의 답변을 넣지 않는다.
- EOS로 끝났다면 종료용 마지막 EOS만 제외한다. 답변 내용의 자르기·strip·decode/re-encode는 하지 않는다.
- 새 BOS나 role tag를 턴 사이에 넣지 않는다. 빈 줄 두 개는 이번 실험이 고정한 구분 규칙이다.
- 질문이 바뀌면 이전 대화 이력을 버린다. 번역·요약·QA·수학·RAG의 기존 raw 지시문·본문·ID는 유지한다.
- 다른 엔진과 서버에서도 같은 tokenizer 파일/버전 및 ID 구성 규칙을 사용한다.
  실제 생성 답변이 다르면 두 번째 입력도 달라질 수 있다. 포맷 동일성은 고정 답변 fixture로 검사한다.

근거: [LayerSkip 모델 설명](https://huggingface.co/facebook/layerskip-llama2-7B),
[Meta Llama-2 base와 Chat의 입력 구분](https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/meta-llama-2).

## 유지하는 측정 조건

문맥 2048, 출력 상한 `min(1024, 2048 - 실제 입력 길이)`, 자연 EOS,
target/draft 온도 0.7, nucleus/top-k 제한 없음, B1/TP1+1을 유지한다.
각 턴의 전체 입력을 cold prefill한다. 답변 이력을 입력에 넣는 것과 턴 사이 KV 재사용은 다르다.
기존 엔진의 speculative lookahead 문맥 보호도 유지하므로 요청한 출력 상한보다 일찍 끝날 수 있다.

| 종료 기록 | 의미 |
|---|---|
| `eos` | 자연 EOS 도달 |
| `output_limit` | 요청한 출력 상한 도달 |
| `context_limit_output` | 누적 입력 때문에 작아진 출력 상한 도달 |
| `context_limit_guard` | 실제 scheduler의 lookahead 문맥 보호 |
| `context_limit_input` | 입력만으로 2048 이상이라 새 출력을 생성할 수 없음 |
| `history_unavailable` | 이전 턴이 입력 초과로 실행되지 않아 후속 턴도 실행하지 않음 |

입력이 넘치면 앞 질문·답변을 잘라 끼워 맞추지 않고 실행하지 않은 사유를 기록한다.
이 턴에 0 TPS나 가짜 시간을 넣지 않는다. `turn_ledger`와 `conversation_population`에 남기고,
유효 턴 수·제외 수를 함께 보고한다. 출력 cap/clip 및 2배 긴 step 제외와 raw 집계는 기존대로다.
기법 간 비교에서는 문맥 제한과 유효 문항 집합 차이도 함께 확인한다.

첫 3개 **데이터 턴**은 성능에서 제외한다. 별도 3회 warmup을 추가하는 방식이 아니다.
원본 순서의 `81_t0`, `81_t1`, `82_t0`가 warmup이고, 측정되는 `82_t1`은 warmup의
`82_t0` 답변을 사용한다. 기존처럼 startup 및 warm/measured 경계의 RNG reseed를 유지하되
대화 이력은 유지한다. 질문 쌍을 독립 GPU로 나누거나 두 번째 턴만 임의 재시작하지 않는다.

입력 조립은 기존 generate 타이머 밖이다. Prefill·decode·generate wall time은 구분해 기록하고,
기존 문항 평균을 유지한다. 첫 답변은 첫 턴에서만 새 출력으로 세며 두 번째 턴에서는 입력 문맥이다.
이전 턴 이력을 포함하는 것만으로 TPS에 prefill을 섞거나 end-to-end latency로 재정의하지 않는다.

## 다른 서버에서 실행

```bash
git fetch origin MLSys-ver1.10
git worktree add -b work/ver110 ../PSD-ver110 origin/MLSys-ver1.10
cd ../PSD-ver110
python ssd/tools/verify_dispatch/check_release.py

export PYTHONPATH="$PWD/ssd"
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/datasets
export SSD_CUDA_ARCH=8.9 TORCH_CUDA_ARCH_LIST=8.9  # 실제 GPU에 맞게 지정
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6,7  # 해당 서버의 topology와 사용 여부 확인
export SSD_DIST_PORT=50710

python -O -B ssd/bench/duet_measure.py \
  --target /path/to/layerskip-llama2-7b \
  --draft /path/to/llama-68m \
  --prompts results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json \
  --input-protocol base-history --task mt_bench \
  --dtype float16 --expansion frontier --seed 42 \
  --output /path/to/new_mt_seed42/run.json

python ssd/tools/verify_dispatch/check_history.py /path/to/new_mt_seed42/run.json
```

seed는 `42 → 123 → 2026`으로 바꾸어 같은 명령을 실행한다.
`--task all`이 기본이며 480문항/560턴 전체를 실행한다.
`--task mt_bench`는 80대화/160턴 전체다. `--limit N`은 선택한 task의 앞 N개 **완전한 문항**이다.
병렬 실행 시 다른 GPU pair·port·output을 쓴다. `check_history.py`의 `--target`, `--prompts`로
다른 서버에서 tokenizer/데이터 경로를 지정할 수 있다.

성능 모드는 기본이며 profile/probe는 OFF다. ver1.9 preset의 기본 tree 확장 방식 `last`를 유지한다.
위 명령은 최근 채택 실험과 맞추기 위해 **`--expansion frontier`를 명시**했다.
P1은 K10/R4(위치당)/W40/G15/C2, P2는 K5/R16/W20/G15/C3, exit는 21 zero-based다.
모델·GPU 경로 또는 새 frontier 기본값을 runtime에 하드코딩하지 않는다.

다른 서버의 입력 규칙을 GPU 실행 전에 확인하려면:

```bash
python ssd/bench/specbench_history.py \
  --target /path/to/layerskip-llama2-7b \
  --prompts results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json \
  --export /new/path/PROTOCOL_FIXTURE.json
```

## 검증과 결과의 범위

배포 검증은 [VALIDATION.json](results/mlsys_ver110/VALIDATION.json)과
[회귀 검사 로그](results/mlsys_ver110/CPU_TESTS.log)에 기록한다.
실제 모델 검사는 첫 3개 MT 대화와 단일 번역 입력, 명시적 문맥 초과 검사 입력을 사용한다.
출력 상한은 1024이며 실제 이전 답변과 warmup 경계 연결, 동적 입력 길이, 초과 요청 제외,
단일 턴 입력 보존을 확인한다. 이 짧은 검사는 새로운 MT 3-seed 성능 결과가 아니다.

수정한 MT만 재측정하고 다른 400문항의 기존 결과를 사용할 수 있다. 이때 모델·입력 ID·파라미터·
측정 기준을 유지하고 **task별 부분 재측정**으로 기록한다. MT에서 난수 소비가 바뀌므로 기존
400문항을 붙인 결과를 수정 버전으로 전체 데이터를 연속 실행한 동일-seed 결과로 표현하지 않는다.
전체 TPS/AL은 MT 결과를 교체한 뒤 턴→문항→seed 기준으로 재집계한다.
Reference들도 같은 이력·입력 규칙을 적용해야 비교 가능하다. 이 배포는 다른 저장소의 실행기를
자동 수정하거나 기존 논문 표의 성능 수치를 바꾸지 않는다.

가중치와 Python/CUDA 환경은 Git에 포함하지 않는다. 배포 브랜치 번호와 upstream 패키지 버전은 별개다.
