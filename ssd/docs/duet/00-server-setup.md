# DUET 새 서버 셋업 및 실행 가이드

이 문서는 **DUET 실험을 처음 보는 서버에서 재현**할 때 필요한 것을 한곳에
모은다. 설계와 알고리즘은 [`TREE_IMPLEMENTATION.md`](TREE_IMPLEMENTATION.md)와
저장소 루트의 [`MESA-SSD.md`](../../../MESA-SSD.md)를 본다. 이 문서는 **무엇을
어디에 두고 어떤 명령으로 돌리는가**만 다룬다.

---

## 1. 먼저 알아야 할 것: 이 저장소만으로는 논문 실험이 돌지 않는다

논문 수치를 만든 실행 드라이버와 지표 스크립트는 **PSD 저장소 밖**에 있고,
git으로 관리되지 않는다. 새 서버로 옮길 때 반드시 함께 가져가야 한다.

| 자산 | 원래 위치(구 서버) | 크기 | 역할 |
|---|---|---|---|
| `runners/run_duet.py` 외 3개 | `<HOME>/baseline/runners/` | 220 KB | **논문용 실행 드라이버.** `bench/bench.py`가 아니라 이쪽이 정본이다 |
| `data/specbench_full.jsonl` | `<HOME>/baseline/data/` | 3.6 MB | Spec-Bench 480문항 / 560턴 |
| `scripts/metrics/`, `scripts/timeline/` 등 | `<HOME>/DUET_PAPER_RESULTS/scripts/` | 392 KB | 문항 단위 집계, context-safe 필터, figure 생성 |
| `.venv-ssd` | `<HOME>/baseline/.venv-ssd` | 8.6 GB | Python 3.11.15 실행 환경 (새 서버에서는 재생성 권장) |

venv를 뺀 실질 전송량은 **약 5 MB**다. 예시:

```bash
tar czf duet-external.tgz \
  baseline/runners baseline/data baseline/analysis \
  baseline/PAPER_RESULTS.md baseline/REPRODUCE.md \
  DUET_PAPER_RESULTS/scripts DUET_PAPER_RESULTS/PAPER_RESULTS.md
```

`ssd/experiments/**/run_*.sh`는 이 경로들을 스크립트 상단 `BASE=`/`PAPER=`
변수로 참조한다. 새 서버에서는 그 두 줄만 고치면 된다.

---

## 2. 하드웨어 요구량

`B=1`, `max_model_len=4096`, bf16 기준. 값은 모델 config와 엔진의 할당식에서
계산한 것이다.

| 모델 조합 | Target (TP2, GPU당) | Draft (1 GPU) | 최소 구성 |
|---|---|---|---|
| LayerSkip-Llama2-70B + TinyLlama-1.1B | 가중치 64.2 GiB → **~70 GiB** | **~8 GiB** | 80 GB×2 + 16 GB×1 |
| LayerSkip-Llama2-7B + AMD-Llama-135m | 6.3 GiB → ~10 GiB | ~3 GiB | 24 GB×3 |
| LayerSkip-Llama3-8B + Qwama-0.5B | 7.5 GiB → ~12 GiB | ~4 GiB | 24 GB×3 |

70B를 bf16 TP2로 올리면 **카드당 64.2 GiB가 가중치 고정비**다. 80 GB 미만
카드(A100 40 GB, L40S 48 GB 등)에는 TP2로 들어가지 않으므로 TP4가 필요하다.

### KV 캐시는 남는 만큼 전부 잡는다

[`model_runner.py`](../../ssd/engine/model_runner.py) `allocate_kv_cache()`는
상한 없이 `usable = free × gpu_memory_utilization`(target 0.7 / draft 0.75~0.8)
을 잡는다. 96 GB 카드에서 실제 점유는 다음과 같다.

| | 실제 할당 | `B=1`/ctx 4096 실사용 | 배율 |
|---|---|---|---|
| Target GPU | ~21 GiB (548 blocks) | 640 MiB (16 blocks) | 34× |
| Draft GPU | ~74 GiB (13,822 blocks) | 88 MiB (16 blocks) | 864× |

**공유 서버에서는 `--gpu_memory_utilization`을 낮춰 실행한다.** 낮춰도 `B=8`
까지는 여유가 충분하다(target 5 GiB, draft 0.7 GiB). 다만 draft 쪽 비율은
현재 [`draft_runner.py`](../../ssd/engine/draft_runner.py)에 0.75/0.8로
하드코딩되어 있어, draft만 따로 낮추려면 그 값을 config에서 받도록 고쳐야 한다.

---

## 3. 소프트웨어 환경

```bash
cd ssd
uv sync                     # pyproject.toml 기준
# torchao 0.12.0은 pyproject 핀 위에 수동 설치한다
```

베이스라인은 FlashInfer 버전이 충돌하므로 **반드시 별도 env**에 둔다.

| env | 용도 |
|---|---|
| `ssd` (`.venv-ssd`) | DUET 엔진과 논문 러너 |
| `sglang` | SpecInfer / EAGLE-2 / EAGLE-3 (`bench/run_sglang_bench.py`) |
| `vllm016` | vLLM 비교 (`bench/run_vllm_bench.py`) |
| `awq-quant` | AutoAWQ 캘리브레이션만 |
| `PSD` | 루트의 오프라인 분석 스크립트(HF Transformers) |

---

## 4. 모델·데이터 경로

경로는 전부 환경 변수로 주입한다. 코드에 하드코딩된 기본값은 구 서버 것이라
신뢰하지 않는다.

```bash
export HF_HOME=/path/to/models
export SSD_HF_CACHE=${HF_HOME}/hub          # 미설정 시 import 단계에서 즉시 실패
export SSD_DATASET_DIR=/path/to/baseline/data
```

필요한 체크포인트:

```text
facebook/layerskip-llama2-70B          # target (bf16, 80 layers, vocab 32000)
TinyLlama/TinyLlama-1.1B-Chat-v1.0     # draft
facebook/layerskip-llama3-8B           # 추가 실험용 target (vocab 128256)
meta-llama/Llama-2-70b-chat-hf         # SpecInfer/EAGLE-2 베이스라인용
lmsys/sglang-EAGLE-llama2-chat-70B     # EAGLE-2 draft head
```

> **주의 — vocab 상한.** 현재 wire는 proxy 점수 `P_iv`를 token id의 상위
> 비트에 패킹하므로 `vocab_size ≤ 32768`이 config의 하드 제약이다.
> Llama-3 계열(vocab 128,256)을 쓰려면 wire를 먼저 확장해야 한다.

---

## 5. GPU 배치와 환경 변수

target을 TP로 앞쪽 GPU에, draft를 마지막 GPU에 둔다. `--gpus N`이면
`num_tp_gpus = N-1`, `draft_rank = N-1`이다.

```bash
export CUDA_VISIBLE_DEVICES=5,6,7       # target TP2(5,6) + draft(7)
export SSD_DIST_PORT=18830              # 동시 실행 시 run마다 다르게
```

### 공통 실행 최적화 (chain/tree 양쪽에 동일 적용)

```bash
export SSD_CHAIN_PROXY_GRAPH=1     # chain proxy를 CUDA Graph로
export SSD_DUET_EXIT_REPLICA=1     # rank0 LM head 복제 → exit 랑데부 제거
export SSD_ASYNC_PROXY_SEND=1      # persistent buffer 비동기 send ring
export SSD_PROXY_STREAM=0          # 별도 proxy stream은 끈다(replica stream 사용)
```

이 네 개를 한쪽 arm에만 켜면 topology 효과와 통신 개선이 섞인다. **비교 실험은
항상 양쪽에 같이 적용한다.**

### GPU 아키텍처별 attention 백엔드

```bash
export SSD_CUDA_ARCH=12.0 TORCH_CUDA_ARCH_LIST=12.0   # RTX PRO 6000 = sm_120
export SSD_ATTN_BACKEND=auto                          # sm_120이면 flashinfer 선택
```

sm_120에서는 설치된 `sgl-kernel` attention이 지원되지 않아 일반
prefill/decode/chain verify까지 FlashInfer로 간다. 다른 GPU에서는 `auto`가
기존 `sgl-kernel`을 그대로 고르므로 기존 champion 경로의 지연에 영향이 없다.
`SSD_ATTN_BACKEND=sgl`은 sm_120에서 사용하지 않는다.

### 측정 전 반드시 해제할 진단 스위치

```bash
export SSD_PROFILE=0 SSD_PROFILE_DUET=0 SSD_PROFILE_DUET_DETAIL=0
unset SSD_TREE_STAGE1 SSD_TREE_STAGE2 SSD_TREE_TOPO_TRACE \
      SSD_TREE_NODE_AUDIT SSD_TREE_EXEC_DELAY_MS SSD_TREE_GAP_PROF \
      SSD_CG_INPUT_CHECK SSD_TREE_CALIB_TRACE SSD_DUET_E0_TRACE \
      SSD_DUET_PROXY_ON_DRAFT SSD_DUET_EXIT_TOPM_GATHER
```

프로파일러를 켠 런은 trace D2H와 파일 I/O가 스케줄링을 바꿔 **arm 간 격차를
왜곡한다.** 과거에 이 때문에 K1=9를 동급으로 오판한 적이 있다. 타임라인
figure용 런만 `SSD_PROFILE_DUET=1`(event cap 12000)로 따로 돌린다.

---

## 6. 정본 실행 명령 (chain champion)

현재 성능 기준선이자 논문 champion은 **P1/P2 모두 `off`(= chain)** 이다.

```bash
CUDA_VISIBLE_DEVICES=5,6,7 SSD_DIST_PORT=18830 \
SSD_TREE_EXEC=0 SSD_TREE_ARENA=0 SSD_TREE_PROXY_GRAPH=0 SSD_TREE_EXEC_WARMUP=0 \
"${BASE}/.venv-ssd/bin/python" -O "${BASE}/runners/run_duet.py" \
  --target facebook/layerskip-llama2-70B \
  --draft  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --gpus 3 \
  --k1 8 --k2 4 --exit-layer 56 \
  --p1-fanout 3 --p2-budget 15 --proxy-top-k 28 \
  --temp 0.7 --top_p 1.0 --max_new_tokens 1024 \
  --max_model_len 4096 --extend-draft-rope --allow-nonpaper-context \
  --template raw --warmup 2 \
  --p1-tree off --p2-tree off --p1-allocation-policy backbone \
  --seed 42 --data "${BASE}/data/specbench_full.jsonl" \
  --resume --out out/chain_s42.jsonl
```

동적 tree arm은 위에서 다음만 바꾼다.

```bash
SSD_TREE_EXEC=1 SSD_TREE_ARENA=1 SSD_TREE_PROXY_GRAPH=1 SSD_TREE_EXEC_WARMUP=all
  --p1-tree on --p2-tree on \
  --roots-per-position 3 --root-count 10 --c-tensor 2 \
  --n1 14 --p1-verify-nodes 12 --n2 8 --p2-verify-nodes 8 \
  --p1-start-threshold 0 --p1-conf-threshold 0 \
  --p2-proxy-threshold 0.01 --p2-conf-threshold 0.01
```

### 최근 추가된 플래그 (2026-08)

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--extend-draft-rope` | off | draft의 RoPE 캐시를 `max_model_len`까지 해석적으로 확장한다. TinyLlama의 native 2,048 window를 넘겨 ctx 4096으로 돌릴 때 필요하다. **품질 저하 가능성이 있으므로 켠 사실을 결과에 함께 적는다** |
| `--allow-nonpaper-context` | off | `max_model_len != 2048`을 허용한다. RoPE 확장과 함께 쓴다 |
| `--only-proxy` | off | **chain 전용 ablation.** 논리적으로 `K1=0`으로 두어 P1 생성을 건너뛰고 proxy만으로 cache를 채운다. 캐시 예산 figure의 Only-Proxy 곡선이 이 경로다. tree와 함께 켤 수 없다 |
| `--p1-allocation-policy` | `dynamic` | P1 root 간 예산 배분. `backbone`은 root마다 full-depth continuation을 보장한다. P1 tree 확대 실험은 `backbone`을 명시한다 |

GPU 커널 스위치 두 개는 **기본이 켜짐(`1`)** 이며, tree 경로에서만 의미가 있다.
chain 실험 스크립트는 비교를 깨끗하게 하려고 명시적으로 `0`으로 둔다.

```bash
export SSD_TREE_TOPOLOGY_GPU=1      # target측 tree metadata를 GPU에서 구성
export SSD_P1_RERANK_PRECOMPUTE=1   # P1 hit 시 verify view rerank 선계산
```

### 사후 처리

```bash
"${PY}" "${PAPER}/scripts/metrics/filter_context_safe_subset.py" raw.jsonl safe.jsonl
"${PY}" "${PAPER}/scripts/metrics/question_level_metrics.py" --json raw.jsonl safe.jsonl
```

`filter_context_safe_subset.py`는 `prefill + 1024 ≤ 2048`인 문항만 남긴다.
**이 필터가 제거하는 24문항은 전부 summarization**이며, 원인은 TinyLlama의
native 2,048 window에서 발생하는 AL 붕괴다. 결과를 보고할 때 이 사실을 함께
적는다.

---

## 7. 실행이 정상인지 확인하는 항목

짧은 smoke(`RUN_SCOPE=smoke`) 후 다음을 본다.

- 종료 코드 `EXIT:0`
- executor `fallback=0`, `error=0` — fallback이 0이 아니면 지원되지 않는
  shape로 참조 경로가 돈 것이다
- tree arm이면 `p2exec replay > 0`
- P1/P2 hit와 accepted length가 finite
- 종료 후 draft/target GPU 프로세스가 남지 않음
- 로그에 `Traceback|CUDA error|RuntimeError|AssertionError` 없음
- 산출 JSONL 행 수가 정확히 560

---

## 8. 새 서버에서 다시 캘리브레이션해야 하는 값

exit layer, K1, K2, threshold는 **GPU/모델/배치가 바뀌면 이전 값이 무효**다.
전부 grid로 쓸지 말고 시간 균형을 먼저 측정한 뒤 소수 후보만 최종 TPS로 가른다.

| 파라미터 | 결정 근거 |
|---|---|
| exit layer | proxy 품질 ↑ vs P2 시작 지연 ↑의 상충. 구 서버 값은 56이지만, 2026-08-14 postmortem은 **49**로 이동했다 |
| K1 | `P1 완료 시각 − proxy 도착 시각`이 0에 가장 가까운 정수 |
| K2 | `P2 완료 − target verify 완료`가 0에 가장 가까운 정수 |
| threshold | trace를 0/0으로 모아 사후 hit/accepted child를 라벨로 재계산 |

```bash
bash tools/duet_calibration/calibrate_k_balance.sh
python tools/duet_calibration/analyze_k_balance.py /path/to/profile_dirs
bash tools/duet_calibration/collect_tree_thresholds.sh
python tools/duet_calibration/analyze_thresholds.py --input /path/to/trace.jsonl
```

도구가 추천한 값은 **시간 균형 후보이지 최종 champion이 아니다.** 반드시
프로파일러를 끈 paired run으로 확정한다.

---

## 9. 측정 방법론 (어기면 결론이 뒤집힌다)

2026-08 실험에서 실제로 겪은 실패를 규칙으로 만든 것이다.

1. **프로파일러 OFF + 동일 UID paired + 3-seed**로만 파라미터를 판정한다.
   subset에서 이긴 설정이 full Spec-Bench에서 뒤집힌 사례가 두 번 있다
   (N2/M2=12/10: subset +3.36% → full −1.47%; P2 conf 0.02: +3.29% → −2.11%).
2. **1–3% 델타는 주장하지 않는다.** tree 설정을 바꾸면 같은 seed에서도 출력
   해시가 560개 중 6개만 일치할 만큼 샘플링 궤적이 갈린다.
3. **arm 실행 순서를 seed마다 회전**시킨다(예: 42는 chain→tree, 123은
   tree→chain). GPU 상태 드리프트가 한쪽에 몰리는 것을 막는다.
4. 모든 표와 그림은 raw JSONL에서 **스크립트로만** 생성한다.

기록해야 할 최소 지표는 다음과 같다. 전체 AL 하나만 보면 원인을 알 수 없다.

| 범주 | 지표 |
|---|---|
| 품질 | tokens/step, 전체 accepted length |
| P1 / P2 | hit rate, 조건부 AL, `hit×(AL+1)` |
| 속도 | decode TPS, target step p50, draft step p50 |
| executor | phase별 replay / capture / fallback / error |
| target | verify 준비, graph pre/post, proxy 계산·전송 |
| startup | tree all-page warmup 시간과 메모리 |

---

## 10. 자주 밟는 함정

- **`python -O`로 돌린다.** 벤치마크는 `-O`가 전제인데, 이때 `assert`가
  사라진다. 외부 입력 계약은 `ValueError`/`RuntimeError`로 옮겼지만
  `config.__post_init__`에는 아직 `assert`가 남아 있다. stale한 `--k` 값이
  조용히 통과해 결과를 훼손한 전례가 있다.
- **`--k`는 생략한다.** 지정하지 않으면 `K1+K2`로 자동 설정된다. P1 fanout
  목록을 쓸 경우 길이는 `K1+1`이어야 한다.
- **`K2 ≤ K1`은 명시적 계약**이다. graph 레이아웃과 proxy 위치 범위가 이를
  전제하므로 검사만 지우면 안 된다.
- **동적 tree는 `B=1` + `temperature > 0` 전용.** `B>1`과 `temp=0`은 chain
  fallback으로 간다. greedy tree는 ordered residual sampling이 아니라 별도의
  top-C proposal + argmax verifier가 필요하므로 gate를 그냥 제거하면 안 된다.
- **CUDA Graph 캡처와 재생은 query 폭별로 같은 FlashInfer wrapper/plan buffer를
  써야 한다.** 다른 이름의 wrapper로 재계획하면 crash 없이 logits와 accepted
  length가 훼손된다.
- **batch graph bucket은 실제 입력 buffer 행 수 이하만 만든다.** `B=1`의 decode
  buffer가 2행이면 1·2 bucket만 만들고 4·8은 계획하지 않는다.

---

## 11. 관련 문서

| 문서 | 내용 |
|---|---|
| [`TREE_IMPLEMENTATION.md`](TREE_IMPLEMENTATION.md) | tree 설계·실행기·검증의 기준 문서, 실험 이력, 논문 주장 경계 |
| [`../../../MESA-SSD.md`](../../../MESA-SSD.md) | 방법 수준의 DUET 명세(P1/P2, proxy 점수, 파라미터) |
| [`04-split-k1k2-design.md`](04-split-k1k2-design.md) | split-K1/K2 실행 계약 |
| [`13-b-gt-1-design.md`](13-b-gt-1-design.md) | `B>1` 확장 설계(미구현) |
| [`../quantization/03-final-report.md`](../quantization/03-final-report.md) | AWQ W4A16 경로 |
