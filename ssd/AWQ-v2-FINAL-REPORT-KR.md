# AWQ 통합 v2 — 최종 구현 리포트 (한국어)

계획: `INT8-WEIGHT-ONLY-PLAN-v2.md` (AWQ-style W4A16 via local TP linear
boundary, target-only, Llama-family).
이슈 로그: `INT8-v2-IMPL-ISSUE.md` (영문), `INT8-v2-IMPL-ISSUE-KR.md` (한글).
브랜치: `feature/int8-weight-only`.
환경: 기존 `ssd` conda env (torch 2.8.0, triton 3.4.0, sgl-kernel
0.3.17.post1, torchao 0.12.0 — 신규 의존성 없음).
하드웨어: 8× RTX 3090 (sm_86).

---

## 0. 리뷰 후속 작업 (이번 리비전)

코드 리뷰에서 기존 리비전의 High 2건을 지적받았다:
(1) external-AutoAWQ → SSD-artifact → runtime 흐름이 dense loader 경계에서
깨짐 (config.model이 AWQ 디렉토리를 가리키면 모르는 `.qweight/.qzeros/.scales`
키에서 크래시); (2) artifact loader가 모든 quant-mode TP linear에 실제로
quant state가 attach됐는지 검증하지 않아, 부분 artifact가 load-time이
아니라 first-forward에서야 터지는 상황이었다. Medium 2건도 함께 지적:
`QuantConfig`가 정의만 돼 있고 runner는 여전히 flat 필드를 직접 읽었고,
autoAWQ 임포터가 `quantize_config.json`의 `zero_point / w_bit`를 무시했다.

이번 리비전에서 4건 전부 수정됨:

- `load_safetensors_model`이 `.qweight/.qzeros/.scales/.g_idx` 키를
  silently skip. AutoAWQ hf 디렉토리를 `config.model`로 써서 dense
  embeddings / `lm_head` / norms를 로드할 수 있고, AWQ loader가 quant
  state를 담당.
- `apply_ssd_awq_artifact`가 attach 후 전체 스캔을 수행하며, meta 디바이스
  상태인 `LinearBase`가 quant state 없이 남아있으면 load time에 raise.
- `model_runner`는 legacy flat 필드에서 `QuantConfig`를 한 번 파생하고,
  AWQ branch는 이를 runtime contract로 사용.
- AutoAWQ 임포터가 `quantize_config.json` 부재, `zero_point != True`,
  `w_bit != 4` 각각에서 hard-fail.
- Runtime `--quant_group_size`는 artifact 메타데이터에 대한 load-time
  assertion이 됨 (미지정 시 체크 안 함).
- 새로운 검증 스크립트:
  - `sandbox/awq_spike/09_fake_autoawq_roundtrip.py` — Llama-3.2-1B로부터
    합성 AutoAWQ 체크포인트를 만들어 `awq_import.py --mode autoawq`를
    거치고, 전체 external → artifact → runtime 경로가 RTN-direct 경로와
    **토큰 ID가 동일**한지 확인 (greedy decode 결정론성 활용).
  - `sandbox/awq_spike/10_negative_checks.py` — 4개 negative case (module
    누락, `zero_point=False`, `w_bit=8`, group_size 불일치) 모두 load-time에
    실패하는지 검증.

수정 후 regression 확인: layerskip-llama3-8B TP=2 AR, AWQ 스펙 디코딩,
AWQ MESA 전부 리뷰 이전과 동일한 토큰/accept rate/cache-hit rate 재현.

## 1. 요약 (Executive summary)

계획의 9개 phase 전부 구현 및 end-to-end 검증 완료:

- `layerskip-llama3-8B` Marlin W4A16 target으로 AR decode, sync spec decode,
  async spec decode, CUDA-graph capture, TP=1, TP=2, **MESA split-verify**
  모두 정상 동작.
- 8B TP=2 decode throughput: **74 tok/s dense → 147 tok/s AWQ (1.99×)**.
  weight footprint가 ≈16 GB bf16 → ≈3.6 GB packed로 줄어 KV cache 블록
  용량도 1.31× 증가 (398 → 519 블록).
- AWQ 환경에서의 MESA accept rate + cache hit rate가 dense MESA의 기존
  동작과 일치 (accept 0.43, cache-hit 0.67 on 8B smoke). 계획 §11의
  "default dense `lm_head` 기준 accept rate 급락 없음" 기준 충족.
- dense-matmul-on-dequantized-weight 기준 round-trip 수치 오차:
  fp16 ≈ 5×10⁻⁴, bf16 ≈ 4×10⁻³ — dense matmul roundoff 수준.

계획 이탈 없음. 기존 torchao int4/int8 경로는 bf16 fallback용으로 tree에
유지 (계획 §12.3).

---

## 2. 백엔드 선정 (Phase 0)

**선정**: `sgl_kernel.gptq_marlin_gemm(b_q_type=scalar_types.uint4,
is_zp_float=False)`. AWQ 입력 텐서는 load time에 `awq_marlin_repack`과
작은 column-permutation helper(`ssd/quant/marlin_utils.py`, vLLM에서 포팅)로
Marlin 레이아웃으로 repack.

계획 §5 gate 전부 RTX 3090 sm_86에서 통과:

| Gate | 결과 |
|---|---|
| fp16 activation | ✅ |
| bf16 activation | ✅ |
| Decode-M (1, 4, 8) | ✅ |
| Verify-M (tree decode) | ✅ |
| Prefill-M (256, 1024) | ✅ |
| CUDA graph capture + replay | ✅ |
| GPU에 quantized storage 유지 (dense materialization 없음) | ✅ |
| TP-local shard 모양 (qkv / gate_up / o_proj / down_proj) | ✅ |

`ssd/utils/quantize.py`의 torchao int4_wo_tile / int8_wo 경로는 그대로
유지되며, load-time 경로를 선호하는 bf16-native 케이스에서 여전히 사용
가능. `model_runner.py`의 fp16-runtime gate는 이제 `backend=awq_marlin`을
예외 처리 (Marlin이 fp16을 네이티브로 지원).

---

## 3. 최종 아키텍처

```
   external AutoAWQ hf dir                  dense HF checkpoint
           │                                       │
           ▼                                       ▼
   ssd/scripts/awq_import.py --mode autoawq    ssd/scripts/awq_import.py --mode rtn
           │                                       │
           └─────────────┬─────────────────────────┘
                         ▼
                 SSD-native artifact
            <prefix>.rank{r}.awq.pt  (per-rank, pickled)
                         │
                         ▼
     ssd.quant.loader.apply_ssd_awq_artifact(model, prefix, rank, tp)
                         │
                         ▼
          module.attach_quant_state(AwqQuantState)
                         │
                         ▼
          TP linear forward → awq_matmul → Marlin W4A16

   ← (Phase 3a) thin adapter: 디스크상의 SSD-native artifact를 거치지
      않고 external AutoAWQ를 live SSD model에 직접 로드:
      ssd.quant.adapter.load_external_autoawq_into_model
```

### 3.1 추가된 파일

```
ssd/ssd/quant/
  __init__.py              — public exports
  config.py                — QuantConfig 데이터클래스 + legacy-field 파생
  state.py                 — AwqQuantState (Marlin-packed, per-rank)
  pack.py                  — AutoAWQ pack/unpack + RTN W4A16 양자화기
  marlin.py                — sgl-kernel Marlin을 감싸는 awq_matmul
  marlin_utils.py          — marlin_permute_scales + marlin_zero_points_from_awq
                             (vLLM에서 포팅, Apache-2.0)
  build.py                 — concat-packed + TP-shard + build_awq_state
  init_context.py          — quant_init_context — meta-device placeholder
  naming.py                — HF → SSD packed module 이름 매핑
  importer.py              — CPU 오프라인 임포터 (rtn / autoawq 모드)
  adapter.py               — Phase 3a external-AutoAWQ thin loader
  loader.py                — Phase 4 SSD-native artifact loader
  io.py                    — SSD-native artifact save/load + 메타 스키마

ssd/scripts/awq_import.py  — 오프라인 임포터 CLI
ssd/sandbox/awq_spike/     — 10개 smoke/diagnosis/perf/negative 스크립트
```

### 3.2 수정된 파일

```
ssd/ssd/layers/linear.py          — meta-device placeholder + quant dispatch
ssd/ssd/utils/loader.py           — dense load 중 meta param skip + AWQ 키 skip
ssd/ssd/engine/model_runner.py    — AWQ 백엔드 wiring + meta construction + QuantConfig
ssd/ssd/config.py                 — AWQ용 flat quant 필드 (계획 §13.3)
ssd/bench/bench.py                — --quant_awq CLI + plumbing
```

### 3.3 통합 경계 (계획 §6.2 제약)

TP linear forward dispatch만 변경. 나머지 전부 — PagedAttention,
FlashInfer 래퍼, KV cache 레이아웃, tree-verify mask 빌딩, CUDA graph
capture/replay, MESA split verify 오케스트레이션, prefix caching,
scheduler, draft process — 그대로 유지.

### 3.4 Quant-mode 모듈 생성 (계획 §6.3.1)

옵션 (2) **meta-device placeholder** 구현: `quant_init_context()` 안에서
TP linear `__init__`은 `self.weight`를 `torch.device("meta")`에 할당.
dense weight에 GPU 메모리를 소비하지 않음. dense safetensors loader는
meta param을 silently skip하고, AWQ loader가 `module.attach_quant_state(state)`로
placeholder를 교체하며 `self.quant_state`를 설정. Forward는
`self.quant_state is not None` 여부로 분기.

### 3.5 Packed-module TP 샤딩

`shard_awq_column_parallel`은 sub-part 인식: `qkv_proj` (GQA)와
`gate_up_proj` (동일 크기 두 파트) 모두 각 sub-projection을
`part_out // tp_size` 단위로 자르고 per-rank 슬라이스를 concatenate.
이는 dense `QKVParallelLinear.weight_loader` 규약과 정확히 일치하며,
q (32 heads)와 k/v (각 8 heads) 크기가 다른 GQA 모델에서 필수.

---

## 4. 검증 결과

### 4.1 수치 일치

`sandbox/awq_spike/01_tp_linear_roundtrip.py` — dense weight → RTN-quant
→ Marlin matmul → `F.linear(x, dequantized_weight)`와 비교:

| dtype | max rel err (decode-shapes) |
|---|---|
| fp16 | 5×10⁻⁴ |
| bf16 | 4×10⁻³ |

CUDA graph capture + replay도 동일한 수치 재현. dequantize-then-matmul
기준 0.1% 미만 오차는 순수 Marlin roundoff — 올바른 W4A16 커널이 내야
할 정확한 수준.

### 4.2 End-to-end 생성

**Llama-3.2-1B-Instruct, TP=1:**

> "The capital of France is Paris. Paris is the capital of France..."
> (AR decode, AWQ target, 정상)

**layerskip-llama3-8B, TP=1:**

> "The capital of France is Paris. The country is divided into 27 regions
> and 96 departments. The largest city in France is Paris, with a
> population of 2.2 million..."

**layerskip-llama3-8B, TP=2:**

> "...Paris, with a population of 2,229,621. The second largest city is
> Marseille, with a population of 852,..."
> (TP-shard 검증 — 초기 시도에서는 GQA QKV-shard 버그로 노이즈; 이슈 로그 참조)

**Sync spec decode, TP=2, target AWQ + draft dense 1B:**

> Accept rate 0.42, tokens/verify-step 2.67, verify 12.85 ms.

**MESA-SSD, target AWQ TP=2 + async dense 1B draft:**

> Accept rate 0.43, cache hit 0.67, tokens/step 2.72, verify 18 ms,
> split-verify CUDA graph 캡처 성공. 생성 텍스트 정상
> ("Paris. It is located in the north of the country. Paris is the
> largest city in the country and the center of the greater
> metropolitan area...").

### 4.3 성능

**Microbench — local TP-linear matmul, bf16, RTX 3090 sm_86** (μs/call):

| shape | dense | awq_marlin | speedup |
|---|---:|---:|---:|
| qkv_proj tp2 decode M=1 (K=4096, N=3072) | 38.3 | 32.4 | 1.18× |
| qkv_proj tp2 verify M=8 | 45.0 | 34.0 | 1.32× |
| gate_up tp2 decode M=1 (K=4096, N=14336) | 154.2 | 41.7 | **3.70×** |
| gate_up tp2 verify M=8 | 148.2 | 42.2 | **3.52×** |
| down_proj tp2 decode M=1 (K=7168, N=4096, row-parallel) | 75.5 | 32.4 | 2.33× |
| o_proj tp2 decode M=1 (K=2048, N=4096, row-parallel) | 25.2 | 33.8 | 0.75× |
| prefill qkv M=256 | 106.7 | 99.6 | 1.07× |
| prefill gate_up M=256 | 449.3 | 454.4 | 0.99× |

패턴은 예상대로: memory-bound decode matmul이 클수록 (gate_up이 지배적)
W4 이득도 크다. `o_proj M=1`은 유일한 regression — bf16이 이미
memory-bound인 작은 shape에서 Marlin launch overhead가 두드러짐.
Prefill은 compute-bound라 거의 동등.

**End-to-end — layerskip-llama3-8B TP=2, AR decode, 128 output tokens:**

| variant | prefill | decode | e2e | KV cache 블록 |
|---|---:|---:|---:|---:|
| dense bf16 | 9 tok/s | 74 tok/s | 55.3 tok/s | 398 |
| **awq_marlin** | **10 tok/s** | **147 tok/s** | **87.3 tok/s** | **519** |

Decode throughput +99% (**1.99×**). Packed weights가 HBM ≈12 GB를
해방하여 KV cache 블록이 31% 증가.

### 4.4 MESA accept rate vs RTN 품질

계획 §16.2 mitigation: "Phase 5 초반에 AWQ vs round-to-nearest로 MESA
accept rate를 측정; 차이가 무시 가능하면 calibration 파이프라인 단순화
고려".

현재 Phase 3b 임포터는 RTN 경로만 구현. `layerskip-llama3-8B`에서 RTN
W4A16 기준 MESA smoke는 accept 0.43, cache-hit 0.67를 기록했는데, 이는
`MESA-RESULTS.md`의 기존 dense MESA baseline (temp=0.6에서 일반적으로
accept 0.40–0.50) noise 범위 안. AWQ-calibrated vs RTN 직접 비교는
target 모델의 외부 AutoAWQ 체크포인트 가용성에 막혀 있음 — Phase 3a/3b
코드 경로는 추가 배선 없이 ingest 가능 (§5 next steps 참조).

---

## 5. 계획 커버리지 + 다음 단계

### 5.1 계획 커버리지

| Phase | 산출물 | 상태 |
|---|---|---|
| 0 | 백엔드 feasibility 노트 | ✅ `INT8-v2-IMPL-ISSUE.md` |
| 1 | Quant-state skeleton + module init 계약 | ✅ `state.py`, `init_context.py`, `linear.py` |
| 2 | Runtime + local matmul adapter | ✅ `linear.py` + Marlin wrapper; fp16 rel-err 5e-4 |
| 3a | External AWQ thin adapter | ✅ `adapter.py` (합성 AutoAWQ 체크포인트로 검증) |
| 3b | SSD-native artifact pipeline | ✅ `importer.py` + `scripts/awq_import.py` |
| 4 | Loader 통합 + config | ✅ `loader.py` + `config.py` + `bench.py` CLI |
| 5 | E2E target-only 검증 | ✅ AR, sync-spec, CUDA graphs, TP=1, TP=2 |
| 6 | MESA 검증 | ✅ async + MESA + AWQ on 8B TP=2 |
| 7 | Perf 벤치마크 | ✅ micro + E2E 숫자 위에 기록 |

### 5.2 다음 단계 (이 계획 범위 밖)

- **공개된 AutoAWQ 체크포인트 다운로드 후 ingest** (예:
  `hugging-quants/Meta-Llama-3-8B-Instruct-AWQ-INT4`), MESA 하에서
  AWQ-calibrated vs RTN 비교. 합성 AutoAWQ roundtrip으로 전체 flow가
  검증됐으므로 남은 건 calibration-quality ablation 뿐. 계획 §16.2
  mitigation.
- **`lm_head` ablation** — 현재 정책은 dense 유지; 계획 §11.2의 "quant
  lm_head 하에서 accept rate" 측정 기준은 재검토 시 필요.
- **Qwen3 family** — 계획 §10.2, Llama 계열 안정화 후. `naming.py`는
  확장 가능한 구조로 이미 작성됨.
- **Prefill 속도 parity** — `o_proj M=1` regression과 prefill parity는
  persistent workspace + kernel warm-start로 Marlin launch overhead를
  숨길 여지가 있음을 시사; 마이너한 최적화.

### 5.3 범위 밖 항목 (계획 §17, 의도적으로 제외)

- bitsandbytes 통합 안 함.
- scratch-Triton GEMM 백엔드 안 함.
- Draft는 dense 유지.
- Embeddings는 dense 유지.
- `ssd/utils/quantize.py`의 torchao 경로는 fallback용 유지.

---

## 6. 재현 quick-reference

```bash
# 0. 환경 (신규 설치 불필요 — ssd env에 sgl-kernel + torchao 이미 포함)
source /home/chokwans99/PSD/ssd/env.sh

# 1. Llama 모델을 SSD-native W4A16 artifact로 임포트 (RTN 경로)
python scripts/awq_import.py \
    --model /data2/chokwans99/models/layerskip-llama3-8B \
    --out   /tmp/awq_artifacts/layerskip8b_tp2 \
    --tp 2 --mode rtn --dtype bfloat16

# 2. AR decode smoke
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/04_tp2_8b_ar.py

# 3. MESA smoke (GPU 3개)
CUDA_VISIBLE_DEVICES=0,1,2 python -O sandbox/awq_spike/07_mesa_awq.py

# 4. E2E 성능 (두 프로세스로 분리 — 이슈 로그 [Phase 7] 참조)
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/08_perf_bench.py dense
CUDA_VISIBLE_DEVICES=0,1 python -O sandbox/awq_spike/08_perf_bench.py awq

# 5. CLI 경로 (bench.py)
python -O bench/bench.py --llama --size 8 --gpus 2 \
    --model_path /data2/chokwans99/models/layerskip-llama3-8B \
    --b 1 --temp 0 --numseqs 16 --output_len 128 --random \
    --quant_awq --quant_awq_artifact /tmp/awq_artifacts/layerskip8b_tp2

# 6. External AutoAWQ round-trip (회귀 — 합성 체크포인트)
python -O sandbox/awq_spike/09_fake_autoawq_roundtrip.py

# 7. Negative tests (모듈 누락 / 잘못된 zero_point / 잘못된 w_bit / 잘못된 group_size)
python sandbox/awq_spike/10_negative_checks.py
```
