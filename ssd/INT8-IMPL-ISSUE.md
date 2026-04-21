# INT8 Weight-Only 구현 이슈 로그

브랜치: `feature/int8-weight-only`
계획 문서: `INT8-WEIGHT-ONLY-PLAN.md`

구현 중 발생하는 이슈, 결정, 해결 과정을 시간순으로 기록한다.
자체 해결된 항목은 [resolved], 사용자 판단 필요한 항목은 [open]으로 표기.

---

## 환경

- `torch`: 2.8.0+cu128
- `torchao`: **0.12.0** (0.17.0은 torch 2.8 cpp extension 비호환으로 경고 발생 → 0.12로 downgrade)
- `cuda`: 12.8
- `cudnn`: 91002
- GPU: RTX 3090 × 8 (24 GB each)
- conda env: `/home/chokwans99/anaconda3/envs/ssd`

---

## Phase 0

### [setup] torchao 0.17 → 0.12 downgrade
- torchao 0.17.0은 `torch>=2.11.0`에서만 cpp extension 로드. torch 2.8 환경에선 "Skipping import of cpp extensions" 경고 + 일부 최적 kernel 경로 비활성
- 0.12.0으로 내려서 해결. Int8WeightOnlyConfig, quantize_ 모두 정상 import 확인

### [resolved] Phase 0 모든 체크 통과 (sandbox/int8_spike/)

**01_dispatch_and_shapes.py** (checks 1-3):
- `quantize_()`는 `nn.Linear`만 변환 (SSDLikeLinear 미변환 확인)
- 저장 계약 (A): `dummy.weight → self.weight` 재할당 성공, `F.linear` 정상 dispatch
- 저장 계약 (B): `del _parameters["weight"]` + plain attribute 할당도 성공
- SSD-sized shards (qkv_packed 6144×4096, gate_up 11008×4096, o_proj row, down_proj row) 전부 성공
- cosine similarity 전체 ≥ 0.9999, dense 대비 max abs diff 작음

**02_graph_and_compile.py** (checks 4, 8):
- CUDA graph capture + 10회 replay: max diff 0.0 ✓
- `@torch.inference_mode()` 하에서 dispatch 정상, diff 0.0 ✓
- `@torch.compile` 모듈 + AQT linear 같은 forward 공존 정상 ✓
- **graph capture under inference_mode** (SSD 실제 패턴): diff 0.0 ✓
- 즉 가장 큰 리스크였던 graph + tensor subclass dispatch 완전 해결

**03_scale_tying_loader.py** (checks 5, 6, 7):
- scale shape: `(out_f,)` → per-output-channel (dim=0) 확정
- block_size `(1, 128)` — 각 output row 안에서 input axis를 128 chunk로 grouped (Int8WeightOnly default)
- **Column shard**: local scale = global scale의 row prefix (max diff 0) → TP 안전
- **Row shard**: local scale ≤ global scale 모든 row (mean ratio 0.9311) → Review B 실증: local quantize가 finer-grained이며 수치 정밀도 유리
- **tying 관찰**: `quantize_(lm_head)`는 attribute 교체 방식이라 원본 float 저장소를 수정하지 않음 → tied embed는 그대로 float 유지, F.embedding 정상. 단 메모리 공유 효과는 잃음
- **loader 순서**: float `param.data.copy_()` → quantize 순서 정상. AQT 상태에서 재로드 시 `ValueError: Not supported args for copy_` (loud failure, 안전)

### Phase 0 결정

- 저장 계약: **(A)** 채택 — `self.weight = dummy.weight`로 Param(AQT) 재할당. Forward 코드 변경 없음
- 양자화 API: **dummy `nn.Linear` + `quantize_(Int8WeightOnlyConfig())`** 경로 (내부 API 직접 호출 안 함)
- 교체 시점: float load 후, warmup 전
- tying 방어: 방어 분기를 두되 naive 케이스도 torchao가 자동 안전 (embed data ptr 유지)

---

## Phase 2

### [done] 통합 구현 완료

- `ssd/config.py`: `target_quant_enabled`, `target_quant_lm_head`, `target_quant_backend`, `target_quant_mode` 플래그 추가 (모두 기본 off)
- `ssd/utils/quantize.py`: `apply_int8_weight_only_to_target()` hook — Phase 0 계약 (A) 그대로 구현
  - dummy `nn.Linear` 생성 → `quantize_(Int8WeightOnlyConfig())` → `mod.weight = dummy.weight`
  - `tie_word_embeddings` 방어 (untie)
  - `target_quant_lm_head` flag
  - `SSD_INT8_SKIP` 환경변수로 module name substring 기반 exclude 가능
- `ssd/engine/model_runner.py`: `load_model(...)` 직후, `warmup_model()` 직전에 hook 삽입 (§3.5). `is_draft` 검사로 draft 완전 제외
- `bench/bench.py`: `--quant_int8`, `--no_quant_lm_head` 플래그 추가

### [done] Dense flag-off 회귀 테스트 통과

- layerskip-llama2-7B TP=2 + TinyLlama draft, async spec K=7 geo
- 기존 (pre-quant): TP=52-53 → 변경 후 flag off: TP=59.09
- 유사 범위, run-to-run 변동 내. 기존 MESA/SSD 경로 변경 없음 확인.

### [done] INT8 path 동작하는 케이스

| config | 결과 | 비고 |
|---|---|---|
| AR (target only) + int8, layerskip-7B | TP=30.44 | 5 seqs × 48 tok OK |
| AR (target only) + int8, **codellama-34B** TP=4 | TP=9.43 | 16.7 GB → 8.4 GB per rank |
| Greedy spec (temp=0) + int8, 7B | TP=12.18 | 동작하지만 accept rate 낮아질 수 있음 |
| Greedy spec (temp=0) + int8, 34B TP=4 | TP=8.53 | **accept=0.03** (dense 대비 급락) |

### [blocker] Sampling spec (temp>0) — inf/nan overflow

- 모델: layerskip-llama2-7B **및 codellama-34B** 모두 재현
- 증상: 처음 2 sequence는 정상 생성, 3번째 prefill에서 layer 1 hidden output에 8 inf 발생 → layer 2 input_norm에서 NaN으로 전파 → softmax → multinomial `probability tensor contains inf/nan` assert
- 범위: `--eager` 모드에서도 동일 (CUDA graph 이슈 아님), `torch.compile` off도 동일 (compile 이슈 아님)
- 원인 진단: Llama-family 모델의 **outlier activation channel** 문제. 특정 채널에서 활성화가 매우 크게 나와서 `x @ dequant_w` 결과가 bf16 overflow → inf
  - `SSD_INT8_SKIP=down_proj` → inf 수 8→2로 감소 (down_proj가 주요 기여)
  - `SSD_INT8_SKIP=down_proj,o_proj` → 여전히 1 inf 잔존 (qkv/gate_up도 기여)
  - 즉 특정 layer만 제외로는 해결 불가. 전체 Llama 활성화가 int8 weight-only 단독으로는 취약
- 실제 prompt 데이터에서만 발생 (warmup의 random 데이터로는 재현 안 됨) — 진짜 outlier 채널을 실제 프롬프트가 자극하는 것

### [blocker] Greedy spec에서 accept rate 급락 (34B)

- 34B + greedy + int8: accept rate 0.03 (dense baseline ~0.38)
- int8 양자화로 logits이 충분히 교란되어 argmax가 draft와 자주 달라짐
- 즉 크래시는 안 나지만 spec speedup 거의 증발

### 결론 및 path forward (사용자 판단 필요)

**Phase 2 통합 자체는 성공**: 코드 변경, flag, hook 모두 정상 동작. Dense 경로 회귀 없음. AR 경로는 34B까지 확인.

**하지만 spec/MESA 사용 시나리오는 plain `Int8WeightOnlyConfig`로 해결 안 됨**. 둘 다 동일한 근본 원인 = **Llama outlier activation**이 int8 weight-only에서 수치적으로 깨짐.

가능한 후속 방향 (사용자 결정 필요):

1. **SmoothQuant** — pre-quantization에서 activation outlier를 weight로 diag scale matrix로 옮김. torchao에 구현 있음 (`torchao.quantization.Int8SmoothQuantInt8WeightConfig` 등). 추가 보정 스텝 필요
2. **Int8DynamicActivationInt8WeightConfig (A8W8)** — activation도 per-token으로 런타임 양자화. outlier tolerance 더 좋음. 계획 문서의 "weight-only" 원칙에서 벗어남
3. **FP8 weight-only** — H100 이상 전용. 우리 3090 환경에선 쓸 수 없음
4. **GPTQ/AWQ with outlier-aware calibration** — 계획 §3.3 "외부 quantized model importer 범용화 제외" 원칙에 걸림
5. **AR 전용 운영** — spec 포기. 70B는 AR로만 돌림. 이번 계획 동기(MESA+spec)와 상충
6. **Phase 3+ 진행 보류** — Phase 2에서 발견된 blocker 해결 없이는 뒤 Phase에서 동일 이슈 재현

### 진단 sandbox 로그

- `tmp/int8_smoke/dbg_full/` — 층별 inf 탐지 결과
- `tmp/int8_smoke/dbg_fine/` — layer 내부 sub-step NaN 전파 추적
- `tmp/int8_smoke/34b_ar_int8/` — 34B AR 정상 동작 확인
- `tmp/int8_smoke/34b_spec_int8*/` — 34B spec 실패 재현
