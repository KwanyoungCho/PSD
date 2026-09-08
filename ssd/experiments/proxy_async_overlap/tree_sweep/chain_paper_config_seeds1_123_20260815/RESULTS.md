# DUET-chain paper-config seed repeats

논문용 DUET-chain seed 42와 동일한 설정에서 sampler seed만 1과 123으로
변경해 full Spec-Bench를 반복했다. Raw 실행은 context 4,096 및 analytic
draft-RoPE extension을 사용하고, 비교에는 prompt 길이로 사전에 정한 동일한
2,048-token-safe subset을 사용한다.

## Configuration

- Engine: `/home/eslab/chokwans99/PSD/ssd`
- Target: `facebook/layerskip-llama2-70B`, TP=2
- Draft: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, 별도 GPU 1장
- Physical GPUs: 5,6,7
- Exit layer 56, `K1/K2=8/4`
- P1 fan-out 3, P2 budget 15, proxy top-k 28
- `N1/M1=14/12`, `N2/M2=8/8`, `C_tensor=2`
- P1/P2 tree off; P1 allocation option `backbone`은 비활성 tree용 값이라
  chain 동작에는 영향을 주지 않는다.
- Temperature 0.7, top-p 1.0, raw prompt, output cap 1,024
- Raw context 4,096, draft RoPE extension on
- Seeds: 1, 123; 기존 paper seed: 42
- Full input: 480 questions/560 turns
- Context-safe output: 456 questions/536 turns

## Overall results on the common 2,048-token-safe subset

P1/P2 AL은 correction/recovery token을 포함한다. TPS와 AL/hit은 MT-Bench의
두 turn을 먼저 question 단위로 결합한 뒤 question 평균으로 집계했다. Target
latency는 verification-step weighted mean이다.

| Seed | Decode TPS | AL | Hit | P1 hit | P1 AL | P2 hit | P2 AL | Target step ms | Verify ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 (paper) | 67.274 | 4.393 | 0.814 | 0.658 | 5.162 | 0.156 | 2.999 | 66.092 | 61.608 |
| 1 | 67.667 | 4.398 | 0.819 | 0.668 | 5.106 | 0.151 | 3.004 | 65.867 | 61.128 |
| 123 | 67.187 | 4.386 | 0.824 | 0.662 | 5.099 | 0.162 | 3.016 | 66.142 | 61.270 |
| 3-seed mean | 67.376 | 4.392 | 0.819 | 0.662 | 5.122 | 0.157 | 3.006 | 66.034 | 61.335 |

세 seed 범위는 TPS 67.187–67.667, AL 4.386–4.398, hit 0.814–0.824다.
기존 seed 42가 모두 이 범위 안에 있으며 동일 설정의 chain 성능이 안정적으로
재현됐다.

## Artifacts

- 실행 스크립트: `../run_full_matched_chain_seeds1_123_20260815.sh`
- Seed 1 raw: `duet_chain_papercfg_s1_o1024_ctx4096.jsonl`
- Seed 1 safe: `duet_chain_papercfg_s1_o1024_ctx4096_ctx2048_safe.jsonl`
- Seed 123 raw: `duet_chain_papercfg_s123_o1024_ctx4096.jsonl`
- Seed 123 safe: `duet_chain_papercfg_s123_o1024_ctx4096_ctx2048_safe.jsonl`
- 통합 question-level 집계: `comparison_seed42_1_123_ctx2048_safe.json`
- 두 신규 seed 집계: `summary_ctx2048_safe.json`
