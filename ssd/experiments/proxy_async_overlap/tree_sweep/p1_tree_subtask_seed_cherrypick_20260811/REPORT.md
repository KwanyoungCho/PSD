# DUET-tree subtask seed selection

Date: 2026-08-12

## Purpose

DUET-tree의 Translation, Summarization, Math 결과가 sampling seed에 따라 얼마나
변하는지 확인하고, seed 42/123/1 중 subtask별 decode TPS가 가장 높은 cell을
선택한 결과다. 이 결과는 의도적인 cherry-pick이며 단일 seed 실험이 아니다.

## Measurement protocol

- Metric: prompt-mean decode-only TPS and prompt-mean accepted length
- Target/draft: LayerSkip Llama-2-70B / TinyLlama-1.1B
- Batch 1, temperature 0.7, top-p 1.0, maximum output 1,024
- P1 tree: K1=8, C=2, N/M=14/12, full-root `backbone`
- P2 tree: K2=4, budget=15, N/M=8/8
- Exit layer 56, proxy top-k 28, context 4,096, draft RoPE extension
- Profiling/debug off
- Dataset order: full Spec-Bench의 MT-Bench부터 Math까지 480-request prefix

요청한 세 task만 잘라 바로 실행하면 앞선 prompt가 소비하는 RNG stream이 달라진다.
따라서 seed 42 full run과 같은 위치에서 각 task가 시작되도록 원래 순서의 480개
prefix를 seed 123과 seed 1에서 각각 실행했다.

## Seed comparison

### Translation

| Seed | Decode TPS | Accepted length | Hit | P1 hit | P2 hit |
|---:|---:|---:|---:|---:|---:|
| 42 | 66.301 | 4.643 | 0.781 | 0.633 | 0.148 |
| **123** | **69.550** | **4.860** | **0.815** | **0.663** | 0.152 |
| 1 | 67.808 | 4.725 | 0.796 | 0.640 | **0.156** |

### Summarization

| Seed | Decode TPS | Accepted length | Hit | P1 hit | P2 hit |
|---:|---:|---:|---:|---:|---:|
| 42 | 54.207 | 3.836 | **0.672** | 0.483 | **0.189** |
| **123** | **56.312** | **3.954** | 0.672 | **0.492** | 0.180 |
| 1 | 53.352 | 3.768 | 0.644 | 0.486 | 0.158 |

Seed 123의 total hit은 0.671752로 seed 42의 0.671928보다 0.000176 낮다.
반면 decode TPS, accepted length, P1 hit은 모두 높다.

### Math

| Seed | Decode TPS | Accepted length | Hit | P1 hit | P2 hit |
|---:|---:|---:|---:|---:|---:|
| 42 | 62.146 | 4.327 | **0.842** | 0.622 | **0.221** |
| 123 | 61.370 | 4.255 | 0.812 | 0.607 | 0.204 |
| **1** | **63.790** | **4.462** | 0.828 | **0.629** | 0.199 |

Seed 1은 TPS와 accepted length가 가장 높지만 total/P2 hit은 seed 42보다 낮다.

## Selected composite

선택 기준은 subtask별 decode TPS 최대이며, 동률일 때 accepted length를 본다.

| Subtask | Selected seed | Decode TPS | Accepted length | Hit |
|---|---:|---:|---:|---:|
| MT-Bench | 42 | 60.351 | 4.209 | 0.804 |
| Translation | 123 | 69.550 | 4.860 | 0.815 |
| Summarization | 123 | 56.312 | 3.954 | 0.672 |
| QA | 42 | 71.064 | 4.978 | 0.850 |
| Math | 1 | 63.790 | 4.462 | 0.828 |
| RAG | 42 | 66.400 | 4.712 | 0.771 |

| Result | Decode TPS | Accepted length | Hit | P1 hit | P2 hit | Target step |
|---|---:|---:|---:|---:|---:|---:|
| Original seed 42 | 62.974 | 4.416 | 0.789 | 0.609 | 0.180 | 71.03 ms |
| Seed-selected composite | **63.974** | **4.484** | **0.792** | **0.616** | 0.176 | 71.02 ms |
| Difference | **+1.000** | **+0.067** | **+0.003** | **+0.007** | -0.004 | -0.01 ms |

Selection으로 TPS가 정확히 1.000 tok/s 증가했지만, P2 hit은 낮아졌다. 또한 이
cherry-pick으로도 DUET-tree가 요청한 세 subtask 모두에서 전체 방법 중 1위가 되지는
않는다.

- Translation: DUET-tree 69.55 < DUET-P2-tree 70.99 < SSD 76.95
- Summarization: DUET-tree 56.31 < SpecInfer 57.59
- Math: DUET-tree 63.79 < DUET-chain 65.18 < SpecInfer 65.31

## Integrity and artifacts

- Seed 123: 480 rows, 480 unique UIDs, exit 0, runtime errors 0
- Seed 1: 480 rows, 480 unique UIDs, exit 0, runtime errors 0
- Composite: 560 rows and 560 unique UIDs
- Composite seed counts: seed 42=320, seed 123=160, seed 1=80

Artifacts:

- `specbench_through_math_480.jsonl`: order-preserving evaluation prefix
- `seed123/p1_backbone_s123_o1024.jsonl`
- `seed1/p1_backbone_s1_o1024.jsonl`
- `duet_tree_cherrypick_s42_s123_s1_o1024.jsonl`

## Scientific-use note

이 결과는 seed 평균도 아니고 단일-seed result도 아니다. 논문에 사용한다면 반드시
`best per-subtask over seeds {42, 123, 1}`이라고 명시해야 한다. 일반적인 최종
성능 주장에는 single-seed full result 또는 여러 seed의 평균과 run-to-run CI를
사용하는 편이 타당하다.
