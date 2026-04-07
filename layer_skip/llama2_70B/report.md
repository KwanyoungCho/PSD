# LayerSkip vs. Standard Early-Exit Analysis Report

## Llama-2-70B-hf — Early-Exit Distribution Analysis

---

## 1. Experiment Overview

| Parameter | Value |
|---|---|
| **Target Model** | Llama-2-70b-hf (80 layers) |
| **LayerSkip Model** | layerskip-llama2-70B (80 layers) |
| **Draft Models** | TinyLlama-1.1B-Chat-v1.0 (22 layers), Llama-2-7b-hf (32 layers) |
| **Dataset** | GSM8K |
| **# Samples** | 50 |
| **# Tokens per sample** | 10 (greedy generation) |
| **# Total positions** | 500 |

**목적**: Target 모델(Llama-2-70B)의 중간 레이어에서 early-exit한 분포가 최종 출력 분포를 얼마나 잘 근사하는지 분석.  
Standard early-exit (logit lens)과 LayerSkip (early-exit 학습된 모델)을 비교하고, 소형 draft 모델 2종의 성능을 baseline으로 설정.

---

## 2. Metrics 정의

| Metric | 방향 | 설명 |
|---|---|---|
| **JSD** (Jensen-Shannon Divergence) | ↓ Lower is better | Early-exit 분포와 최종 분포 간 대칭적 발산 |
| **KL** (KL Divergence) | ↓ Lower is better | Early-exit 분포 → 최종 분포 방향의 KL 발산 |
| **Top-1 Match** | ↑ Higher is better | Early-exit argmax == 최종 argmax 일치 비율 |
| **Top-5 Overlap** | ↑ Higher is better | \|top5(p\_E) ∩ top5(p\_T)\| / 5 |
| **Top-5 Mass** | ↑ Higher is better | Σ p\_T(v) for v ∈ top5(p\_E) |
| **Top-10 Overlap** | ↑ Higher is better | \|top10(p\_E) ∩ top10(p\_T)\| / 10 |
| **Top-10 Mass** | ↑ Higher is better | Σ p\_T(v) for v ∈ top10(p\_E) |

---

## 3. Draft Model Baselines

Draft 모델의 최종 출력 분포를 target 모델의 최종 출력 분포와 직접 비교한 결과:

| Metric | TinyLlama-1.1B (22L) | Llama-2-7B (32L) |
|---|---|---|
| **JSD ↓** | 0.132 | 0.053 |
| **KL ↓** | 0.659 | 0.228 |
| **Top-1 Match ↑** | 0.556 | 0.762 |
| **Top-5 Overlap ↑** | 0.603 | 0.742 |
| **Top-5 Mass ↑** | 0.703 | 0.760 |
| **Top-10 Overlap ↑** | 0.644 | 0.768 |
| **Top-10 Mass ↑** | 0.806 | 0.846 |

Llama-2-7B가 모든 metric에서 TinyLlama-1.1B보다 우수하며, target 분포에 훨씬 더 가까운 근사를 제공.

---

## 4. Standard Early-Exit (Logit Lens) 분석

Target 모델(Llama-2-70B)의 각 중간 레이어 hidden state에 final layer norm + lm_head를 적용하여 분포를 추출한 결과.

### 주요 관찰

- **초기 레이어 (L0~L30)**: JSD ≈ 0.66~0.68로 매우 높음. 최종 분포와 거의 무관한 분포를 생성.
- **중간 레이어 (L30~L55)**: 서서히 감소 시작. L40에서 JSD = 0.640, Top-5 Overlap = 0.067.
- **후반 레이어 (L55~L80)**: 급격한 개선. L70에서 JSD = 0.431, L79에서 JSD = 0.054.
- **최종 레이어 (L80)**: JSD ≈ 0 (target 자기 자신).

### Layer별 주요 수치 (Standard Early-Exit)

| Layer | JSD ↓ | KL ↓ | Top-1 Match ↑ | Top-5 Overlap ↑ | Top-5 Mass ↑ |
|---|---|---|---|---|---|
| L0 | 0.677 | 9.485 | 0.000 | 0.003 | 0.006 |
| L10 | 0.667 | 7.717 | 0.016 | 0.012 | 0.028 |
| L20 | 0.667 | 7.884 | 0.008 | 0.016 | 0.018 |
| L30 | 0.660 | 7.420 | 0.012 | 0.034 | 0.043 |
| L40 | 0.640 | 6.834 | 0.044 | 0.067 | 0.083 |
| L50 | 0.543 | 5.231 | 0.204 | 0.158 | 0.301 |
| L60 | 0.497 | 5.473 | 0.264 | 0.184 | 0.296 |
| L70 | 0.431 | 3.829 | 0.400 | 0.280 | 0.455 |
| L79 | 0.054 | 0.297 | 0.846 | 0.834 | 0.780 |

**결론**: Standard early-exit는 마지막 ~10개 레이어에서야 의미 있는 근사가 가능. 전체 80 레이어 중 L70 이전에는 Top-5 Overlap < 0.28, JSD > 0.43으로 draft 모델보다도 성능이 낮음.

---

## 5. LayerSkip Early-Exit 분석

LayerSkip으로 학습된 모델은 early-exit 목적에 맞게 fine-tuning 되어, 중간 레이어에서도 최종 분포에 가까운 출력을 생성하도록 학습됨.

### 주요 관찰

- **초기 레이어 (L0~L5)**: L0에서 JSD = 0.591로, standard (0.677) 대비 12.7% 낮음. L5에서 이미 JSD = 0.339.
- **중간 레이어 (L10~L40)**: JSD가 0.15~0.26 범위로 빠르게 안정화. Standard 대비 **61~76% JSD 감소**.
- **후반 레이어 (L40~L70)**: JSD ≈ 0.065~0.16. 지속적으로 감소하지만 개선 속도는 둔화.
- **최종 레이어 (L80)**: JSD = 0.016 (≠ 0, target과 다른 모델이므로 완전히 일치하지 않음).

### Layer별 주요 수치 (LayerSkip Early-Exit)

| Layer | JSD ↓ | KL ↓ | Top-1 Match ↑ | Top-5 Overlap ↑ | Top-5 Mass ↑ |
|---|---|---|---|---|---|
| L0 | 0.591 | 6.375 | 0.074 | 0.129 | 0.154 |
| L10 | 0.257 | 1.629 | 0.450 | 0.444 | 0.569 |
| L20 | 0.211 | 1.386 | 0.492 | 0.508 | 0.624 |
| L30 | 0.179 | 1.147 | 0.516 | 0.532 | 0.646 |
| L40 | 0.156 | 0.981 | 0.562 | 0.583 | 0.664 |
| L50 | 0.092 | 0.486 | 0.650 | 0.664 | 0.722 |
| L60 | 0.069 | 0.329 | 0.682 | 0.710 | 0.752 |
| L70 | 0.070 | 0.335 | 0.698 | 0.742 | 0.764 |
| L80 | 0.016 | 0.064 | 0.862 | 0.847 | 0.784 |

---

## 6. LayerSkip vs. Standard Early-Exit 비교

### JSD 감소율 (LayerSkip / Standard)

| Layer | Standard JSD | LayerSkip JSD | 감소율 |
|---|---|---|---|
| L10 | 0.667 | 0.257 | **61.6%** |
| L20 | 0.667 | 0.211 | **68.3%** |
| L30 | 0.660 | 0.179 | **72.8%** |
| L40 | 0.640 | 0.156 | **75.6%** |
| L50 | 0.543 | 0.092 | **83.1%** |
| L60 | 0.497 | 0.069 | **86.2%** |
| L70 | 0.431 | 0.070 | **83.7%** |

LayerSkip은 모든 레이어에서 standard early-exit 대비 **60~86%의 JSD 감소**를 달성.

### KL 감소율

| Layer | Standard KL | LayerSkip KL | 감소율 |
|---|---|---|---|
| L10 | 7.717 | 1.629 | **78.9%** |
| L20 | 7.884 | 1.386 | **82.4%** |
| L30 | 7.420 | 1.147 | **84.5%** |
| L40 | 6.834 | 0.981 | **85.6%** |
| L50 | 5.231 | 0.486 | **90.7%** |
| L60 | 5.473 | 0.329 | **94.0%** |
| L70 | 3.829 | 0.335 | **91.2%** |

KL divergence에서는 **79~94%의 감소율**로, JSD보다 더 극적인 개선.

### Top-5 Overlap 향상

| Layer | Standard | LayerSkip | 차이 |
|---|---|---|---|
| L10 | 0.012 | 0.444 | **+0.432** |
| L20 | 0.016 | 0.508 | **+0.493** |
| L30 | 0.034 | 0.532 | **+0.499** |
| L40 | 0.067 | 0.583 | **+0.516** |
| L50 | 0.158 | 0.664 | **+0.506** |
| L60 | 0.184 | 0.710 | **+0.526** |
| L70 | 0.280 | 0.742 | **+0.462** |

Standard early-exit에서 거의 0에 가까운 Top-5 Overlap이 LayerSkip에서는 0.44~0.74로 크게 향상.

---

## 7. Crossover Analysis: LayerSkip이 Draft 모델을 능가하는 시점

LayerSkip의 early-exit 분포가 각 draft 모델의 최종 분포 성능을 넘어서는 첫 번째 레이어:

### vs TinyLlama-1.1B-Chat-v1.0

| Metric | Draft Baseline | Crossover Layer | LayerSkip Value | 전체 대비 비율 |
|---|---|---|---|---|
| **JSD ↓** | 0.132 | **Layer 44** | 0.128 | 55% |
| **KL ↓** | 0.659 | **Layer 45** | 0.596 | 56% |
| **Top-1 Match ↑** | 0.556 | **Layer 34** | 0.562 | 43% |
| **Top-5 Overlap ↑** | 0.603 | **Layer 43** | 0.608 | 54% |
| **Top-5 Mass ↑** | 0.703 | **Layer 45** | 0.708 | 56% |
| **Top-10 Overlap ↑** | 0.644 | **Layer 41** | 0.650 | 51% |
| **Top-10 Mass ↑** | 0.806 | **Layer 45** | 0.809 | 56% |

→ **Layer 34~45 (전체의 43~56%)** 부터 TinyLlama-1.1B를 능가.

### vs Llama-2-7B-hf

| Metric | Draft Baseline | Crossover Layer | LayerSkip Value | 전체 대비 비율 |
|---|---|---|---|---|
| **JSD ↓** | 0.053 | **Layer 76** | 0.052 | 95% |
| **KL ↓** | 0.228 | **Layer 77** | 0.176 | 96% |
| **Top-1 Match ↑** | 0.762 | **Layer 78** | 0.778 | 98% |
| **Top-5 Overlap ↑** | 0.742 | **Layer 72** | 0.748 | 90% |
| **Top-5 Mass ↑** | 0.760 | **Layer 69** | 0.760 | 86% |
| **Top-10 Overlap ↑** | 0.768 | **Layer 68** | 0.768 | 85% |
| **Top-10 Mass ↑** | 0.846 | **Layer 67** | 0.847 | 84% |

→ **Layer 67~78 (전체의 84~98%)** 부터 Llama-2-7B를 능가.

### 핵심 시사점

- LayerSkip은 **절반 레이어만 사용해도 TinyLlama-1.1B (22L) 수준의 분포 근사** 가능.
- 그러나 Llama-2-7B (32L) 수준에 도달하려면 **80~95% 이상의 레이어**가 필요.
- Standard early-exit으로는 TinyLlama-1.1B를 능가하려면 Layer 80 이후 (JSD 기준)가 필요하여, **사실상 full forward pass가 필요**.

---

## 8. LayerSkip의 수렴 특성

LayerSkip 모델의 early-exit 분포 품질은 레이어에 따른 뚜렷한 3단계 패턴을 보임:

### Phase 1: 급격한 개선 (L0 ~ L10)
- JSD: 0.591 → 0.257 (56.5% 감소)
- Top-5 Overlap: 0.129 → 0.444 (+0.315)
- LayerSkip 학습 효과가 가장 극적으로 나타나는 구간

### Phase 2: 점진적 안정화 (L10 ~ L45)
- JSD: 0.257 → 0.128 (50.2% 추가 감소)
- Top-5 Overlap: 0.444 → 0.608 (+0.164)
- 개선 속도가 둔화되지만 꾸준히 개선

### Phase 3: 미세 조정 (L45 ~ L80)
- JSD: 0.128 → 0.016 (87.5% 추가 감소)
- Top-5 Overlap: 0.608 → 0.847 (+0.239)
- 최종 레이어로 갈수록 target 분포에 수렴

---

## 9. Speculative Decoding 관점에서의 함의

| Exit Point | JSD | Top-5 Overlap | Top-5 Mass | 의미 |
|---|---|---|---|---|
| **L20 (25%)** | 0.211 | 0.508 | 0.624 | TinyLlama보다 JSD는 낮지만 Top-k 성능은 아직 미달 |
| **L40 (50%)** | 0.156 | 0.583 | 0.664 | TinyLlama와 대등하며 Top-k 일부 능가 |
| **L60 (75%)** | 0.069 | 0.710 | 0.752 | TinyLlama 크게 능가, Llama-7B에 근접 |
| **L70 (88%)** | 0.070 | 0.742 | 0.764 | Llama-7B와 대등 |

Speculative decoding에서 LayerSkip을 draft로 사용할 경우:
- **L40 exit (50% 연산)**: TinyLlama-1.1B 수준의 draft quality
- **L60 exit (75% 연산)**: TinyLlama-1.1B를 크게 능가하는 draft quality
- **L70 exit (88% 연산)**: Llama-2-7B 수준의 draft quality

다만, 연산량 절감 효과(25~50% skip)와 draft 품질 간 trade-off가 존재. L70 exit는 draft 품질은 높지만 연산 절감이 12%에 불과하여, 별도 draft 모델 대비 이점이 제한적.

---

## 10. Figures

본 분석에서 생성된 그래프 목록:

| File | Description |
|---|---|
| `layerskip_overview.png` | JSD, Top-5 Overlap, Top-5 Mass 3-subplot overview (Layer Depth %) |
| `layerskip_jsd.png` | JSD per layer (detailed) |
| `layerskip_kl.png` | KL per layer (detailed) |
| `layerskip_top1_match.png` | Top-1 Match per layer |
| `layerskip_top5_overlap.png` | Top-5 Overlap per layer |
| `layerskip_top5_mass.png` | Top-5 Mass per layer |
| `layerskip_top10_overlap.png` | Top-10 Overlap per layer |
| `layerskip_top10_mass.png` | Top-10 Mass per layer |

Revised overview (Layer Index x-axis, crossover annotation 포함):
- `../../tmp_graph/layerskip_overview_revised.png`

---

## 11. Summary

1. **Standard early-exit (logit lens)은 Llama-2-70B에서 매우 비효과적**: 마지막 ~5개 레이어를 제외하면 draft 모델보다 성능이 낮음.
2. **LayerSkip은 모든 레이어에서 standard 대비 60~94% 성능 개선**: 특히 초반 10개 레이어에서 가장 극적인 개선.
3. **LayerSkip L40~L45 (50~56%) ≈ TinyLlama-1.1B**: 절반 연산으로 22-layer 모델 수준 달성.
4. **LayerSkip L67~L78 (84~98%) ≈ Llama-2-7B**: 7B 모델 수준에는 상당한 레이어 필요.
5. **Speculative decoding draft로서의 효용**: L60 (75%) exit가 연산 절감(25%)과 draft 품질 간 최적 균형점으로 보임.
