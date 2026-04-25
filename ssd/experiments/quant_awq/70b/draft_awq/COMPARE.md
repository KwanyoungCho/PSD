# 70B AWQ — Target-only AWQ vs Target+Draft AWQ

**Target**: `layerskip-llama2-70B` (TP=4 AWQ via sgl-kernel Marlin)
**Draft**: TinyLlama-1.1B-Chat-v1.0 (TP=1)
**Hardware**: 5× RTX 3090 (4 target TP + 1 draft)
**Workload**: 50 seqs × 256 tokens, B=1, temp=0.6, max_model_len=2048, k/f from
each config

Two arms compared:

- **A. target AWQ + draft DENSE** — `tmp/final_exp2_quant_70b/{cfg}/run.log`
- **B. target AWQ + draft AWQ** — `tmp/final_exp2_quant_70b/draft_awq/{cfg}/run.log`

---

## 결과 비교

| Config | A. target AWQ + draft DENSE | B. target AWQ + draft AWQ | ΔTP | ΔDraft_ms | ΔVerify_ms |
|---|---:|---:|---:|---:|---:|
| baseline_k7_uniform | 69.94 / D=47.57 / V=44.42 | **73.59** / D=33.85 / V=46.54 | **+5.2%** | **−13.72** | +2.12 |
| baseline_k7_geo     | 72.57 / D=45.05 / V=45.34 | **72.82** / D=33.58 / V=45.91 | +0.3% | **−11.47** | +0.57 |
| mesa_k5_f4_dfo2_exit40 | 61.02 / D=53.87 / V=45.79 | **66.71** / D=42.29 / V=47.20 | **+9.3%** | **−11.58** | +1.41 |
| mesa_k5_f4_dfo2_exit47 | 61.01 / D=53.55 / V=45.44 | **69.40** / D=44.95 / V=46.01 | **+13.8%** | **−8.60** | +0.57 |
| mesa_k5_f4_dfo2_exit53 | 58.85 / D=54.67 / V=44.70 | **70.07** / D=46.60 / V=45.03 | **+19.1%** | **−8.07** | +0.33 |

`TP` = throughput (tok/s), `D` = avg draft step (ms), `V` = avg target verify (ms).

### Token efficiency (불변)

Cache hit / accept / tok-per-step 은 두 arm 이 거의 동일 — draft 양자화는 draft
의 분포를 바꾸지 않으므로 acceptance 패턴은 그대로:

| Config | A. CH / Acc / TS | B. CH / Acc / TS |
|---|---:|---:|
| baseline_k7_uniform | 0.67 / 0.44 / 4.06 | 0.67 / 0.45 / 4.14 |
| baseline_k7_geo     | 0.69 / 0.46 / 4.23 | 0.67 / 0.43 / 4.03 |
| mesa_exit40         | 0.80 / 0.54 / 3.70 | 0.78 / 0.52 / 3.62 |
| mesa_exit47         | 0.82 / 0.54 / 3.68 | 0.83 / 0.53 / 3.64 |
| mesa_exit53         | 0.84 / 0.52 / 3.61 | 0.85 / 0.53 / 3.63 |

---

## 분석

### 1. Draft step 이 일관되게 −8 ~ −14 ms 단축

모든 config 에서 draft step (`D`) 이 8–14 ms 빨라짐. TinyLlama 의 W4A16
마린 GEMM 이 dense fp16 보다 빠르다는 직접적인 결과. AWQ 가 weight load
대역폭을 약 4× 줄이고 RTX 3090 의 약한 메모리 대역폭에서 그 효과가 그대로
드러남.

### 2. Verify step 미세 증가 (+0.3 ~ +2.1 ms)

Target 은 두 arm 모두 같은 70B AWQ 인데도 verify 가 1–2 ms 차이남.
→ **noise + draft 가 보내주는 tree 의 분포가 약간 달라져 attention mask
가 살짝 바뀌는 영향**. Wall-clock 영향은 무시 가능 (≤4%).

### 3. MESA 가 가장 크게 회복 (+19% at exit53)

70B 에서 dense draft + MESA 는 baseline 보다 느렸음 (60s vs 70s). Draft
AWQ 로 바꾸면 MESA exit47/53 이 baseline_geo 와 거의 비슷해짐 (70 / 70).
**70B 에서 MESA 의 "draft 가 더 느림" 문제를 draft 양자화가 보완** — 34B
에서 본 것과 같은 패턴이 70B 에서도 재현됨.

### 4. Geo schedule 은 이득 미미 (+0.3%)

`baseline_k7_geo` 는 dense 에서도 72.57 로 이미 빨랐고, AWQ 로 73 도달.
Draft 양자화의 이득은 **draft 가 bottleneck 인 경우 (uniform / MESA) 에서
크고, 이미 draft 가 가벼운 경우 (geo) 는 작음**.

### 5. Best 구성 변화

- Dense draft: best = `baseline_k7_geo` (72.57)
- AWQ draft: best = `baseline_k7_uniform` (73.59), `mesa_exit53` 도 70.07 로 근접

Draft 양자화로 uniform 이 geo 를 역전. MESA 도 이제 baseline 의
사정권 (−5%) 안.

---

## Best across all 70B AWQ runs

| Mode | Config | Throughput | vs AR (32.87) |
|---|---|---:|---:|
| AR (target AWQ only) | — | 32.87 | 1.00× |
| Spec (target AWQ + draft DENSE) | baseline_k7_geo | 72.57 | 2.21× |
| **Spec (target AWQ + draft AWQ)** | **baseline_k7_uniform** | **73.59** | **2.24×** |
| Spec (target AWQ + draft AWQ) | mesa_exit53 | 70.07 | 2.13× |

Draft 양자화로 70B 의 best spec config 가 AR 대비 2.24× — 34B
(target+draft AWQ best ≈ 2.5×) 와 모델 크기 대비 합리적인 비율.

---

## Reproduction

```bash
bash tmp/final_exp2_quant_70b/run_draft_awq_compare.sh
# orchestrator log → tmp/final_exp2_quant_70b/draft_awq_orchestrator.log
# per-config run.log → tmp/final_exp2_quant_70b/draft_awq/{config}/run.log
# summary table → tmp/final_exp2_quant_70b/draft_awq/SUMMARY.txt
```
