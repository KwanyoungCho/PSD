# AWQ W4A16 양자화 실험 결과

`docs/quantization/03-final-report.md` 가 인용하는 raw 실험 결과물을
영구 보존한다. 원본은 `tmp/final_exp2_quant{,_70b}/` 에 있었으나
`tmp*/` gitignore 로 잡혀서 main 에 들어가지 않았기 때문에 이리로 이동.

## 디렉토리

```
34b/   layerskip-codellama-34B (TP=4 AWQ) + Llama-3.2-1B draft
       ├── ar/, baseline_k7_*/, mesa_k5_f4_dfo2_exit{24,28,32}/
       └── draft_awq/  ← target AWQ + draft AWQ (1B AWQ) 비교

70b/   layerskip-llama2-70B (TP=4 AWQ) + TinyLlama-1.1B draft
       ├── ar/, baseline_k7_*/, mesa_k5_f4_dfo2_exit{40,47,53}/
       └── draft_awq/  ← target AWQ + draft AWQ (TinyLlama AWQ) 비교
```

## 각 config 폴더 안

| 파일 | 의미 |
|---|---|
| `run.log` | bench.py stdout — METRICS (TP, accept, CH, TS), reproduction command 1 줄 |
| `mesa_breakdown_summary.csv` | phase 별 mean / median / p95 / n |
| `mesa_per_step_contribution.csv` | phase 별 step 당 ms 기여 (report 표의 출처) |
| `mesa_breakdown.png` | phase latency bar |
| `mesa_breakdown_over_time.png` | step 진행에 따른 phase 변화 |
| `mesa_timeline_step*.png` | step 의 timeline |

## 제외된 것 — `mesa_profile_*.json`

`SSD_PROFILE_MESA=1` 일 때 dump 되는 raw CUDA-event log (config 당 ~30MB,
전체 ~1GB). CSV 와 PNG 가 여기서 파생되므로 결과 확인엔 불필요. 다른
통계로 re-aggregate 하거나 다른 step 의 timeline 을 그리려면 각 폴더의
`run.log` 첫 줄 reproduction command 로 재생성 가능.

`.gitignore` 에 `experiments/**/mesa_profile_*.json` rule 이 있어서,
향후 실험을 이 디렉토리에서 돌려도 raw JSON 은 자동 제외됨.

## Plot 재생성

```bash
# Per-config breakdown + timeline
for d in experiments/quant_awq/{34b,70b}/{baseline_*,mesa_*}; do
    python bench/plot_mesa_breakdown.py "$d"
    python bench/plot_mesa_timeline.py "$d"
done

# 70B 전용 cross-config compare
python bench/plot_compare_breakdown_70b.py experiments/quant_awq/70b

# dense draft vs AWQ draft (70B)
python bench/plot_compare_dense_vs_awq_draft_70b.py experiments/quant_awq/70b
```

(스크립트들은 `tmp/...` 절대경로를 default 로 가짐. CLI 인자로 새 경로 전달.)
