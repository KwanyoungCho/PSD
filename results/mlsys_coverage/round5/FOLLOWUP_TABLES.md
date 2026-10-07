# Supplementary comparisons

All full480 measured last passes. Transfer settings are not independently tuned B2/B4/long optima. Optimization ratios compare identical parameters, seeds and GPU IDs; both passes must preserve token outputs and proposal/acceptance events.

| Model | Test | B | DUET AL* | SSD AL* | TPS* DUET/SSD |
|---|---|---:|---:|---:|---:|
| llama2 | batch_transfer | 2 | 2.1492 | 2.1527 | 1.012x |
| llama2 | batch_transfer | 4 | 2.1078 | 2.1231 | 0.954x |
| llama2 | long_transfer | 8 | 2.3588 | 2.4076 | 0.880x |
| llama3 | batch_transfer | 2 | 2.3892 | 2.6753 | 0.941x |
| llama3 | batch_transfer | 4 | 2.4109 | 2.6955 | 0.849x |
| llama3 | long_transfer | 8 | 2.5814 | 2.9209 | 0.696x |

| Model | Optimization | B | Role/repeat | TPS* on/off | Raw returned TPS on/off |
|---|---|---:|---|---:|---:|
| llama2 | fused+bulk | 8 | fast/0 | 1.014x | 1.014x |
| llama3 | fused+bulk | 8 | fast/0 | 1.038x | 1.037x |
| llama2 | ladder trim | 1 | duet_fast/0 | 1.004x | 1.004x |
| llama2 | ladder trim | 8 | duet_fast/1 | 1.004x | 1.004x |
| llama2 | ladder trim | 1 | duet_fast/1 | 1.003x | 1.003x |
| llama2 | ladder trim | 8 | duet_fast/0 | 1.004x | 1.005x |
| llama2 | ladder trim | 8 | duet_al/0 | 1.005x | 1.005x |
| llama3 | ladder trim | 1 | duet_fast/0 | 1.003x | 1.003x |
| llama3 | ladder trim | 8 | duet_fast/1 | 1.005x | 1.006x |
| llama3 | ladder trim | 1 | duet_fast/1 | 1.007x | 1.007x |
| llama3 | ladder trim | 8 | duet_fast/0 | 1.013x | 1.013x |
| llama3 | ladder trim | 8 | duet_al/0 | 1.008x | 1.008x |
| llama2 | proxy stream after trim | 1 | fast/0 | 1.003x | 1.003x |
| llama2 | proxy stream after trim | 8 | fast/1 | 0.927x | 0.927x |
| llama2 | proxy stream after trim | 1 | fast/1 | 1.011x | 1.011x |
| llama2 | proxy stream after trim | 8 | fast/0 | 0.939x | 0.938x |
| llama3 | proxy stream after trim | 1 | fast/0 | 1.014x | 1.014x |
| llama3 | proxy stream after trim | 8 | fast/1 | 0.974x | 0.974x |
| llama3 | proxy stream after trim | 1 | fast/1 | 1.015x | 1.015x |
| llama3 | proxy stream after trim | 8 | fast/0 | 0.950x | 0.951x |
