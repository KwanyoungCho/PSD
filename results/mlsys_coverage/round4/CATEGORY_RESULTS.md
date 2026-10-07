# Task group별 결과

B8 / 3 target-seed pass / task group각80질문. Boundary-excluded AL, 질문 paired bootstrap2,000회. 다중비교 보정 없이 세부 경향을 보는 사후 분석이며 개별 CI를 독립적인 확증으로 해석하지 않는다.

| Model | Policy | Task | AL* Δ% | ΔAL* CI95 |
|---|---|---|---:|---|
| llama2 | reach_gain_frontier | math_reasoning | +1.32% | [-0.029, +0.083] |
| llama2 | reach_gain_frontier | mt_bench | +2.65% | [-0.015, +0.124] |
| llama2 | reach_gain_frontier | qa | +1.57% | [-0.024, +0.089] |
| llama2 | reach_gain_frontier | rag | +0.73% | [-0.037, +0.072] |
| llama2 | reach_gain_frontier | summarization | -0.59% | [-0.041, +0.023] |
| llama2 | reach_gain_frontier | translation | -1.07% | [-0.091, +0.047] |
| llama2 | dense_reach | math_reasoning | +1.20% | [-0.023, +0.075] |
| llama2 | dense_reach | mt_bench | -0.73% | [-0.085, +0.051] |
| llama2 | dense_reach | qa | +0.29% | [-0.049, +0.060] |
| llama2 | dense_reach | rag | -0.51% | [-0.070, +0.050] |
| llama2 | dense_reach | summarization | -0.19% | [-0.033, +0.027] |
| llama2 | dense_reach | translation | -0.67% | [-0.075, +0.046] |
| llama2 | dense_reach_gain_frontier | math_reasoning | +1.07% | [-0.021, +0.067] |
| llama2 | dense_reach_gain_frontier | mt_bench | +0.80% | [-0.069, +0.093] |
| llama2 | dense_reach_gain_frontier | qa | +3.60% | [+0.016, +0.134] |
| llama2 | dense_reach_gain_frontier | rag | +2.52% | [+0.010, +0.120] |
| llama2 | dense_reach_gain_frontier | summarization | -1.41% | [-0.054, +0.013] |
| llama2 | dense_reach_gain_frontier | translation | +1.13% | [-0.046, +0.089] |
| llama3 | reach_gain_frontier | math_reasoning | +1.95% | [-0.031, +0.133] |
| llama3 | reach_gain_frontier | mt_bench | +1.31% | [-0.036, +0.103] |
| llama3 | reach_gain_frontier | qa | +1.80% | [-0.050, +0.135] |
| llama3 | reach_gain_frontier | rag | +1.18% | [-0.028, +0.082] |
| llama3 | reach_gain_frontier | summarization | +4.20% | [+0.029, +0.171] |
| llama3 | reach_gain_frontier | translation | +1.70% | [-0.044, +0.119] |
| llama3 | dense_reach | math_reasoning | -0.78% | [-0.097, +0.058] |
| llama3 | dense_reach | mt_bench | +1.87% | [-0.025, +0.115] |
| llama3 | dense_reach | qa | +0.93% | [-0.061, +0.108] |
| llama3 | dense_reach | rag | +1.04% | [-0.033, +0.081] |
| llama3 | dense_reach | summarization | +0.89% | [-0.050, +0.088] |
| llama3 | dense_reach | translation | +0.28% | [-0.062, +0.083] |
| llama3 | dense_reach_gain_frontier | math_reasoning | +0.48% | [-0.068, +0.092] |
| llama3 | dense_reach_gain_frontier | mt_bench | +3.09% | [+0.002, +0.147] |
| llama3 | dense_reach_gain_frontier | qa | +2.89% | [-0.014, +0.158] |
| llama3 | dense_reach_gain_frontier | rag | +1.88% | [-0.015, +0.106] |
| llama3 | dense_reach_gain_frontier | summarization | +3.58% | [+0.015, +0.150] |
| llama3 | dense_reach_gain_frontier | translation | +1.09% | [-0.056, +0.101] |
