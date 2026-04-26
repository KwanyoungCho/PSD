# 70B AWQ Sweep Summary

Total completed runs: 51
Stack: layerskip-llama2-70B (AWQ TP=4) + TinyLlama-1.1B (AWQ TP=1)

Reference: AR baseline (from experiments/quant_awq/70b/ar) = 32.87 tok/s

| run | mode | k | f | dfo | exit | policy | TP | accept | CH | draft_ms | verify_ms | tok/step |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| modeasync_k7_f8_ns128_ol256 | async | 7 | 8 | — | — | — | 79.41 | 0.47 | 0.80 | 43.96 | 45.53 | 4.26 |
| modeasync_k6_f8_ns128_ol256 | async | 6 | 8 | — | — | — | 77.59 | 0.48 | 0.81 | 36.94 | 43.21 | 3.90 |
| modeasync_k6_f7_ns128_ol256 | async | 6 | 7 | — | — | — | 77.10 | 0.48 | 0.79 | 36.92 | 43.10 | 3.88 |
| modeasync_k7_f8_ns200_ol256 | async | 7 | 8 | — | — | — | 76.78 | 0.44 | 0.79 | 43.95 | 45.31 | 4.10 |
| modeasync_k7_f7_ns128_ol256 | async | 7 | 7 | — | — | — | 75.75 | 0.43 | 0.78 | 43.61 | 45.10 | 4.03 |
| modeasync_k6_f9_ns128_ol256 | async | 6 | 9 | — | — | — | 75.60 | 0.47 | 0.81 | 37.34 | 43.27 | 3.80 |
| modeasync_k5_f9_ns128_ol256 | async | 5 | 9 | — | — | — | 75.54 | 0.53 | 0.83 | 30.49 | 41.79 | 3.64 |
| modeasync_k5_f8_ns128_ol256 | async | 5 | 8 | — | — | — | 74.30 | 0.52 | 0.81 | 27.45 | 41.79 | 3.59 |
| modeasync_k5_f7_ns128_ol256 | async | 5 | 7 | — | — | — | 73.32 | 0.51 | 0.80 | 27.24 | 41.94 | 3.56 |
| modemesa_k6_f3_dfo2_exit53_policya_ns128_ol256 | mesa | 6 | 3 | 2 | 53 | a | 72.81 | 0.48 | 0.81 | 48.37 | 45.85 | 3.88 |
| modemesa_k6_f3_dfo2_exit53_policya_ns200_ol256 | mesa | 6 | 3 | 2 | 53 | a | 72.25 | 0.48 | 0.83 | 48.87 | 46.83 | 3.91 |
| modemesa_k5_f3_dfo2_exit53_policya_ns128_ol256 | mesa | 5 | 3 | 2 | 53 | a | 71.99 | 0.53 | 0.82 | 44.48 | 44.21 | 3.65 |
| modemesa_k5_f4_dfo2_exit53_policya_ns128_ol256 | mesa | 5 | 4 | 2 | 53 | a | 71.06 | 0.52 | 0.84 | 45.97 | 43.92 | 3.62 |
| modemesa_k4_f4_dfo2_exit53_policya_ns128_ol256 | mesa | 4 | 4 | 2 | 53 | a | 69.28 | 0.58 | 0.86 | 41.62 | 42.59 | 3.33 |
| modemesa_k4_f5_dfo2_exit53_policya_ns128_ol256 | mesa | 4 | 5 | 2 | 53 | a | 67.84 | 0.57 | 0.87 | 42.81 | 42.94 | 3.29 |
| modemesa_k4_f3_dfo2_exit53_policya_ns128_ol256 | mesa | 4 | 3 | 2 | 53 | a | 67.74 | 0.58 | 0.83 | 40.71 | 42.98 | 3.30 |
| modemesa_k6_f4_dfo2_exit53_policya_ns128_ol256 | mesa | 6 | 4 | 2 | 53 | a | 67.58 | 0.48 | 0.84 | 52.22 | 45.82 | 3.87 |
| modeasync_k6_f8_ns64_ol128 | async | 6 | 8 | — | — | — | 66.47 | 0.44 | 0.76 | 37.26 | 44.13 | 3.62 |
| modeasync_k7_f9_ns128_ol256 | async | 7 | 9 | — | — | — | 66.16 | 0.44 | 0.81 | 51.56 | 44.96 | 4.06 |
| modeasync_k7_f8_ns64_ol128 | async | 7 | 8 | — | — | — | 65.10 | 0.39 | 0.75 | 44.20 | 45.77 | 3.70 |
| modeasync_k7_f3_ns64_ol128 | async | 7 | 3 | — | — | — | 64.97 | 0.40 | 0.62 | — | — | — |
| modeasync_k7_f6_ns64_ol128 | async | 7 | 6 | — | — | — | 64.69 | 0.39 | 0.72 | 39.13 | 45.95 | — |
| modeasync_k5_f3_ns64_ol128 | async | 5 | 3 | — | — | — | 64.37 | 0.47 | 0.64 | — | — | — |
| modemesa_k5_f5_dfo2_exit53_policya_ns128_ol256 | mesa | 5 | 5 | 2 | 53 | a | 64.33 | 0.53 | 0.86 | 50.51 | 44.30 | 3.63 |
| modemesa_k5_f4_dfo2_exit53_policya_ns64_ol128 | mesa | 5 | 4 | 2 | 53 | a | 64.19 | 0.48 | 0.82 | 45.36 | 43.81 | 3.39 |
| modeasync_k7_f2_ns64_ol128 | async | 7 | 2 | — | — | — | 63.56 | 0.39 | 0.55 | — | — | — |
| modeasync_k7_f4_ns64_ol128 | async | 7 | 4 | — | — | — | 63.43 | 0.39 | 0.66 | — | — | — |
| modemesa_k6_f5_dfo2_exit53_policya_ns128_ol256 | mesa | 6 | 5 | 2 | 53 | a | 63.03 | 0.49 | 0.86 | 56.04 | 45.83 | 3.93 |
| modemesa_k5_f4_dfo2_exit53_policyb_ns64_ol128 | mesa | 5 | 4 | 2 | 53 | b | 62.08 | 0.48 | 0.83 | 46.98 | 45.97 | 3.41 |
| modemesa_k5_f4_dfo2_exit40_policyb_ns64_ol128 | mesa | 5 | 4 | 2 | 40 | b | 61.19 | 0.51 | 0.77 | 42.86 | 48.22 | 3.55 |
| modemesa_k5_f4_dfo2_exit46_policya_ns64_ol128 | mesa | 5 | 4 | 2 | 46 | a | 61.04 | 0.48 | 0.80 | 44.03 | 46.22 | 3.38 |
| modemesa_k5_f8_dfo4_exit46_policya_ns64_ol128 | mesa | 5 | 8 | 4 | 46 | a | 60.99 | 0.49 | 0.86 | 48.32 | 45.75 | 3.47 |
| modemesa_k5_f8_dfo4_exit46_policya_ns64_ol128 | mesa | 5 | 8 | 4 | 46 | a | 60.99 | 0.48 | 0.86 | 47.44 | 45.06 | 3.41 |
| modemesa_k5_f4_dfo2_exit46_policyb_ns64_ol128 | mesa | 5 | 4 | 2 | 46 | b | 60.91 | 0.50 | 0.80 | 45.69 | 48.02 | 3.49 |
| modeasync_k3_f3_ns64_ol128 | async | 3 | 3 | — | — | — | 60.84 | 0.62 | 0.71 | — | — | — |
| modemesa_k5_f4_dfo2_exit46_policya_ns64_ol128 | mesa | 5 | 4 | 2 | 46 | a | 60.20 | 0.48 | 0.80 | 45.60 | 46.47 | 3.38 |
| modemesa_k5_f6_dfo3_exit46_policya_ns64_ol128 | mesa | 5 | 6 | 3 | 46 | a | 60.12 | 0.47 | 0.83 | 47.43 | 45.03 | 3.35 |
| modemesa_k5_f4_dfo2_exit46_policyb_ns64_ol128 | mesa | 5 | 4 | 2 | 46 | b | 60.02 | 0.48 | 0.81 | 45.09 | 47.88 | 3.42 |
| modemesa_k5_f4_dfo1_exit46_policya_ns64_ol128 | mesa | 5 | 4 | 1 | 46 | a | 59.66 | 0.48 | 0.78 | 48.82 | 46.49 | 3.41 |
| modemesa_k5_f8_dfo4_exit40_policya_ns64_ol128 | mesa | 5 | 8 | 4 | 40 | a | 59.65 | 0.48 | 0.83 | 47.77 | 47.44 | 3.39 |
| modemesa_k5_f4_dfo2_exit40_policya_ns64_ol128 | mesa | 5 | 4 | 2 | 40 | a | 59.42 | 0.47 | 0.76 | 41.73 | 46.70 | 3.34 |
| modemesa_k5_f8_dfo4_exit46_policyb_ns64_ol128 | mesa | 5 | 8 | 4 | 46 | b | 59.28 | 0.49 | 0.87 | 49.95 | 47.96 | 3.47 |
| modemesa_k5_f6_dfo2_exit46_policya_ns64_ol128 | mesa | 5 | 6 | 2 | 46 | a | 59.09 | 0.47 | 0.82 | 48.63 | 46.32 | 3.37 |
| modemesa_k5_f8_dfo4_exit53_policya_ns64_ol128 | mesa | 5 | 8 | 4 | 53 | a | 58.86 | 0.49 | 0.88 | 50.34 | 44.86 | 3.47 |
| modemesa_k5_f6_dfo3_exit46_policyb_ns64_ol128 | mesa | 5 | 6 | 3 | 46 | b | 58.46 | 0.48 | 0.84 | 49.69 | 47.70 | 3.39 |
| modeasync_k5_f3_ns16_ol64 | async | 5 | 3 | — | — | — | 57.18 | 0.45 | 0.61 | — | — | — |
| modemesa_k5_f8_dfo2_exit46_policya_ns64_ol128 | mesa | 5 | 8 | 2 | 46 | a | 55.97 | 0.48 | 0.83 | 51.31 | 44.23 | 3.40 |
| modemesa_k3_f6_dfo2_exit46_policya_ns64_ol128 | mesa | 3 | 6 | 2 | 46 | a | 55.96 | 0.59 | 0.85 | 35.25 | 42.38 | 2.77 |
| modeasync_k8_f8_ns64_ol128 | async | 8 | 8 | — | — | — | 54.26 | 0.37 | 0.75 | 59.25 | 46.43 | 3.92 |
| modemesa_k7_f6_dfo2_exit46_policya_ns64_ol128 | mesa | 7 | 6 | 2 | 46 | a | 54.11 | 0.41 | 0.80 | 61.27 | 50.59 | 3.84 |
| modemesa_k5_f4_dfo2_exit46_policya_ns16_ol64 | mesa | 5 | 4 | 2 | 46 | a | 53.09 | 0.44 | 0.77 | — | — | — |
