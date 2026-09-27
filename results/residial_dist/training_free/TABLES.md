# Independent confirmation tables

Prompt-balanced coverage; 96 unseen prompts per T. CI: 4,000 paired, dataset-stratified prompt resamples.
The original proxy baseline already uses q in its position weights; “proxy” describes the candidate source score.

| T | Policy | Coverage (%) | vs original proxy (pp, 95% CI) | vs optimized proxy (pp, 95% CI) |
|---|---|---:|---:|---:|
| 1.0 | original_residual | 61.7617 | -3.2777 [-4.2829, -2.3523] | -4.9248 [-6.0031, -3.9004] |
| 1.0 | original_proxy | 65.0394 | +0.0000 [+0.0000, +0.0000] | -1.6471 [-2.2429, -1.1034] |
| 1.0 | proxy_allocation | 66.6864 | +1.6471 [+1.1034, +2.2429] | +0.0000 [+0.0000, +0.0000] |
| 1.0 | overall | 66.8117 | +1.7724 [+1.2189, +2.3647] | +0.1253 [+0.0464, +0.2092] |
| 0.7 | original_residual | 78.2751 | -3.7673 [-4.7319, -2.8995] | -6.1641 [-7.4744, -5.0000] |
| 0.7 | original_proxy | 82.0424 | +0.0000 [+0.0000, +0.0000] | -2.3968 [-3.1484, -1.6802] |
| 0.7 | proxy_allocation | 84.4392 | +2.3968 [+1.6802, +3.1484] | +0.0000 [+0.0000, +0.0000] |
| 0.7 | overall | 84.4305 | +2.3880 [+1.6172, +3.2073] | -0.0088 [-0.2687, +0.2396] |
| 0.5 | original_residual | 85.0998 | -3.8310 [-4.6389, -3.0837] | -4.7889 [-5.9713, -3.6657] |
| 0.5 | original_proxy | 88.9309 | +0.0000 [+0.0000, +0.0000] | -0.9579 [-1.8653, -0.0906] |
| 0.5 | proxy_allocation | 89.8887 | +0.9579 [+0.0906, +1.8653] | +0.0000 [+0.0000, +0.0000] |
| 0.5 | overall | 91.3268 | +2.3959 [+1.4569, +3.4022] | +1.4380 [+0.8783, +2.0571] |

## Factorized ablations

| T | Change | Coverage (%) | vs original proxy (pp, 95% CI) |
|---|---|---:|---:|
| 1.0 | original_proxy | 65.0394 | +0.0000 [+0.0000, +0.0000] |
| 1.0 | original_residual | 61.7617 | -3.2777 [-4.2829, -2.3523] |
| 1.0 | full_only | 65.4518 | +0.4124 [+0.1748, +0.7063] |
| 1.0 | hazard_only | 66.6510 | +1.6116 [+1.0992, +2.1813] |
| 1.0 | allocation_only | 66.6864 | +1.6471 [+1.1034, +2.2429] |
| 1.0 | source_only | 65.1280 | +0.0886 [-0.0208, +0.2064] |
| 1.0 | source_plus_full | 65.4891 | +0.4497 [+0.2396, +0.6858] |
| 1.0 | source_plus_hazard | 66.6560 | +1.6167 [+1.1026, +2.1821] |
| 1.0 | selected | 66.8117 | +1.7724 [+1.2189, +2.3647] |
| 1.0 | optimized_proxy | 66.6864 | +1.6471 [+1.1034, +2.2429] |
| 1.0 | optimized_residual | 62.8680 | -2.1714 [-3.2005, -1.1879] |
| 1.0 | legacy_unscaled_proxy | 65.0394 | +0.0000 [-0.0000, +0.0000] |
| 1.0 | legacy_unscaled_residual | 61.7617 | -3.2777 [-4.2829, -2.3523] |
| 0.7 | original_proxy | 82.0424 | +0.0000 [+0.0000, +0.0000] |
| 0.7 | original_residual | 78.2751 | -3.7673 [-4.7319, -2.8995] |
| 0.7 | full_only | 82.1038 | +0.0613 [-0.0136, +0.1463] |
| 0.7 | hazard_only | 84.3872 | +2.3448 [+1.6179, +3.0944] |
| 0.7 | allocation_only | 84.4392 | +2.3968 [+1.6802, +3.1484] |
| 0.7 | source_only | 82.0410 | -0.0014 [-0.0935, +0.1031] |
| 0.7 | source_plus_full | 82.0785 | +0.0360 [-0.0747, +0.1592] |
| 0.7 | source_plus_hazard | 84.4228 | +2.3804 [+1.6145, +3.1931] |
| 0.7 | selected | 84.4305 | +2.3880 [+1.6172, +3.2073] |
| 0.7 | optimized_proxy | 84.4392 | +2.3968 [+1.6802, +3.1484] |
| 0.7 | optimized_residual | 79.5275 | -2.5149 [-3.5519, -1.5979] |
| 0.7 | legacy_unscaled_proxy | 81.4927 | -0.5497 [-1.1454, +0.0047] |
| 0.7 | legacy_unscaled_residual | 77.2134 | -4.8290 [-5.9764, -3.7881] |
| 0.5 | original_proxy | 88.9309 | +0.0000 [+0.0000, +0.0000] |
| 0.5 | original_residual | 85.0998 | -3.8310 [-4.6389, -3.0837] |
| 0.5 | full_only | 88.9345 | +0.0036 [-0.0004, +0.0100] |
| 0.5 | hazard_only | 91.2649 | +2.3340 [+1.3898, +3.3423] |
| 0.5 | allocation_only | 91.2695 | +2.3386 [+1.3938, +3.3450] |
| 0.5 | source_only | 88.8264 | -0.1045 [-0.3606, +0.1077] |
| 0.5 | source_plus_full | 88.8308 | -0.1001 [-0.3551, +0.1121] |
| 0.5 | source_plus_hazard | 91.2142 | +2.2833 [+1.2968, +3.3362] |
| 0.5 | selected | 91.3268 | +2.3959 [+1.4569, +3.4022] |
| 0.5 | optimized_proxy | 89.8887 | +0.9579 [+0.0906, +1.8653] |
| 0.5 | optimized_residual | 86.2567 | -2.6742 [-3.5779, -1.7834] |
| 0.5 | legacy_unscaled_proxy | 88.6242 | -0.3067 [-1.1723, +0.5825] |
| 0.5 | legacy_unscaled_residual | 83.9965 | -4.9343 [-6.2172, -3.7308] |

## Frozen family winners (not reselected on confirmation)

| T | Family | Frozen key | Coverage (%) | vs original proxy (pp) |
|---|---|---|---:|---:|
| 1.0 | bounded_subtraction | `policy::complement_power0.5__full__expected_mix0.25` | 66.8117 | +1.7724 |
| 1.0 | local_gate | `policy::qpeak0.9__full__expected_mix0.25` | 66.5963 | +1.5569 |
| 1.0 | proxy_allocation | `policy::proxy__full__expected_mix0.25` | 66.6864 | +1.6471 |
| 1.0 | rank_only | `policy::interval1__full__expected_mix0.25` | 66.6864 | +1.6471 |
| 1.0 | residual_allocation | `policy::residual__full__expected_mix0.25` | 62.8680 | -2.1714 |
| 1.0 | source_temperature | `policy::source_tau0.9__full__expected_mix0.25` | 63.0642 | -1.9752 |
| 1.0 | overall | `policy::complement_power0.5__full__expected_mix0.25` | 66.8117 | +1.7724 |
| 1.0 | source_only | `policy::interval1__topm__original` | 65.0394 | +0.0000 |
| 1.0 | original_proxy | `policy::proxy__topm__original` | 65.0394 | +0.0000 |
| 1.0 | original_residual | `policy::residual__topm__original` | 61.7617 | -3.2777 |
| 0.7 | bounded_subtraction | `policy::complement_power1__full__expected_mix0.25` | 84.4305 | +2.3880 |
| 0.7 | local_gate | `policy::qpeak0.9__full__expected_mix0.25` | 84.1511 | +2.1086 |
| 0.7 | proxy_allocation | `policy::proxy__full__expected_mix0.25` | 84.4392 | +2.3968 |
| 0.7 | rank_only | `policy::interval1__full__expected_mix0.25` | 84.4392 | +2.3968 |
| 0.7 | residual_allocation | `policy::residual__full__uniform0.05` | 79.5275 | -2.5149 |
| 0.7 | source_temperature | `policy::source_tau1.1__full__uniform0.05` | 79.4165 | -2.6259 |
| 0.7 | overall | `policy::complement_power1__full__expected_mix0.25` | 84.4305 | +2.3880 |
| 0.7 | source_only | `policy::complement_power1__topm__original` | 82.0410 | -0.0014 |
| 0.7 | original_proxy | `policy::proxy__topm__original` | 82.0424 | +0.0000 |
| 0.7 | original_residual | `policy::residual__topm__original` | 78.2751 | -3.7673 |
| 0.5 | bounded_subtraction | `policy::floor0.75__full__expected_mix0.25` | 91.3268 | +2.3959 |
| 0.5 | local_gate | `policy::qpeak0.9__full__expected_mix0.25` | 90.6222 | +1.6913 |
| 0.5 | proxy_allocation | `policy::proxy__full__uniform0.05` | 89.8887 | +0.9579 |
| 0.5 | rank_only | `policy::keep12__full__uniform0.05` | 89.8884 | +0.9575 |
| 0.5 | residual_allocation | `policy::residual__full__expected_mix0.75` | 86.2567 | -2.6742 |
| 0.5 | source_temperature | `policy::source_tau1.1__full__expected_mix0.5` | 86.8104 | -2.1204 |
| 0.5 | overall | `policy::floor0.75__full__expected_mix0.25` | 91.3268 | +2.3959 |
| 0.5 | source_only | `policy::floor0.75__topm__original` | 88.8264 | -0.1045 |
| 0.5 | original_proxy | `policy::proxy__topm__original` | 88.9309 | +0.0000 |
| 0.5 | original_residual | `policy::residual__topm__original` | 85.0998 | -3.8310 |

## Dataset means for primary policies

| T | Policy | Alpaca | C4 | GSM | HumanEval |
|---|---|---:|---:|---:|---:|
| 1.0 | original_residual | 60.404 | 55.382 | 60.780 | 70.481 |
| 1.0 | original_proxy | 62.171 | 60.508 | 64.120 | 73.358 |
| 1.0 | proxy_allocation | 62.875 | 62.650 | 66.237 | 74.983 |
| 1.0 | overall | 62.950 | 62.817 | 66.312 | 75.169 |
| 0.7 | original_residual | 78.259 | 73.939 | 75.386 | 85.516 |
| 0.7 | original_proxy | 80.504 | 79.240 | 79.278 | 89.147 |
| 0.7 | proxy_allocation | 82.297 | 80.654 | 83.504 | 91.302 |
| 0.7 | overall | 82.388 | 80.514 | 83.440 | 91.379 |
| 0.5 | original_residual | 88.055 | 77.268 | 87.039 | 88.038 |
| 0.5 | original_proxy | 90.726 | 83.944 | 90.407 | 90.646 |
| 0.5 | proxy_allocation | 91.683 | 84.430 | 90.877 | 92.565 |
| 0.5 | overall | 92.548 | 87.361 | 92.009 | 93.389 |
