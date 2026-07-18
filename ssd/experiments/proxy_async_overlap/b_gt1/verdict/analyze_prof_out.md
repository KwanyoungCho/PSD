
## B=4 draft — 1240 steps, status shares: hit=529 (42.7%), hit_k1=9 (0.7%), hit_k2=2 (0.2%), miss=1 (0.1%), mixed=699 (56.4%)

### status=mixed (n=699, wall 124.28 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| phase1_replay | 47.307 | 9.00 | 5.256 | 22.58 |
| proxy_wait | 21.531 | 1.00 | 21.531 | 4.71 |
| phase2_replay | 17.808 | 4.00 | 4.452 | 9.93 |
| draft_recv_cmd | 12.188 | 1.00 | 12.188 | 4.59 |
| hit_cache_respond_mixed | 8.643 | 1.00 | 8.643 | - |
| glue | 4.481 | 1.00 | 4.481 | 2.70 |
| phase1_prep | 4.037 | 9.00 | 0.449 | 3.23 |
| draft_glue_replay | 3.601 | 1.00 | 3.601 | 1.78 |
| phase2_prep | 1.951 | 4.00 | 0.488 | 1.69 |
| phase1_build | 0.998 | 1.00 | 0.998 | 0.78 |
| phase2_build | 0.833 | 1.00 | 0.833 | 0.85 |
| draft_send_response | 0.471 | 1.00 | 0.471 | 0.22 |
| draft_recv_request | 0.325 | 1.00 | 0.325 | 0.34 |
| merge_cache | 0.321 | 1.00 | 0.321 | 0.11 |
| _top-level sum_ | 124.49 | | | wall 124.28 |

### status=hit (n=529, wall 119.31 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| phase1_replay | 47.898 | 9.00 | 5.322 | 22.58 |
| proxy_wait | 22.703 | 1.00 | 22.703 | 4.71 |
| phase2_replay | 17.829 | 4.00 | 4.457 | 9.93 |
| draft_recv_cmd | 12.884 | 1.00 | 12.884 | 4.59 |
| glue | 4.703 | 1.00 | 4.703 | 2.70 |
| phase1_prep | 4.026 | 9.00 | 0.447 | 3.23 |
| draft_glue_replay | 3.659 | 1.00 | 3.659 | 1.78 |
| phase2_prep | 1.953 | 4.00 | 0.488 | 1.69 |
| phase1_build | 1.058 | 1.00 | 1.058 | 0.78 |
| hit_cache_respond_hit | 0.889 | 1.00 | 0.889 | - |
| phase2_build | 0.833 | 1.00 | 0.833 | 0.85 |
| draft_send_response | 0.467 | 1.00 | 0.467 | 0.22 |
| draft_recv_request | 0.326 | 1.00 | 0.326 | 0.34 |
| merge_cache | 0.322 | 1.00 | 0.322 | 0.11 |
| _top-level sum_ | 119.55 | | | wall 119.31 |

### status=hit_k1 (n=9, wall 159.86 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| draft_recv_cmd | 104.112 | 1.00 | 104.112 | 4.59 |
| phase1_replay | 23.025 | 9.00 | 2.558 | 22.58 |
| phase2_replay | 10.265 | 4.00 | 2.566 | 9.93 |
| proxy_wait | 5.550 | 1.00 | 5.550 | 4.71 |
| phase1_build | 4.872 | 1.00 | 4.872 | 0.78 |
| phase1_prep | 3.166 | 9.00 | 0.352 | 3.23 |
| glue | 2.705 | 1.00 | 2.705 | 2.70 |
| draft_glue_replay | 1.809 | 1.00 | 1.809 | 1.78 |
| phase2_prep | 1.614 | 4.00 | 0.404 | 1.69 |
| hit_cache_respond_hit_k1 | 0.880 | 1.00 | 0.880 | 0.89 |
| phase2_build | 0.808 | 1.00 | 0.808 | 0.85 |
| draft_recv_request | 0.322 | 1.00 | 0.322 | 0.34 |
| draft_send_response | 0.221 | 1.00 | 0.221 | 0.22 |
| merge_cache | 0.108 | 1.00 | 0.108 | 0.11 |
| _top-level sum_ | 159.46 | | | wall 159.86 |

### status=hit_k2 (n=2, wall 49.02 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| phase1_replay | 22.321 | 9.00 | 2.480 | 22.58 |
| phase2_replay | 10.211 | 4.00 | 2.553 | 9.93 |
| proxy_wait | 3.784 | 1.00 | 3.784 | 4.71 |
| phase1_prep | 3.158 | 9.00 | 0.351 | 3.23 |
| glue | 2.716 | 1.00 | 2.716 | 2.70 |
| draft_glue_replay | 1.818 | 1.00 | 1.818 | 1.78 |
| phase2_prep | 1.597 | 4.00 | 0.399 | 1.69 |
| hit_cache_respond_hit_k2 | 0.880 | 1.00 | 0.880 | - |
| phase2_build | 0.802 | 1.00 | 0.802 | 0.85 |
| draft_recv_cmd | 0.545 | 1.00 | 0.545 | 4.59 |
| draft_recv_request | 0.325 | 1.00 | 0.325 | 0.34 |
| draft_send_response | 0.220 | 1.00 | 0.220 | 0.22 |
| phase1_build | 0.129 | 1.00 | 0.129 | 0.78 |
| merge_cache | 0.102 | 1.00 | 0.102 | 0.11 |
| _top-level sum_ | 48.61 | | | wall 49.02 |

### status=miss (n=1, wall 83.78 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| phase1_replay | 40.020 | 9.00 | 4.447 | 22.58 |
| phase2_replay | 17.773 | 4.00 | 4.443 | 9.93 |
| hit_cache_respond_miss | 8.295 | 1.00 | 8.295 | - |
| phase1_prep | 3.987 | 9.00 | 0.443 | 3.23 |
| glue | 3.673 | 1.00 | 3.673 | 2.70 |
| draft_glue_replay | 2.815 | 1.00 | 2.815 | 1.78 |
| phase2_prep | 1.911 | 4.00 | 0.478 | 1.69 |
| proxy_wait | 1.785 | 1.00 | 1.785 | 4.71 |
| draft_recv_cmd | 1.418 | 1.00 | 1.418 | 4.59 |
| phase2_build | 0.818 | 1.00 | 0.818 | 0.85 |
| draft_send_response | 0.470 | 1.00 | 0.470 | 0.22 |
| draft_recv_request | 0.322 | 1.00 | 0.322 | 0.34 |
| merge_cache | 0.285 | 1.00 | 0.285 | 0.11 |
| phase1_build | 0.220 | 1.00 | 0.220 | 0.78 |
| _top-level sum_ | 83.79 | | | wall 83.78 |

## B=4 target — 1240 steps, status shares: hit_k1=118 (9.5%), hit_k2=9 (0.7%), miss=1 (0.1%), mixed=1112 (89.7%)

### status=mixed (n=1112, wall 122.63 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| graph_pre | 78.937 | 1.00 | 78.937 | 31.54 |
| graph_post | 30.111 | 1.00 | 30.111 | 12.14 |
| target_spec_wait | 7.933 | 1.00 | 7.933 | 2.68 |
| child:target_recv_response_wait | 6.586 | 1.00 | 6.586 | 1.52 |
| verify_sample_accept | 2.404 | 1.00 | 2.404 | 3.64 |
| exit_logits | 0.626 | 1.00 | 0.626 | 0.78 |
| final_logits | 0.582 | 1.00 | 0.582 | 0.36 |
| proxy_compute_send | 0.498 | 1.00 | 0.498 | 1.48 |
| child:target_send_request | 0.353 | 1.00 | 0.353 | 0.53 |
| verify_setup | 0.254 | 1.00 | 0.254 | 0.35 |
| target_postprocess | 0.074 | 1.00 | 0.074 | 0.09 |
| child:target_response_received | 0.003 | 1.00 | 0.003 | 0.00 |
| _top-level sum_ | 121.42 | | | wall 122.63 |

### status=hit_k1 (n=118, wall 114.49 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| graph_pre | 76.396 | 1.00 | 76.396 | 31.54 |
| graph_post | 29.171 | 1.00 | 29.171 | 12.14 |
| target_spec_wait | 2.950 | 1.00 | 2.950 | 2.68 |
| verify_sample_accept | 2.487 | 1.00 | 2.487 | 3.64 |
| child:target_recv_response_wait | 1.638 | 1.00 | 1.638 | 1.52 |
| proxy_compute_send | 0.672 | 1.00 | 0.672 | 1.48 |
| exit_logits | 0.665 | 1.00 | 0.665 | 0.78 |
| final_logits | 0.571 | 1.00 | 0.571 | 0.36 |
| child:target_send_request | 0.367 | 1.00 | 0.367 | 0.53 |
| verify_setup | 0.261 | 1.00 | 0.261 | 0.35 |
| target_postprocess | 0.081 | 1.00 | 0.081 | 0.09 |
| child:target_response_received | 0.002 | 1.00 | 0.002 | 0.00 |
| _top-level sum_ | 113.25 | | | wall 114.49 |

### status=hit_k2 (n=9, wall 72.67 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| graph_pre | 45.241 | 1.00 | 45.241 | 31.54 |
| graph_post | 16.545 | 1.00 | 16.545 | 12.14 |
| target_spec_wait | 5.364 | 1.00 | 5.364 | 2.68 |
| child:target_recv_response_wait | 4.122 | 1.00 | 4.122 | 1.52 |
| verify_sample_accept | 2.741 | 1.00 | 2.741 | 3.64 |
| final_logits | 0.418 | 1.00 | 0.418 | 0.36 |
| proxy_compute_send | 0.410 | 1.00 | 0.410 | 1.48 |
| exit_logits | 0.404 | 1.00 | 0.404 | 0.78 |
| child:target_send_request | 0.365 | 1.00 | 0.365 | 0.53 |
| verify_setup | 0.270 | 1.00 | 0.270 | 0.35 |
| target_postprocess | 0.071 | 1.00 | 0.071 | 0.09 |
| child:target_response_received | 0.003 | 1.00 | 0.003 | 0.00 |
| _top-level sum_ | 71.46 | | | wall 72.67 |

### status=miss (n=1, wall 83.56 ms)
| label | ms/step | n/step | ms/unit | B=1 ref (hit_k1) |
|---|---|---|---|---|
| graph_pre | 49.144 | 1.00 | 49.144 | 31.54 |
| graph_post | 18.520 | 1.00 | 18.520 | 12.14 |
| target_spec_wait | 10.396 | 1.00 | 10.396 | 2.68 |
| child:target_recv_response_wait | 9.087 | 1.00 | 9.087 | 1.52 |
| verify_sample_accept | 2.708 | 1.00 | 2.708 | 3.64 |
| final_logits | 0.434 | 1.00 | 0.434 | 0.36 |
| exit_logits | 0.427 | 1.00 | 0.427 | 0.78 |
| proxy_compute_send | 0.414 | 1.00 | 0.414 | 1.48 |
| child:target_send_request | 0.339 | 1.00 | 0.339 | 0.53 |
| verify_setup | 0.239 | 1.00 | 0.239 | 0.35 |
| target_postprocess | 0.129 | 1.00 | 0.129 | 0.09 |
| child:target_response_received | 0.002 | 1.00 | 0.002 | 0.00 |
| _top-level sum_ | 82.41 | | | wall 83.56 |

## per-forward distributions (draft)
B=1 phase1_replay/fwd: n=85752 mean=2.465 p10=2.188 p25=2.379 p50=2.538 p75=2.579 p90=2.610 p99=2.619
B=4 phase1_replay/fwd: n=11160 mean=5.260 p10=5.168 p25=5.257 p50=5.328 p75=5.402 p90=5.488 p99=5.535
  B=4 forwards/step: mean 9.00
B=1 phase2_replay/fwd: n=38112 mean=2.466 p10=2.260 p25=2.452 p50=2.551 p75=2.558 p90=2.564 p99=2.570
B=4 phase2_replay/fwd: n=4960 mean=4.437 p10=4.371 p25=4.440 p50=4.465 p75=4.495 p90=4.532 p99=4.595
  B=4 forwards/step: mean 4.00
B=1 draft_glue_replay/fwd: n=9528 mean=1.721 p10=1.632 p25=1.641 p50=1.769 p75=1.785 p90=1.796 p99=1.808
B=4 draft_glue_replay/fwd: n=1240 mean=3.609 p10=3.650 p25=3.660 p50=3.671 p75=3.683 p90=3.699 p99=3.753
  B=4 forwards/step: mean 1.00

## verify width (vk_max) distribution
B=4 graph_pre: n=1240 mean=78.426 p10=80.072 p25=80.493 p50=80.663 p75=80.776 p90=80.939 p99=82.150
B=1 graph_pre: n=9528 mean=29.018 p10=25.224 p25=25.462 p50=31.267 p75=31.562 p90=31.751 p99=32.343
graph_pre histogram (bin_start:count):
    25.41: # 1
    28.46: # 0
    31.51: # 8
    34.56: # 2
    37.60: # 0
    40.65: # 0
    43.70: # 0
    46.75: ### 67
    49.79: # 4
    52.84: # 1
    55.89: # 0
    58.94: # 0
    61.98: # 0
    65.03: # 0
    68.08: # 0
    71.13: # 0
    74.17: # 0
    77.22: ##### 104
    80.27: ############################################################ 1050
    83.32: # 3
