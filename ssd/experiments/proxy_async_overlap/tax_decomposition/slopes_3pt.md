
## proc = draft

### status shares
| status | K1=7 | K1=8 | K1=9 |
|---|---|---|---|
| hit_k1 | 0.592 | 0.595 | 0.577 |
| hit_k2 | 0.224 | 0.219 | 0.233 |
| miss | 0.184 | 0.187 | 0.190 |

### step wall (ms, per status)
| status | K1=7 | K1=8 | K1=9 | slope | mid-resid |
|---|---|---|---|---|---|
| hit_k1 | 50.61 | 54.60 | 58.00 | +3.70 | +0.30 |
| hit_k2 | 45.48 | 47.25 | 47.74 | +1.13 | +0.64 |
| miss | 61.55 | 67.79 | 75.10 | +6.78 | -0.54 |

### per-label mean per-step cuda_ms — status=hit_k1
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| phase1_replay | 17.627 | 28.861 | 32.662 | +7.518 | +3.717 |
| proxy_wait | 8.126 | 0.148 | 0.048 | -4.039 | -3.940 |
| draft_recv_cmd | 2.932 | 0.982 | 0.076 | -1.428 | -0.522 |
| phase2_replay | 10.142 | 12.556 | 12.585 | +1.222 | +1.193 |
| phase1_prep | 2.561 | 2.812 | 3.281 | +0.360 | -0.109 |
| draft_glue_replay | 1.659 | 1.785 | 1.790 | +0.066 | +0.061 |
| glue | 2.565 | 2.667 | 2.676 | +0.056 | +0.046 |
| phase2_prep | 1.890 | 1.845 | 1.859 | -0.015 | -0.029 |
| merge_cache | 0.086 | 0.100 | 0.114 | +0.014 | +0.000 |
| phase2_build | 0.813 | 0.778 | 0.789 | -0.012 | -0.023 |
| hit_cache_respond_hit_k1 | 0.887 | 0.853 | 0.864 | -0.011 | -0.022 |
| draft_recv_request | 0.341 | 0.319 | 0.320 | -0.011 | -0.011 |
| phase1_build | 0.136 | 0.145 | 0.154 | +0.009 | +0.000 |
| draft_send_response | 0.218 | 0.216 | 0.223 | +0.003 | -0.004 |

### per-label mean per-step cuda_ms — status=hit_k2
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| phase1_replay | 16.978 | 19.569 | 22.110 | +2.566 | +0.025 |
| proxy_wait | 4.992 | 3.151 | 0.477 | -2.257 | +0.417 |
| phase2_replay | 10.024 | 12.343 | 12.387 | +1.181 | +1.138 |
| draft_recv_cmd | 1.763 | 0.292 | 0.390 | -0.686 | -0.785 |
| phase1_prep | 2.553 | 2.784 | 3.271 | +0.359 | -0.128 |
| phase2_prep | 1.885 | 1.837 | 1.855 | -0.015 | -0.033 |
| phase2_build | 0.813 | 0.777 | 0.788 | -0.012 | -0.024 |
| merge_cache | 0.083 | 0.096 | 0.107 | +0.012 | +0.001 |
| hit_cache_respond_hit_k2 | 0.887 | 0.850 | 0.863 | -0.012 | -0.025 |
| draft_recv_request | 0.342 | 0.320 | 0.321 | -0.011 | -0.012 |
| glue | 2.553 | 2.536 | 2.537 | -0.008 | -0.009 |
| draft_send_response | 0.218 | 0.215 | 0.223 | +0.003 | -0.005 |
| draft_glue_replay | 1.649 | 1.657 | 1.653 | +0.002 | +0.006 |
| phase1_build | 0.134 | 0.133 | 0.135 | +0.001 | -0.002 |

### per-label mean per-step cuda_ms — status=miss
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| phase1_replay | 17.283 | 28.557 | 32.430 | +7.573 | +3.700 |
| proxy_wait | 7.999 | 0.056 | 0.003 | -3.998 | -3.945 |
| hit_cache_respond_miss | 13.389 | 15.195 | 16.974 | +1.792 | +0.013 |
| phase2_replay | 10.014 | 12.299 | 12.371 | +1.179 | +1.107 |
| phase1_prep | 2.565 | 2.994 | 3.134 | +0.284 | +0.144 |
| draft_recv_cmd | 2.108 | 0.386 | 1.705 | -0.202 | -1.521 |
| draft_glue_replay | 1.651 | 1.778 | 1.783 | +0.066 | +0.061 |
| glue | 2.556 | 2.656 | 2.668 | +0.056 | +0.044 |
| phase2_prep | 1.890 | 1.846 | 1.859 | -0.015 | -0.028 |
| merge_cache | 0.086 | 0.100 | 0.114 | +0.014 | +0.001 |
| phase2_build | 0.812 | 0.778 | 0.790 | -0.011 | -0.023 |
| phase1_build | 0.134 | 0.147 | 0.155 | +0.010 | +0.003 |
| draft_recv_request | 0.341 | 0.320 | 0.320 | -0.010 | -0.010 |
| draft_send_response | 0.220 | 0.218 | 0.224 | +0.002 | -0.004 |

## proc = target_rank0

### status shares
| status | K1=7 | K1=8 | K1=9 |
|---|---|---|---|
| hit_k1 | 0.592 | 0.595 | 0.577 |
| hit_k2 | 0.224 | 0.219 | 0.233 |
| miss | 0.184 | 0.187 | 0.190 |

### step wall (ms, per status)
| status | K1=7 | K1=8 | K1=9 | slope | mid-resid |
|---|---|---|---|---|---|
| hit_k1 | 49.95 | 53.97 | 57.40 | +3.73 | +0.30 |
| hit_k2 | 45.33 | 47.76 | 49.34 | +2.01 | +0.43 |
| miss | 61.70 | 67.12 | 72.99 | +5.65 | -0.22 |

### per-label mean per-step cuda_ms — status=hit_k1
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| child:target_recv_response_wait | 1.314 | 3.286 | 5.999 | +2.342 | -0.370 |
| target_spec_wait | 2.428 | 4.364 | 7.007 | +2.290 | -0.353 |
| graph_pre | 28.517 | 30.364 | 31.748 | +1.615 | +0.231 |
| graph_post | 10.954 | 11.644 | 12.159 | +0.603 | +0.087 |
| proxy_compute_send | 1.432 | 1.155 | 0.721 | -0.356 | +0.079 |
| verify_sample_accept | 3.599 | 3.489 | 3.104 | -0.247 | +0.138 |
| child:target_send_request | 0.529 | 0.488 | 0.464 | -0.033 | -0.008 |
| verify_setup | 0.330 | 0.324 | 0.292 | -0.019 | +0.012 |
| final_logits | 0.369 | 0.407 | 0.332 | -0.019 | +0.056 |
| exit_logits | 0.658 | 0.704 | 0.635 | -0.012 | +0.058 |
| target_postprocess | 0.093 | 0.078 | 0.075 | -0.009 | -0.007 |
| child:target_response_received | 0.002 | 0.003 | 0.003 | +0.000 | +0.000 |

### per-label mean per-step cuda_ms — status=hit_k2
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| child:target_recv_response_wait | 1.364 | 2.904 | 5.544 | +2.090 | -0.550 |
| target_spec_wait | 2.480 | 3.993 | 6.557 | +2.039 | -0.526 |
| verify_sample_accept | 4.037 | 3.943 | 3.751 | -0.143 | +0.049 |
| graph_pre | 25.951 | 26.131 | 26.131 | +0.090 | +0.090 |
| final_logits | 0.305 | 0.310 | 0.482 | +0.089 | -0.084 |
| proxy_compute_send | 0.354 | 0.347 | 0.475 | +0.060 | -0.068 |
| child:target_send_request | 0.531 | 0.492 | 0.466 | -0.033 | -0.007 |
| verify_setup | 0.324 | 0.329 | 0.291 | -0.016 | +0.022 |
| graph_post | 9.878 | 9.906 | 9.859 | -0.010 | +0.038 |
| target_postprocess | 0.090 | 0.073 | 0.080 | -0.005 | -0.012 |
| exit_logits | 0.312 | 1.243 | 0.318 | +0.003 | +0.928 |
| child:target_response_received | 0.002 | 0.003 | 0.002 | +0.000 | +0.000 |

### per-label mean per-step cuda_ms — status=miss
| label | K1=7 | K1=8 | K1=9 | slope ms/pos | mid-resid |
|---|---|---|---|---|---|
| child:target_recv_response_wait | 13.930 | 17.196 | 21.909 | +3.990 | -0.723 |
| target_spec_wait | 15.024 | 18.257 | 22.906 | +3.941 | -0.708 |
| graph_pre | 28.682 | 30.526 | 31.830 | +1.574 | +0.270 |
| graph_post | 10.947 | 11.648 | 12.160 | +0.606 | +0.094 |
| verify_sample_accept | 3.373 | 3.275 | 2.973 | -0.200 | +0.102 |
| proxy_compute_send | 0.935 | 0.764 | 0.563 | -0.186 | +0.015 |
| child:target_send_request | 0.523 | 0.492 | 0.463 | -0.030 | -0.001 |
| exit_logits | 0.677 | 0.719 | 0.652 | -0.012 | +0.054 |
| final_logits | 0.323 | 0.324 | 0.331 | +0.004 | -0.003 |
| target_postprocess | 0.073 | 0.060 | 0.066 | -0.004 | -0.010 |
| verify_setup | 0.267 | 0.273 | 0.265 | -0.001 | +0.006 |
| child:target_response_received | 0.002 | 0.003 | 0.003 | +0.000 | +0.000 |
