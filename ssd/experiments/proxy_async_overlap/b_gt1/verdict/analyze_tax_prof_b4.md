
## proc = draft

### status shares
| status | K1=4 |
|---|---|
| hit | 0.427 |
| hit_k1 | 0.007 |
| hit_k2 | 0.002 |
| miss | 0.001 |
| mixed | 0.564 |

### step wall (ms, per status)
| status | K1=4 | slope | mid-resid |
|---|---|---|---|
| hit | 119.31 | +nan | - |
| hit_k1 | 159.86 | +nan | - |
| hit_k2 | 49.02 | +nan | - |
| miss | 83.78 | +nan | - |
| mixed | 124.28 | +nan | - |

### per-label mean per-step cuda_ms — status=hit_k1
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| draft_recv_request | 0.322 | +nan | - |
| hit_cache_respond_hit_k1 | 0.880 | +nan | - |
| draft_send_response | 0.221 | +nan | - |
| draft_glue_replay | 1.809 | +nan | - |
| glue | 2.705 | +nan | - |
| phase1_build | 4.872 | +nan | - |
| phase1_prep | 3.166 | +nan | - |
| phase1_replay | 23.025 | +nan | - |
| proxy_wait | 5.550 | +nan | - |
| phase2_build | 0.808 | +nan | - |
| phase2_prep | 1.614 | +nan | - |
| phase2_replay | 10.265 | +nan | - |
| merge_cache | 0.108 | +nan | - |
| draft_recv_cmd | 104.112 | +nan | - |

### per-label mean per-step cuda_ms — status=hit_k2
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| draft_recv_request | 0.325 | +nan | - |
| hit_cache_respond_hit_k2 | 0.880 | +nan | - |
| draft_send_response | 0.220 | +nan | - |
| draft_glue_replay | 1.818 | +nan | - |
| glue | 2.716 | +nan | - |
| phase1_build | 0.129 | +nan | - |
| phase1_prep | 3.158 | +nan | - |
| phase1_replay | 22.321 | +nan | - |
| proxy_wait | 3.784 | +nan | - |
| phase2_build | 0.802 | +nan | - |
| phase2_prep | 1.597 | +nan | - |
| phase2_replay | 10.211 | +nan | - |
| merge_cache | 0.102 | +nan | - |
| draft_recv_cmd | 0.545 | +nan | - |

### per-label mean per-step cuda_ms — status=miss
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| draft_recv_request | 0.322 | +nan | - |
| hit_cache_respond_miss | 8.295 | +nan | - |
| draft_send_response | 0.470 | +nan | - |
| draft_glue_replay | 2.815 | +nan | - |
| glue | 3.673 | +nan | - |
| phase1_build | 0.220 | +nan | - |
| phase1_prep | 3.987 | +nan | - |
| phase1_replay | 40.020 | +nan | - |
| proxy_wait | 1.785 | +nan | - |
| phase2_build | 0.818 | +nan | - |
| phase2_prep | 1.911 | +nan | - |
| phase2_replay | 17.773 | +nan | - |
| merge_cache | 0.285 | +nan | - |
| draft_recv_cmd | 1.418 | +nan | - |

## proc = target_rank0

### status shares
| status | K1=4 |
|---|---|
| hit_k1 | 0.095 |
| hit_k2 | 0.007 |
| miss | 0.001 |
| mixed | 0.897 |

### step wall (ms, per status)
| status | K1=4 | slope | mid-resid |
|---|---|---|---|
| hit_k1 | 114.49 | +nan | - |
| hit_k2 | 72.67 | +nan | - |
| miss | 83.56 | +nan | - |
| mixed | 122.63 | +nan | - |

### per-label mean per-step cuda_ms — status=hit_k1
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| child:target_send_request | 0.367 | +nan | - |
| child:target_recv_response_wait | 1.638 | +nan | - |
| child:target_response_received | 0.002 | +nan | - |
| target_spec_wait | 2.950 | +nan | - |
| verify_setup | 0.261 | +nan | - |
| graph_pre | 76.396 | +nan | - |
| exit_logits | 0.665 | +nan | - |
| proxy_compute_send | 0.672 | +nan | - |
| graph_post | 29.171 | +nan | - |
| final_logits | 0.571 | +nan | - |
| verify_sample_accept | 2.487 | +nan | - |
| target_postprocess | 0.081 | +nan | - |

### per-label mean per-step cuda_ms — status=hit_k2
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| child:target_send_request | 0.365 | +nan | - |
| child:target_recv_response_wait | 4.122 | +nan | - |
| child:target_response_received | 0.003 | +nan | - |
| target_spec_wait | 5.364 | +nan | - |
| verify_setup | 0.270 | +nan | - |
| graph_pre | 45.241 | +nan | - |
| exit_logits | 0.404 | +nan | - |
| proxy_compute_send | 0.410 | +nan | - |
| graph_post | 16.545 | +nan | - |
| final_logits | 0.418 | +nan | - |
| verify_sample_accept | 2.741 | +nan | - |
| target_postprocess | 0.071 | +nan | - |

### per-label mean per-step cuda_ms — status=miss
| label | K1=4 | slope ms/pos | mid-resid |
|---|---|---|---|
| child:target_send_request | 0.339 | +nan | - |
| child:target_recv_response_wait | 9.087 | +nan | - |
| child:target_response_received | 0.002 | +nan | - |
| target_spec_wait | 10.396 | +nan | - |
| verify_setup | 0.239 | +nan | - |
| graph_pre | 49.144 | +nan | - |
| exit_logits | 0.427 | +nan | - |
| proxy_compute_send | 0.414 | +nan | - |
| graph_post | 18.520 | +nan | - |
| final_logits | 0.434 | +nan | - |
| verify_sample_accept | 2.708 | +nan | - |
| target_postprocess | 0.129 | +nan | - |
