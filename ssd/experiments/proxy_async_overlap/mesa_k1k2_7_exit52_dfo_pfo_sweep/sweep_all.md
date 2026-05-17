# MESA K1=K2=7 exit=52 sweep — dfo × pfo grid (PROFILE_MESA=0)

70B AWQ TP=4 + TinyLlama-1.1B AWQ TP=1, ns=50 in=512 out=512, seed=42, temp=0.7, --k 14 --mesa_phase1_k 7 --mesa_phase2_k 7, --mesa_exit_layer 52, SSD_FORCE_SPLIT_K1K2=1, SSD_PROFILE_MESA=0.

## decode_tps (tok/s)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |  77.94 |  71.14 |   —    |
| dfo=3 |  75.38 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## target_full_step_ms

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |  54.52 |  59.14 |   —    |
| dfo=3 |  54.81 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## avg_tokens_per_step (incl recovery)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |  4.15 |  4.11 |   —    |
| dfo=3 |  4.04 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## accept_fraction

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 | 0.450 | 0.440 |   —    |
| dfo=3 | 0.430 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## cache_hit_rate (Avg Cache Hits)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 | 0.800 | 0.820 |   —    |
| dfo=3 | 0.820 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## p1_hit (Phase 1 — draft-sourced hit rate)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |   —    |   —    |   —    |
| dfo=3 |   —    |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## p2_hit (Phase 2 — proxy-sourced hit rate)

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |   —    |   —    |   —    |
| dfo=3 |   —    |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## p1_avg_accepted_len

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |   —    |   —    |   —    |
| dfo=3 |   —    |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## p2_avg_accepted_len

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |   —    |   —    |   —    |
| dfo=3 |   —    |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## draft_step_ms

| dfo \ pfo | pfo=1 | pfo=2 | pfo=3 |
|---|---:|---:|---:|
| dfo=2 |  51.42 |  55.63 |   —    |
| dfo=3 |  51.71 |   —    |   —    |
| dfo=4 |   —    |   —    |   —    |
| dfo=5 |   —    |   —    |   —    |


## Best decode_tps
  dfo=2 pfo=1 → **77.94 tok/s**


## Failed runs

- dfo=2 pfo=3: decode_tps not parsed
- dfo=3 pfo=2: decode_tps not parsed
- dfo=3 pfo=3: decode_tps not parsed
- dfo=4 pfo=1: decode_tps not parsed
- dfo=4 pfo=2: decode_tps not parsed
- dfo=4 pfo=3: decode_tps not parsed
- dfo=5 pfo=1: decode_tps not parsed
- dfo=5 pfo=2: decode_tps not parsed
- dfo=5 pfo=3: decode_tps not parsed