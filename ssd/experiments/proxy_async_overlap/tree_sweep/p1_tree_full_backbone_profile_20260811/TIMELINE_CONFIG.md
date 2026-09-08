# P1–P2 tree timeline and configuration

## Outputs

- Paper-oriented overview: [`p1_backbone_profile/timeline_p1_p2_tree_overview.png`](p1_backbone_profile/timeline_p1_p2_tree_overview.png)
- Vector version: [`p1_backbone_profile/timeline_p1_p2_tree_overview.pdf`](p1_backbone_profile/timeline_p1_p2_tree_overview.pdf)
- Exact plotted intervals: [`p1_backbone_profile/timeline_p1_p2_tree_overview.csv`](p1_backbone_profile/timeline_p1_p2_tree_overview.csv)
- Detailed P1-hit trace: [`p1_backbone_profile/timeline_cache_hit_k1.png`](p1_backbone_profile/timeline_cache_hit_k1.png)
- Detailed P2-hit trace: [`p1_backbone_profile/timeline_cache_hit_k2.png`](p1_backbone_profile/timeline_cache_hit_k2.png)
- Detailed miss trace: [`p1_backbone_profile/timeline_cache_miss.png`](p1_backbone_profile/timeline_cache_miss.png)

The overview groups low-level kernels into algorithmic stages. The three detailed
plots retain every profiled event. The trace contains two warmup `generate()` calls
followed by seven measured prompts. The overview excludes the warmups and chooses the
median target-step duration independently for P1 hit, P2 hit, and miss.

| Case | Request epoch / step | Target step | Cache wait | Draft clock correction |
|---|---:|---:|---:|---:|
| P1 hit | 8 / 52 | 70.984 ms | 5.023 ms | 0.104 ms |
| P2 hit | 4 / 8 | 66.822 ms | 3.857 ms | 0.044 ms |
| Miss | 8 / 18 | 73.203 ms | 13.328 ms | 0.124 ms |

These are representative individual steps, not throughput averages. The small draft
clock correction is the median response-causality offset around the selected step; it
only aligns the two GPU clocks in the drawing and does not alter measured durations.

## How to read the timeline

One plotted row is one speculative step, but two logical activities overlap in it.

1. The target asks the draft for the proposal needed **now**. `Cache wait` and
   `Proposal response` show whether that request hits P1, hits P2, or requires a miss
   response.
2. After responding, the draft runs `Glue` to connect the accepted path to the next
   candidate contexts.
3. The draft constructs the next step's P1 forest **before** receiving the target's
   early-exit proxy. At the same time, the target runs `Target pre-exit`.
4. At exit layer 56, the target computes and asynchronously sends the proxy. The draft
   normally reaches `Proxy wait` first, so this wait is intentional overlap slack.
5. After the proxy arrives, the draft constructs the proxy-guided P2 tree while the
   target continues `Target post-exit` and verification.
6. The P1/P2 cache produced on the draft side is for a future request. Therefore the
   status in a panel describes the proposal consumed at the start of that panel, not
   the P1/P2 tree being generated later in the same panel.

The selected profile's aggregate balance diagnostic gives a median P1-ready-to-proxy
gap of **+8.60 ms** and a median P2-cache-ready-to-next-target-request gap of
**+1.09 ms**. Positive means the draft finishes first. The median per-round graph times
are 3.74 ms for P1 and 3.55 ms for P2. Profiling is diagnostic and was disabled for the
reported TPS experiment.

## Algorithm configuration

The timeline uses the selected full-backbone P1+P2-tree configuration. It is the same
algorithm configuration as the full Spec-Bench run; only the profile run used output
256 and profiling, whereas the paper experiment used output 1024 with profiling off.

### Split and proxy parameters

| Variable | Value | Meaning |
|---|---:|---|
| `K1` / `--k1` | 8 | Number of P1 draft rounds before the proxy is available. It is the long, proxy-independent horizon. |
| `K2` / `--k2` | 4 | Number of proxy-guided P2 draft rounds after proxy arrival. `K2 <= K1`. The ordinary speculative width is `K1+K2=12`. |
| `exit_layer` | 56 | Target early-exit point. The target produces proxy information here and continues the remaining target layers concurrently with proxy handling. |
| `p1_fanout` | 3 | Uniform P1 chain-compatible layout width per one of the `K1+1=9` positions; its long-layout sum is 27. In tree mode this is the base compute/layout budget, not the semantic root count. |
| `p2_budget` / `W` | 15 | Physical P2 parent lanes evaluated by one draft forward. It is compute width, not the number of P2 roots retained. |
| `proxy_top_k` | 28 | Early-exit vocabulary candidate width retained for proxy/root selection. It limits the candidate pool but is not itself a cache-root budget. It is fixed across comparison arms. |

### Tree topology and budgets

| Variable | Value | Meaning |
|---|---:|---|
| `p1_tree` | `on` | Replaces the P1 chain cache entry with a dynamic tree/forest. |
| `p2_tree` | `on` | Builds a proxy-guided dynamic P2 tree after the early-exit proxy arrives. |
| `p1_allocation_policy` | `backbone` | Reserves a continuation lane for every P1 root in every round. This full-root guarantee matters because P1 does not yet know which root the future request will hit. |
| `roots_per_position` / `U1` | 3 | Number of P1 starting root tokens made for each glue context. This is distinct from `p1_fanout`, although both are 3 here. With at most `M1+1=13` contexts, the largest first round has up to 39 roots. |
| `root_count` / `R` | 10 | Number of meaningful P2 roots evaluated in round 0 and retained as cache keys. `R=10` is smaller than the physical width `W=15`. |
| `C` / `c_tensor` | 2 | Maximum ordered, without-replacement child candidates sampled from one selected parent. `C` is branch width, not tree depth. |
| `N1` | 14 | Maximum P1 nodes generated and cached **per P1 root**. It is the search budget. |
| `M1` | 12 | Maximum P1 nodes reranked and sent to the target on a hit. The selected nodes preserve ancestors and earlier ordered siblings, so target verification remains lossless. At most 12 tree nodes plus the recovery context are verified. |
| `N2` | 8 | Maximum P2 nodes generated and cached per P2 root. |
| `M2` | 8 | Maximum P2 nodes sent on a hit. Here `M2=N2`, so P2 does not shrink the generated tree before transfer. |

`N` and `M` are deliberately separate: draft search may generate a wider tree (`N`),
then transmit only a closure-valid high-confidence subtree (`M`) to reduce target
verification cost. The full-backbone policy affects P1 root coverage; `M1=12` limits
the selected hit tree afterward.

### Expansion thresholds

| Variable | Value | Meaning |
|---|---:|---|
| `p1_start_threshold` | 0 | Minimum P1 `context reach × root probability` for expansion after round 0. Zero disables this pruning. |
| `p1_conf_threshold` | 0 | Minimum local draft confidence for deeper P1 expansion after round 0. Zero disables this pruning. |
| `p2_proxy_threshold` | 0.01 | Minimum calibrated target-proxy root score for deeper P2 expansion after round 0. |
| `p2_conf_threshold` | 0.01 | Minimum local draft confidence for deeper P2 expansion after round 0. |

Thresholds do not remove round-0 roots or already sampled leaves. They only decide
whether a node is expanded in a later round. Thus the P1 setting uses no threshold
pruning, while P2 applies mild proxy and draft-confidence pruning.

## Model, sampling, and length configuration

| Variable | Value | Meaning |
|---|---:|---|
| Target | `facebook/layerskip-llama2-70B` | Target/verifier model. |
| Draft | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Draft model on its own GPU. |
| `gpus` | 3 | Two target tensor-parallel ranks plus one draft rank in this run. |
| `temperature` | 0.7 | Stochastic target/draft sampling temperature used by the experiment. |
| `top_p` | 1.0 | No nucleus truncation beyond the other candidate controls. |
| `seed` | 42 | Sampler seed. Fixed seed improves pairing, but GPU execution at nonzero temperature should not be treated as bitwise deterministic across independent runs. |
| `max_new_tokens` | 1024 paper / 256 profile | Paper evaluation output cap and shorter diagnostic profile cap. Per-step topology is unchanged. |
| `max_model_len` | 4096 | Maximum prompt plus generated context admitted by the engine. |
| `extend_draft_rope` | on | Extends TinyLlama's RoPE cache from 2048 to 4096 to prevent long-context indexing failure. Quality beyond its training window is not guaranteed. |
| `warmup` | 2 | Two requests run before measured/profiled benchmark prompts; they are excluded from the overview selection. |
| template | `raw` | Uses Spec-Bench text without adding a chat template. |

## Execution switches

These switches change implementation/overlap, not the tree scoring rule.

| Variable | Value | Meaning |
|---|---:|---|
| `SSD_TREE_EXEC` | 1 | Uses the captured GPU tree executor instead of the eager diagnostic path. |
| `SSD_TREE_ARENA` | 1 | Uses the tensor arena for topology and node state. |
| `SSD_TREE_PROXY_GRAPH` | 1 | Captures target-side proxy computation for tree hits. |
| `SSD_CHAIN_PROXY_GRAPH` | 1 | Captures the chain/miss proxy path as well; misses have no incoming tree topology. |
| `SSD_TREE_EXEC_WARMUP` | `all` | Captures all reachable tree-executor page buckets before serving, avoiding first-request capture stalls. |
| `SSD_DUET_EXIT_REPLICA` | 1 | Keeps an LM-head replica on target rank 0 so proxy computation avoids a target-TP collective and overlaps with post-exit layers. |
| `SSD_ASYNC_PROXY_SEND` | 1 | Sends proxy payloads through the persistent asynchronous send ring. |
| `SSD_PROXY_STREAM` | 0 | Does not use the older separate proxy-stream option; exit-replica plus async send provides the selected overlap path. |
| Tree verify workspace | 224 MiB | Preallocated target tree-verification workspace; a capacity setting, not a search budget. |
| P1/P2 executor workspace | 128 MiB each | Preallocated draft executor workspace; also not a tree-node budget. |
| `SSD_PROFILE_DUET` | 1 profile / 0 TPS | Records the aligned target/draft spans for this figure. It stays off in throughput measurements because profiling adds overhead. |
| `SSD_PROFILE_DUET_DETAIL` | 1 profile | Adds the low-level child spans shown in the detailed plots. |

## Recreating the overview

From `PSD/ssd`:

```bash
uv run --with matplotlib --no-project python \
  tools/duet_timeline/plot_p1_p2_overview.py \
  experiments/proxy_async_overlap/tree_sweep/\
p1_tree_full_backbone_profile_20260811/p1_backbone_profile \
  --skip-request-epochs 2
```

The aligned plotter now tags all events with `(request_epoch, step_id)`. This is
necessary because every `generate()` call restarts `step_id` at one; grouping only by
`step_id` would incorrectly merge several prompts into a tens-of-seconds timeline.
