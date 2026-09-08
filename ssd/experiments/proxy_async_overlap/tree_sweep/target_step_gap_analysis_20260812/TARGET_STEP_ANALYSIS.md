# DUET tree target-step latency analysis

Date: 2026-08-12

> Final decomposition: `FINAL_CAUSE_ANALYSIS.md`. The final report adds the
> repeated chain/P2-tree/full-tree triad, equal-target-row controls, and a
> profiler-off validation. Its conclusion supersedes the preliminary
> recommended-order section below.

## Conclusion

Tree의 target-step 증가를 target verification 계산 증가만으로 설명할 수 없다.
현재 metric에서 `target step`은 순수 target 시간이 아니라 draft 응답 대기부터
target verify와 scheduler postprocess까지 포함한 **전체 speculative decode-step
wall time**이다. Full 결과에서 chain 대비 tree가 증가한 약 6.00 ms 중 target
verify 증가는 2.26 ms이고, 나머지 약 3.74 ms는 verify 바깥이다.

동일한 7-prompt profile의 상태별 비교에서는 tree hit 한 step의 증가가 다음처럼
분해됐다.

| Hit source | Full-step increase | Draft response wait | Response→verify gap | Target verify | Post-verify |
|---|---:|---:|---:|---:|---:|
| P1 | +5.986 ms | +2.769 ms | +1.436 ms | +1.799 ms | -0.018 ms |
| P2 | +4.515 ms | +1.865 ms | +1.037 ms | +1.615 ms | -0.003 ms |
| Miss | -0.675 ms | -0.649 ms | +0.038 ms | -0.053 ms | -0.011 ms |

Miss 경로는 사실상 같고 tree hit에서만 증가하므로, 일반적인 GPU 변동이나 target
모델 전체의 일괄 slowdown이 아니라 tree response/metadata 경로의 추가 작업이다.

## Metric boundary

- `target_step_times`: `LLMEngine.generate()`가 `self.step()` 전체를 감싼 wall time.
  Draft request/response, target verify, state restore, scheduler postprocess를 모두 포함한다.
- `target_verify_times`: `Verifier.verify()`가 target `model_runner.call("run", ...)`
  직전부터 acceptance/tree walk와 target KV commit 직후까지만 측정한다.
- Tree wire 해석과 target proxy topology 준비는 `target_verify_times` 시작 전에
  실행되므로 `target step - target verify`에 들어간다.

따라서 report의 `Target step`은 의미상 `Full decode-step latency`에 가깝다.

## Full-run evidence

현재 paper-compatible question 집계에 쓰인 raw의 verification-step-weighted latency다.
두 raw의 coverage가 달라 이 표만으로 원인을 확정하지 않고, 아래 matched diagnostics로
확인했다.

| Config | Coverage | Target step | Target verify | Outside verify |
|---|---:|---:|---:|---:|
| DUET-chain K1=8,K2=4 | 462 questions / 542 turns | 65.028 ms | 60.692 ms | 4.336 ms |
| DUET-tree seed42 | 480 questions / 560 turns | 71.028 ms | 62.950 ms | 8.078 ms |
| Difference | — | +6.000 ms | +2.258 ms | +3.742 ms |

## Controlled diagnostics

### Current P2-tree-only vs P1+P2-tree, same smoke21

이 비교는 같은 current engine, seed42, 21 turns, output 1,024이며 P2 tree는 양쪽에
공통이다. 따라서 P1 tree를 추가한 영향에 가깝다.

| Config | Target step | Target verify | Outside verify |
|---|---:|---:|---:|
| P2-tree-only | 67.092 ms | 61.637 ms | 5.456 ms |
| P1+P2-tree | 71.366 ms | 63.511 ms | 7.855 ms |
| Difference | +4.274 ms | +1.874 ms | +2.399 ms |

### Detailed profile pair, same tiny7/output256

Chain profile은 이번 분석에서 새로 측정했고, tree는 동일 code/config로 2026-08-11에
측정한 profile을 사용했다. 두 실험 모두 같은 seven prompts와 seed42다. 출력과 hit
mix는 stochastic하므로 overall TPS 비교가 아니라 같은 hit source의 span 위치를
찾는 용도다.

| Config | Target step | Target verify | Outside verify | Verify steps |
|---|---:|---:|---:|---:|
| Chain | 67.681 ms | 60.986 ms | 6.695 ms | 393 |
| P1+P2-tree | 72.233 ms | 62.005 ms | 10.228 ms | 436 |
| Difference | +4.552 ms | +1.020 ms | +3.533 ms | — |

## Where the extra time is spent

### 1. Draft-side tree hit response

Target의 `target_spec_wait`는 cache hit에서도 draft가 응답을 만들고 전송할 때까지
막혀 있다. 평균 hit response span은 다음과 같다.

| Draft span | Chain | Tree | Difference |
|---|---:|---:|---:|
| `draft_recv_request` (all statuses) | 0.358 ms | 0.761 ms | +0.403 ms |
| `hit_cache_respond_hit_k1` | 0.941 ms | 2.778 ms | +1.837 ms |
| `hit_cache_respond_hit_k2` | 0.945 ms | 1.366 ms | +0.421 ms |

Tree hit response에는 chain cache row copy에 없는 다음 작업이 있다.

- P1 generated tree 14 nodes를 M1=12 closure-valid subtree로 rerank한다.
- topology를 GPU에서 CPU로 복사해 parse/validate하고 여러 작은 GPU buffer를 다시 채운다.
- hit root의 full-vocabulary parent-q rows를 gather한다.
- P1은 최대 12 parent-q rows를 보내므로 chain의 fixed 8 q rows보다 payload도 크다.
- 이전 tree에서 수락한 경로의 draft KV를 다음 request 시작에서 canonical slot으로
  복원한다. 이 작업은 `draft_recv_request` 안에 들어간다.

특히 `N1=14, M1=12`의 on-hit rerank가 P1 response critical path에 있다. `N1=M1=12`
이면 equal-limit fast path를 타므로 target verify cap을 바꾸지 않고 이 rerank를 없앨
수 있다.

### 2. Target-side response-to-verify preparation

Tree response를 받은 후 `target_verify_times`가 시작되기 전에 다음을 수행한다.

- `tree_ints` GPU→CPU `.tolist()`와 topology parse/validation
- phase/valid-node 확인
- target tree-proxy CUDA graph의 parent/sibling topology pack과 작은 H2D copies
- parent-q reference gather 및 tree-specific proxy closure 준비

그 결과 response→verify gap은 P1 hit에서 1.300→2.736 ms(+1.436), P2 hit에서
1.294→2.331 ms(+1.037)로 증가했다. Miss는 1.064→1.102 ms로 동일하다.

### 3. Target tree verification itself

Verify 내부에도 실제 증가는 있다.

- Chain P1/P2 verify는 각각 9/5 rows이고, tree cap은 각각 13/9 rows
  (`M+1`, recovery row 포함)다.
- Tree custom attention을 위해 ancestry 복원, packed mask 작성, padded input copy,
  FlashInfer attention buffer update가 필요하다.
- Profile의 `verify_setup`은 chain hit 약 0.36 ms에서 tree hit 0.97–1.04 ms로
  약 0.61–0.68 ms 증가했다.
- 더 많은 rows 때문에 graph pre/post도 증가한다. 다만 전체 full-run에서 verify
  증가는 2.26 ms로, total step 증가의 절반보다 작다.

Postprocess는 약 0.06–0.07 ms이며 원인이 아니다.

## Repeated latency-only A/B: N1=14/M1=12 vs N1=12/M1=12

위 분석에서 가장 먼저 지목한 P1 hit-time rerank를 같은 7개 입력, seed42,
output 256에서 각 설정 3회씩 교차 실행했다. 두 arm 모두 M1=12이므로 target
verify 최대 폭은 고정하고, N1=14에서 생성한 tree를 12개로 다시 고르는 경로만
제거하는 진단이다. 프로파일 warm-up은 raw의 실제 step 수를 이용해 제외했다.

| Metric | 14/12 | 12/12 | Paired delta (12−14) |
|---|---:|---:|---:|
| P1 hit rerank | 1.290 ± 0.158 ms | 0.376 ± 0.094 ms | **-0.914 ± 0.250 ms** |
| P1 cache-hit response | 2.029 ± 0.259 ms | 1.138 ± 0.294 ms | **-0.891 ± 0.545 ms** |
| P1 draft/spec wait | 4.089 ± 0.345 ms | 3.130 ± 0.431 ms | **-0.959 ± 0.762 ms** |
| P1 full step | 70.585 ± 0.468 ms | 69.360 ± 0.269 ms | **-1.225 ± 0.714 ms** |
| P2 full step (control) | 66.975 ± 0.298 ms | 66.739 ± 0.328 ms | -0.236 ± 0.616 ms |
| Miss full step (control) | 72.017 ± 0.373 ms | 72.027 ± 0.527 ms | +0.010 ± 0.900 ms |

P1 rerank 감소량과 P1 cache-hit response 감소량이 거의 일치하고, P2 response와
KV restore는 변하지 않았다. 따라서 `N1=14 → M1=12` subtree 재선택/compaction이
P1 critical path에 약 0.9 ms를 더한다는 원인은 확인됐다. `N1=M1=12`는 P1
hit당 약 1 ms를 회수하는 유효한 후보이다.

다만 두 arm은 N1 변경으로 생성 token 경로가 달라져 각각 436/470 step이었고,
overall target step delta는 -0.531 ± 0.863 ms로 분산보다 작다. 이 실험은 AL/hit
우열이나 paper TPS를 판단하는 실험이 아니라, 동일 코드 span과 hit-source 조건부
latency로 원인을 찾는 실험이다.

Rerank를 제거해도 tree 공통 비용은 남는다. P1 hit 기준 target-side
wire parse/validation은 약 0.67 ms, topology pack/H2D 준비는 약 1.00 ms,
parent-q select는 약 0.14 ms이고 target verify setup은 약 1.11 ms다. 따라서
rerank 하나만으로 chain 대비 4–6 ms 전체 차이를 없앨 수는 없다.

## Exact metadata path and remaining latency targets

현재 tree metadata는 작지만(최대 수십 개 정수) 여러 번 CPU/GPU 경계를 지난다.

1. Draft의 `_rerank_tree_hit_view()`가 GPU view를 `pack_tree_ints()`로 묶고
   `.cpu()`로 복사해 parse/validate한다. `14/12`는 생성 tree와 compact tree에
   이 작업을 두 번 수행하고, `12/12`는 한 번 수행한다. 위 A/B가 이 차이를
   직접 측정했다.
2. Metadata는 fused NCCL response의 GPU tensor로 target에 도착한다.
   `Verifier.verify()`는 이를 `.tolist()`로 CPU에 읽고, 다시
   `torch.tensor(list)`를 만든 뒤 parse/validate한다. list를 다시 tensor로
   감싸는 부분은 불필요한 CPU allocation이다.
3. `TreeProxyCUDAGraph.prepare_topology()`는 parent/sibling list에서 매 hit마다
   CPU tensor 5개(`child`, `child_valid`, `par`, `sib`, `node_valid`)를 새로 만들고,
   각각 `.to(device)`한 뒤 persistent graph buffer로 `copy_`한다. 측정된 약
   1.00 ms는 작은 데이터 크기가 아니라 Python allocation과 5개의 작은 H2D
   dispatch가 지배하는 형태다.
4. 동일 CPU list는 SHM으로 target TP rank들에 전달되고, 각 rank의
   `_run_tree_verify()`가 다시 parse/validate와 depth/ancestor reconstruction을
   한다. P1 hit의 verify setup 1.126 ms는 meta CPU 0.204 ms, mask 준비 0.454 ms,
   input copy 0.111 ms, attention-buffer update 0.169 ms 등을 포함한다.
5. Target forward 뒤 acceptance walk에서도 metadata를 다시 CPU로 읽고 tree
   invariant를 재검사한다. 안전성 검사는 필요하지만, 동일 immutable wire를 여러
   단계에서 반복 검증하는 현재 구조는 latency 관점에서 통합 여지가 있다.

따라서 다음 최적화는 tree 선택 정책이나 hit/AL을 바꾸지 않고 다음 순서가 적절하다.

- rank0에서 이미 얻은 CPU list를 tensor로 재생성하지 않고 직접 parse한다.
- topology용 persistent packed buffer를 두고 단일 H2D copy 또는 GPU-side pack으로
  5개의 작은 전송을 합친다.
- wire validation 결과를 immutable/validated 상태로 전달해 동일 프로세스 내 중복
  검사를 줄이되, TP rank가 모두 같은 metadata를 검사하고 함께 실패한다는 현재
  deadlock-safety 계약은 보존한다.
- tree mask/depth도 topology와 함께 한 번만 pack해 verify setup에서 반복 생성하지
  않도록 한다.

## Recommended order before the full sweep

1. `N1/M1=12/12` 소규모 반복 진단으로 P1 hit당 약 1 ms 절감을 확인했다.
2. 다음 소규모 latency 진단/최적화는 tree metadata의 중복 CPU round trip과 여러
   작은 H2D topology copy를 persistent packed GPU buffer로 합치는 경로를 대상으로
   한다. 이는 hit/AL 정책을 바꾸지 않는 순수 실행경로 최적화다.
3. 그 다음 profile-off smoke/full-data에서 `14/12`와 `12/12`의 실제 decode TPS와
   P1 AL 손실을 비교해 최종 paper config를 결정한다.
4. 이후에만 C와 threshold full sweep을 재개한다. 현재 상태에서 C/N을 넓히면
   draft-side tree critical path가 더 커져 AL 개선을 TPS가 상쇄할 수 있다.

## Artifacts

- New chain raw: `chain_s42_o256.jsonl`
- New chain log: `chain_s42_o256.log`
- New chain target/draft profiles: `chain_profile/`
- Repeated rerank A/B report: `rerank_ab/LATENCY_AB.md`
- Repeated rerank A/B per-run values: `rerank_ab/run_summary.csv`
- Repeated rerank A/B runner: `run_rerank_latency_ab.sh`
- Repeated rerank A/B analyzer: `analyze_rerank_latency_ab.py`
- Tree comparison profile: `../p1_tree_full_backbone_profile_20260811/p1_backbone_profile/`
- Full sweep document: `../p1_p2_tree_full_local_sweep_seed42_20260812/SWEEP_RESULTS.md`
