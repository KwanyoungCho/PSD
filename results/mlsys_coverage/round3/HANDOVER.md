# B>1 dynamic tree 실행 경로 인수인계

**Merge 전에 읽을 단일 기준 문서:** [MERGE_REVIEW.md](../MERGE_REVIEW.md). 전체 commit/파일 inventory와 연구 branch의 의미상 충돌 검토를 포함한다.

- 작업 checkout: `/home/chokwans99/PSD-mlsys-coverage`, branch `feat/duet-mlsys-coverage`.
- 논문 기준: `feat/duet-p2tree-g0@a82f7d2`. 원래 `/home/chokwans99/PSD`의 root-selection 연구 파일은 수정하지 않았다.
- Python: `/home/chokwans99/PSD/ssd/.venv/bin/python`. 공유 환경에 `uv sync`하지 않는다.
- 환경 버전: [ENVIRONMENT.json](ENVIRONMENT.json), 연결 관계: [GPU_TOPOLOGY.txt](GPU_TOPOLOGY.txt).
- 수치 원본 및 실패 기록: [NUMBERS.md](NUMBERS.md), [NUMBERS.json](NUMBERS.json), [RUN_INVENTORY.csv](RUN_INVENTORY.csv). `analyze.py`로 재집계한다.

## 이번에 연결한 경로

기존 `BatchedTreeExecutor`는 draft forward만 합치는 primitive였다. 이제 다음 전체 경로가 연결된다.

1. Target request는 sequence ID, 직전 terminal context, 실제 target recovery token을 보낸다.
2. Draft cache는 `(seq_id, terminal_context, recovery_token)`으로 조회한다. 요청 사이의 root/view/q 참조가 섞이지 않는다.
3. Miss 행만 기존 JIT로 처리한다. Hit 행은 해당 root의 tree 또는 chain을 보낸다. 혼합 배치가 가능하다.
4. 새 wire epoch2는 node마다 원래 parent-q logits를 전달한다. 같은 부모의 형제는 같은 분포/참조를 갖는다. Root 후보의 proxy 확률은 accept 분모가 아니다.
5. Target은 prefix+ancestor+self mask, depth에 맞는 RoPE, 요청별 physical KV pages로 배치 forward한다. Padding은 slot=-1이라 KV를 쓰지 않는다.
6. Early-exit logits로 기존 Policy-B proxy 점수를 계산해서 draft에 보낸다. 이후 target 나머지 layer와 draft P1/P2가 진행된다.
7. Stochastic 검증은 각 부모의 형제 순서를 유지한다. 기각할 때 target residual과 비복원 draft 조건부분포를 모두 갱신한다. 서로 다른 요청과 context의 연산은 병렬화한다. Greedy는 target argmax 경로를 따른다.
8. Target은 수락된 경로 KV를 gather한 뒤 canonical slots에 scatter한다. Draft는 tree glue KV를 sequence ID별로 보존하고 다음 request의 terminal과 실제 수락 길이에 따라 복원한다.
9. Re-prefill은 해당 sequence의 cache/staging만 무효화한다. 배치 row 번호를 identity로 사용하지 않는다.
10. P1/P2 forest는 요청마다 독립 top-W 예산과 RNG를 유지하고, 각 round의 model forward를 하나의 배치로 실행한다.

B>1 tree config에서 자동 선택한다. 한 번 선택된 경로는 활성 요청이 B=1로 줄어도 유지된다. 원래 max_num_seqs=1 실행은 기존 B1 경로를 유지한다. `SSD_BATCHED_TREE=1`은 B1에서도 새 경로를 강제로 검사할 때만 필요하다.

## 최적화와 계약

| 부분 | 변경 | 검사 |
|---|---|---|
| Attention scratch | 같은 compute stream의 forest/ancestor forward가 float workspace를 공유 | 실제 full model B16, 별도 dense-attention reference |
| Attention plans | geometry별 integer plan은 보존 | 다른 페이지/배치/너비의 plan을 잘못 공유하지 않음 |
| Glue width | 실제 valid 범위를 K2+1/K1+1/tree-width+1 버킷으로 묶음 | 요청별 mask와 padding slot 유지 |
| Target width | 작은 view 배치는 작은 query 버킷 사용 | `physical_verify_width` event 기록 |
| Root 준비 | 요청별 P1 계산을 배치 CUDA graph로 합침 | ragged per-request 기존 함수와 점수/토큰 비교 |
| Arena 입력 | mask/RoPE/slot/page 준비를 CUDA graph로 합침 | eager 입력과 동일성, padding/모델 한계 검사 |
| Target accept | 형제 residual ladder, 경로 선택, RNG/recovery를 graph로 합침 | 고정 coin reference, Monte Carlo, replay RNG 변화 |
| Forest workspace | arena별로 실제 사용하지 않는 개별 FlashInfer wrapper/float workspace 생성 제거 | 기존 primitive 회귀 + 통합 실행 |

WOR 후보가 floating-point softmax의 nonzero support를 소진했으면 뒤의 zero-mass 후보를 수락하지 않는다. T>0 target과 T=0 draft 조합은 별도 proposal 법칙이 필요하므로 명시적으로 거부한다.

새 경로는 dense full-vocabulary 모델용이다. EAGLE, raw proxy-on-draft, exit top-m gather는 명시적으로 거부한다. 이들을 이번 두 dense pair 실험으로 지원한다고 주장하지 않는다.

## 핵심 파일

- `ssd/ssd/engine/helpers/batch_tree_draft.py`: cache, 혼합 응답, KV 복원, phase orchestration.
- `batch_tree_common.py`: ancestor 입력과 restore 계획.
- `batch_tree_forward.py`: target split/draft glue의 batched ancestor attention.
- `packed_tree_forward.py`: 선택적 ragged target query, persistent FlashInfer plan 갱신.
- `batch_tree_sampling.py`: proxy, ordered sibling 검증, captured accept.
- `batch_tree_roots.py`: 기존 P1 점수의 batch 계산.
- `batch_tree_inputs.py`: arena 입력 준비 graph.
- `batch_tree_verify.py`: target wire 검사, proxy callback, 검증, KV commit, metrics.
- `batched_tree_executor.py`: 요청별 arena를 유지하는 round 단위 fused draft forward.
- `ssd/tests/test_batch_tree_serving.py`: 실제 mask/페이지 계산, sampling 보존, graph input/ranking/accept 검사.

## 재현

```bash
cd /home/chokwans99/PSD-mlsys-coverage
CUDA_VISIBLE_DEVICES=0 bash results/mlsys_coverage/run_regressions.sh
/home/chokwans99/PSD/ssd/.venv/bin/python ssd/bench/mlsys_campaign.py \
  --plan results/mlsys_coverage/round3/llama2_full_plan.json \
  --directory /새로운/결과/경로 --gpus 0,1 --port 33000 --timeout 2100
```

Campaign은 시작 시 GPU가 사용 중이면 실행하지 않는다. 결과 폴더를 재사용해 실패 기록을 덮어쓰지 않는다. 성공 출력뿐 아니라 로그, telemetry, campaign manifest를 같이 보관한다.

| 환경 변수 | 기본 | 비교 목적 |
|---|---:|---|
| `SSD_BATCHED_TREE` | 0 | max_num_seqs=1에서도 새 배치 경로 사용 |
| `SSD_BATCH_TREE_ACCEPT_GRAPH` | 1 | 0이면 eager 동일 검증식 |
| `SSD_BATCH_TREE_ROOT_GRAPH` | 1 | 0이면 per-request 기존 P1 준비 |
| `SSD_BATCH_TREE_INPUT_GRAPH` | 1 | 0이면 eager arena 입력 준비 |
| `SSD_BATCH_TREE_FIXED_VERIFY` | 0 | 1이면 최대 tree query 폭 고정 |
| `SSD_PACKED_TREE_VERIFY` | 0 | Packed query 실험용. 두 모델 TPS 악화로 기본값0 유지 |
| `SSD_BATCH_TREE_WORKSPACE_MB` | 128 | 같은 runner의 공유 FlashInfer float scratch |

`SSD_PACKED_VERIFY`와 `SSD_MIXED_MISS_AR`는 기존 chain 경로 최적화다. Tree 결과를 그 최적화 결과로 설명하지 않는다. 특히 miss를 AR로 바꾸면 AL 자체가 달라지므로, 이번 tree 비교에는 그런 변화를 섞지 않았다.

## 해석 시 주의

- Full model은 가중치를 모두 사용했다는 뜻이다. GPU는 RTX4090, 모델은 7B/8B다. 원 논문의 70B/Blackwell 결과를 재현한 것이 아니다.
- Full input 실험은 저장소 corpus의 480개 first turn 전부이며 input512/output64 cap이다. 긴 context 전체나 후속 turn 전체가 아니다.
- 한 engine에서 seed별 cell을 반복하면 target RNG만 reset된다. Draft RNG는 시작 seed 뒤 계속 진행한다. 독립 프로세스 3회 반복과 구분한다.
- 모델 로딩·engine 초기화·16-token warm-up은 TPS 계측 밖이다.
- CUDA graph는 geometry별로 처음 사용할 때 생성된다. 첫 반복의 capture 비용을 숨기지 말고 이후 반복과 함께 보고한다.
- Greedy token 완전 일치와 수학적 greedy rule 구현을 구분한다. 서로 다른 kernel/query shape의 near-tie 차이는 독립 HF logits로 분석한다.
- 낮은 메모리의 자연 workload는 preemption이 있어도 output_position=0일 수 있다. 이것만으로 생성 중 KV 복원을 검증했다고 하지 않는다. `boundary_prompts.json`은 page 경계 직전 240-token prefix와 EOS 무시 출력으로 생성 도중 preemption을 유도하는 별도 합성 진단이다.
- 경계 진단/부분 실패/smoke TPS를 논문 throughput 표에 섞지 않는다.

## 이 round의 최종 검증·실험을 다시 실행하려면

- 최종 회귀: `regressions_final.log`, **280 tests passed**, 85.124초. 이 중 새 serving 검사는17개다.
- 기본 full tree/chain: `llama2_full_plan.json`, `llama3_full_plan.json` (B8, T0/.7, 480 first turns, target seed3개).
- P2-only: `llama2_p2_plan.json`, `llama3_p2_plan.json` (B8/16, T0/.7, 480 first turns).
- 동일 GPU 순차 최적화 비교: `comparison_plan.json` (graph helpers off/on 및 packed chain, 두 모델).
- Packed tree 추가 비교: `packed_comparison_plan.json`. 기본 tree 비교가 끝난 다음 같은 GPU에서 실행한다. T0.7 뒤 T0를 각각2 pass 실행한다.
- GPU 배치 확인: `placement_plan.json`, GPU4,5의 PIX pair에서 Llama3 optimized tree/chain을 재실행한다. GPU0,1의 NODE pair와 다른 배치이며 동시 실행하지 않는다.
- Packed greedy HF 진단: `CUDA_VISIBLE_DEVICES=2 python results/mlsys_coverage/round3/audit_packed.py`; 전체480개 출력에서 기존 AR48개와 정확히 같은 prompt만 뽑는다. 추출 파일에는 성능 수치를 넣지 않는다. 이어서 기존 full tree480개와 다른 모든 첫 분기 지점도 HF로 평가한다.
- Ragged 출력/B3: `llama{2,3}_edge_plan.json`.
- 생성 중 preemption: `llama{2,3}_boundary_plan.json` + `ssd/bench/mlsys_preemption_audit.py`. 자연 workload `preempt1`만으로 생성 이후 복원을 인증하지 않는다.
- Target TP2: `tp2_plan.json`, target2+draft1 GPU, B2/T0/.7. 성능 최종표가 아니라 통합 검사다.
- 집계: `python results/mlsys_coverage/round3/analyze.py`, `python results/mlsys_coverage/round3/summarize_final.py`.

Plan의 target/draft/prompts는 현재 서버의 절대 경로다. 새 서버에서 경로와 CUDA arch만 해당 환경에 맞춰 변경하고, 나머지 인자·환경은 우선 유지한다. CUDA arch는 4090의8.9를 Blackwell에 그대로 적용하지 않는다.

### 검증 cap과 proposal 법칙

새 배치 경로에서 G>M이면 생성 순서의 앞 M개를 유지한다. Sampled token의 confidence를 보고 사후 선택하면, ancestry와 sibling 순서를 보존해도 그 선택으로 조건부 proposal이 바뀔 수 있다. 그러므로 원래 parent-q를 분모로 사용하는 현재 verifier에서는 그런 rerank를 하지 않는다. 이번 모든 성능 실험은 G=M이므로 결과에 변화가 없다. 기존 maxB1 legacy rerank의 G>M 조합은 이 round의 losslessness 검증 범위가 아니다.

### Packed tree query 구현의 재검증 지점

`SSD_PACKED_TREE_VERIFY=1`은 target 쪽에만 적용된다. Draft glue/forest와 검증식은 유지한다. Query 경계가 바뀔 때 FlashInfer plan을 persistent integer buffers에 다시 쓰고 `_plan_info` 구조 동일성을 검사한다. 라이브러리 내부 `_mask_indptr_buf`의 byte-offset 보정에 의존하므로 버전 업그레이드 후 반드시 `PackedForwardAttention` 및 MHA/GQA probe를 먼저 실행한다. 정렬용 padding은 별도 가상 sequence이며 KV slot=-1이다.

### 코드 변경 순서

| Commit | 내용 |
|---|---|
| `7af3f5c` | B>1 tree serving 전체 연결 |
| `33dda1e` | 공유 workspace, root/accept graph |
| `72475b8` | arena 입력 graph |
| `d7fea9f` | target query 너비 버킷 |
| `c8f7b94` | 설정한 sibling cap 반영, 고정 coin 경로 검사 |
| `800f011` | support가 소진된 WOR padding 방어 |
| `225d1ab` | mixed T0 row의 deterministic tie |
| `30d8851` | opt-in packed tree target query |
| `b90f118` | G>M에서 생성 prefix 보존, seq_id/cap 회귀 검사 |

Raw run은 당시 commit과 diff hash를 가진다. 실험 도중 추가된 후속 변경이 모두 소급 적용됐다고 해석하지 않는다. 예를 들어 초기 Llama2 full tree는 fixed target width, 후속 순차 비교는 short width 경로다.

## 완료 상태

- 요청한 B>1 tree serving 통합, 두 full-model pair, T0/T0.7, B8 전체 입력 및 P2-only B16 검증을 완료했다.
- 마지막 comparison6 jobs, packed comparison2 jobs, placement2 jobs 및 두 모델 HF audits가 모두 종료됐다. 실행을 남겨둔 background experiment는 없다.
- 전체 회귀280개 통과. Raw campaign34 jobs 중30개 성공, 초기4개 실패 보존. 완료된480-input cell64개.
- 기본 권장: batch tree 자동 경로 + root/input/accept graph ON, `SSD_PACKED_TREE_VERIFY=0`. Packed는 두 모델에서 TPS 이득이 없었다.
- GPU4·5를 쓰는 것만으로 높은 TPS가 재현되지는 않았다. Pair/실행 상태가 다른 예전 wave의 수치를 현재 안정적인 성능으로 사용하지 않는다.
- Tree AL은 chain보다 높지만, 이번 동일 GPU 대조에서 chain이 더 빠르다. 기능 통합·검증 완료를 tree TPS 우위 달성으로 표현하지 않는다.
- 원래 research checkout은 수정하지 않았다. 이 turn의 변경은 별도 worktree에 로컬 커밋하며 자동으로 push하지 않는다.
