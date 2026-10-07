# 2차 최적화 인수인계

**후속 완료 기록:** 이 문서에 남아 있던 B>1 tree serving 작업은 [../round3/REPORT.md](../round3/REPORT.md)와 [../round3/HANDOVER.md](../round3/HANDOVER.md)에 이어서 기록했다. 아래는 2차 종료 시점의 상태다.

## 가장 먼저 확인

- [REPORT.md](REPORT.md): 범위 정정과 완료/미완료 구분.
- [NUMBERS.md](NUMBERS.md), [NUMBERS.json](NUMBERS.json): raw에서 재집계. `python results/mlsys_coverage/round2/analyze.py`.
- **아직 B>1 dynamic tree end-to-end는 지원하지 않는다.** `BatchedTreeExecutor`만 보고 전체 기능이 완성됐다고 설명하면 안 된다.
- 원본 논문 4.3절 식 (4)의 dynamic tree가 사용자 요청 범위에 포함된다. 이전 server guide의 chain champion을 범위 축소의 근거로 사용하지 않는다.

## Branch / 환경

- Worktree `/home/chokwans99/PSD-mlsys-coverage`, branch `feat/duet-mlsys-coverage`.
- Paper base `a82f7d2` 고정. 원본 연구 checkout `/home/chokwans99/PSD`는 수정하지 않았다.
- Python `/home/chokwans99/PSD/ssd/.venv/bin/python`. 공유 venv에 `uv sync`하지 않는다.
- 이번 실행은 RTX4090, dense target TP1 + draft1, 총 2 GPU. 원 논문의 70B/TP2/Blackwell 성능을 대신하지 않는다.
- local commits: `721df75` packed/mixed; `0974b14` greedy tree/Qwama workspace 및 round generator 분리. `14b4b81` experimental batch tree primitive/diagnostics; `6c1df2f` opt-in host topology 및 regression 확장; `893a6b5` 독립 HF/다양한 prefix/layer 추적 진단. 이 요청에서 원격 push는 하지 않았다.

## 재현 명령

```bash
cd /home/chokwans99/PSD-mlsys-coverage
CUDA_VISIBLE_DEVICES=0 bash results/mlsys_coverage/run_regressions.sh
/home/chokwans99/PSD/ssd/.venv/bin/python ssd/bench/mlsys_campaign.py \
  --plan results/mlsys_coverage/round2/llama2_full_plan.json \
  --directory /새로운/결과/경로 --gpus 0,1 --port 29200 --timeout 900
```

Campaign은 사용 중인 GPU에서 시작하지 않으며 실행 중 foreign PID를 기록한다. 실패 폴더에 덮어쓰지 말고 새 폴더를 쓴다. `campaign.json`, `plan.json`, 로그, raw JSON, telemetry를 함께 보존한다.

구 Qwama 초기화 실패2개의 raw manifest에는 마지막 결과 상태 `running`이 남아 있지만 `returncode=1`과 종료 시간이 기록돼 있다. 실제 진행 중인 작업이 아니다. `RUN_INVENTORY.csv`는 원본 상태를 `recorded_status`로 보존하고 종료 코드 기준으로 실패를 표시한다. Campaign runner도 이후 실행부터 원본 `result_status`와 종료 후 `status`를 구분하도록 수정했다.

주요 환경 변수:

| 변수 | 동작 |
|---|---|
| `SSD_PACKED_VERIFY=1` | B>1 chain의 ragged query packing. 현재 SGL만 지원 |
| `SSD_MIXED_MISS_AR=1` | mixed hit/miss batch의 miss 행을 proposal 없이 target 직접 샘플링 |
| `SSD_TREE_HOST_TOPOLOGY=1` | B1 tree glue에서 CPU parent/length 재사용. 960/960 출력 동일하지만 TPS 개선 미재현으로 기본OFF |
| `--greedy-only --temperatures 0` | 고정 T=0 sampler specialization; tree는 top-C 후보 + target argmax walk |
| `--mode duet-tree --p1-tree --batches 1` | 두 phase dynamic tree 활성화 |
| `SSD_P1_TREE_EXEC_WORKSPACE_MB` | P1 전용 FA2 workspace. Qwama에서는 구 48MiB가 부족; 기본64MiB |

## 실험 폴더

| 폴더 / 파일 | 역할 |
|---|---|
| `packed_smoke/` | 초기 LM-head reshape 오류. 무효 실행, 성능/정확성 성공 근거에서 제외 |
| `packed_smoke_v2/` | 수정 후 소규모 smoke |
| `llama*_packed/` | 48문항 greedy/stochastic base vs packed |
| `llama*_mixed/` | 48문항 mixed-miss AR screen; 일부 GPU pair가 다른 screen끼리 정밀 성능 비교 금지 |
| `llama*_full/` | 480×3seed, base/packed/mixed. 최적화 분리 측정 |
| `llama*_final/` | 480×3seed, 최종 candidate와 SSD K3/F2 교대 실행 |
| `llama*_tree_greedy/` | 16문항 AR/chain/P2 tree/P1+P2 tree; Llama3 P1 workspace 실패 로그 포함 |
| `llama3_tree_retry/` | 실패: P2 workspace를 바꿨지만 별도 P1=48MiB override가 남아 있었다 |
| `llama3_tree_retry_v2/` | 실제 P1 override를64MiB로 수정한 성공 재실행 |
| `llama*_tree_full/` | 480문항 B1 greedy 두 phase tree vs chain |
| `llama*_tree_host/` | 같은 full tree에서 host metadata 재사용 A/B |
| `batch_tree_*.json` | 전체 draft 가중치+합성 prefix의 실행기 microbench. End-to-end TPS 아님 |
| `batch_tree_*_diagnostic.json` | 큰 배치 first-round logit/top-3와 전체 rollout 차이를 구분하는 추가 진단 |
| `batch_tree_*_diverse_hf.json` | 서로 다른8개 문맥×5root, 독립 HF float32 reference 진단. root 분포와 실제 serving 검증을 구분 |
| `batch_tree_qwama_layers.json` | 첫 round의 layer별 중간값 비교. 첫 QKV부터 작은 수치 차이 발생 |

## 코드 핵심과 주의할 계약

### Packed verify

- `helpers/packed_verify.py`: real row마다 정확히 `valid_k+1` queries. alignment padding은 별도 dummy sequence; slot=-1.
- `SpeculatorAsync.extend_seqs_for_verify`: CPU valid_k를 sequence에 일시 저장. ModelRunner.call 인자로 TP rank에 전달한다.
- target graph는 총 토큰 용량과 batch 용량으로 구분한다. logits는 flatten 후 dense logical row로 gather한다.
- 실제 sequence에 padding query를 붙이면 causal alignment가 변하므로 금지한다.
- shape가 달라질 때 fp16/bf16 near-tie 출력은 달라질 수 있다. kernel 수치 차이와 분포 검증 오류를 구분해야 한다.

### Mixed miss

- `hit_cache_and_respond`: 일부 hit인 chain batch에서만 JIT 생략. 모두 miss면 기존 JIT 유지.
- miss `valid_k=0`: recovery 이후 target에서 직접 다음 token을 뽑는다. q 또는 residual을 적용하면 안 된다.
- 해당 row의 proxy root scoring도 ordinary exit P를 사용한다.
- throughput 변경에 AL 감소가 동반된다. 논문 AL 개선 기법으로 설명하면 안 된다.

### Greedy tree

- `tree_sample_wor(...allow_greedy=True, greedy_only=True)`: T0 child는 untempered distribution의 top-C. 유한한 ranking score, RNG 불사용.
- `tree_verify_walk_greedy`: 모든 context의 argmax를 한 번에 읽은 뒤 맞는 child를 따라간다. parent-q ratio 불사용.
- stochastic target + deterministic branching draft는 여전히 차단. 같은 분포인 척 q로 나누지 않는다.
- output ordering, parent-q wire, terminal node id, target KV commit 및 draft KV restore는 기존 B1 규약을 유지한다.

### Batched tree primitive

- `P2TreeExecutor.iter_rounds`가 각 round의 ids/rope/context를 yield하고 logits를 받는다.
- `BatchedTreeExecutor`는 B개 독립 arena를 진행하면서 한 번의 batched model forward를 실행한다. cross-request top-W 경쟁은 하지 않는다.
- 요청별 paged KV index와 custom mask를 concatenation하되 FlashInfer의 query/kv indptr로 request 경계를 유지한다.
- per-arena RNG/parent-q ref/node budget을 분리하고 CUDA graph에 등록한다.
- **아직 DraftRunner/Verifier의 전체 통신 경로에 연결하지 않았다.** Qwama의 큰 최대 logit 차이는 독립 HF 검사까지 추가했다. 다양한8개 문맥의40root에서 개별↔배치 TV 평균0.01074, HF float32 대비 배치 TV0.02180(개별0.02239), argmax40/40(개별39/40). 최대 logit 차이8은 사실상0 확률 토큰에서 발생했다. 큰 절댓값만으로 배치 버그라고 단정하지 말되, 이 소규모 검사로 전체 serving 정확성/성능이 검증됐다고 주장하지도 않는다.
- 재현: `ssd/bench/batched_tree_microbench.py --model <draft> --batches 8 --real-prefix --diverse-prefix --plausible-roots --hf-reference --output <새 파일>`. `--trace-layers`는 eager 중간값 추적용이다. timing 이후 독립 HF를 로드하므로 그 검사 시간은 primitive latency에 포함하지 않는다.
- `BatchedTreeExecutor`에 실제 request를 연결할 때 arena별 RNG seed/stream을 명시적으로 분리한다. 현재 constructor의 같은 초기 seed를 모든 request에 그대로 복제하지 말 것.

## B>1 dynamic tree를 끝내기 위한 작업 순서

1. **Request별 snapshot/cache 분리**: 현재 `_tree_hit_root/views/phase`, `_tree_served_ints/numtok/seq`, `_tree_staged_kv`가 단일 요청 필드다. row 인덱스 대신 seq_id를 키로 관리하고 종료/refill 때 폐기한다.
2. **Mixed chain/tree 응답 wire**: `tree_response_logit_rows`, `DraftRunner._service_spec_request`, `SpeculatorAsync._speculation_request`의 B1 상호 배타 chain-q/parent-q protocol을 배치별 payload로 바꾼다. root별 parent-q ref가 다른 요청의 logits를 참조하지 않게 한다.
3. **Target batched ancestor attention**: `ModelRunner._run_tree_verify`의 [rec+nodes]를 요청별 valid/indptr로 묶는다. padding row에 KV를 쓰지 않는다. tree depth에 맞는 RoPE와 prefix+ancestor+self mask를 적용한다. chain/tree mixed도 명시 처리한다.
4. **검증/commit**: Verifier가 row마다 topology, temperature, phase, q를 선택한다. stochastic에는 정확한 sibling residual ladder, greedy에는 argmax walk. accepted path KV를 각 seq의 canonical slots에 gather→scatter한다. TP commit 순서도 확인한다.
5. **Draft glue / 복원**: 다음 request의 terminal node와 실수락 길이로 각 seq의 staged KV를 복원한다. batch 행 이동, EOS 절단, 새 seq, preemption에 다른 요청의 staging이 섞이면 안 된다.
6. **Batched executor 연결**: 위 snapshot/통신 계약을 확보한 뒤 P1/P2 arena를 BatchedTreeExecutor로 묶는다. 순차 B1 replay를 B>1 최적화라고 부르지 않는다.
7. **검증 게이트**: 서로 다른 prefix·다른 topology·다른 valid_k·mixed hit/miss·batch shrink/refill·preemption; 실제 tree event 필수. fixed-noise 구조/attention parity와 독립 HF greedy first-divergence 검사를 병행한다.
8. **성능**: B1/2/4/8/16, 두 dense pair, T0/.7, 전체480 first turns 및 긴 output. 같은 node/forward budget의 chain/tree를 먼저 비교하고, 이후 parameter tuning 결과를 분리한다.

`filter_unservable_tree_matches`, `_should_run_p2_tree`, `_run_p1_tree_step`, harness B1 guard를 위 계약보다 먼저 제거하지 않는다.

## 추가 시스템 검증

- Blackwell/FlashInfer의 packed verification 지원 구현·실기 측정. 현재 4090 결과로 portable 최적화 완료를 주장하지 않는다.
- paper의 target TP2 또는 더 큰 full model: 4090 TP1 값으로 scaling을 추정하지 않는다.
- 이번 max-output64 이외의 긴 output, 더 큰 context/full canonical dataset은 별도 필요하다.
- 사용자 root-selection 연구(`e(1-q)` 등)는 원래 research branch에 있다. 시스템 검증 도중 무단으로 결합하지 않는다.

## 종료 시 측정 판단

- B8 최종 후보: Llama2 652.45±3.79 TPS, Llama3 673.06±5.54 TPS. 동일 조건 SSD는649.08±1.70 / 652.33±4.16. Llama2 우위는 확실하지 않다.
- Host topology 제거는 출력/phase events 동일하지만 측정 TPS는 개선되지 않았다. 기본OFF 유지.
- Greedy tree는 AL을 높였지만 두 작은 target 조합에서 chain보다 TPS가 낮다. 지원 완료와 최적 성능 달성을 구분한다.
- 가장 큰 미완료는 B>1 tree serving 전체 연결이다. prototype 속도나 B1 greedy 성공으로 이를 완료 처리하지 않는다.
