# MLSys-ver1.5 인수인계

2026-10-19 KST. 기준은 `MLSys-ver1.4@dedc85ccb299df2d76fcf34d50f05f2d8747a32a`. 새 브랜치는 `MLSys-ver1.5`, 로컬 worktree는 `/home/chokwans99/PSD-MLSys-ver1.5`다. Runtime 커밋은 `d9029f27bb14bdc3876a76d606a0a51f029ffb03`이며, 문서·실행 도구는 후속 커밋이다. v1.4와 기존 연구 worktree의 runtime은 수정하지 않았다.

## 변경의 의미

사용자 요구는 생성 순서 prefix 대신 **draft 경로 점수 상위 M개**를 tree 검증에 사용하되 target에 대해 무손실을 유지하는 것이다. 사후 점수 선택 후 기존 q를 p/q verifier에 그대로 넣으면 편향되므로, 사용자 승인에 따라 **tree 선택법과 verifier를 함께 변경**했다. 자세한 반례·증명·선행연구는 [THEORY.md](results/mlsys_ver15/THEORY.md).

| 항목 | v1.4 | v1.5 |
|---|---|---|
| Root 후보 | 기존 complement·혼합 위치 점수 | 그대로 유지하는 휴리스틱 |
| Forward 확장 | root 점수 × 원래 draft q-path, 마지막 깊이 경쟁 | 동일 |
| 생성 제한 | root별 G=M | 별도 root별 G 제거, `C×실제 phase 입력 수 합`의 자연 상한 |
| 검증 subtree | 생성 순서 앞 M | root별 경로 확률 top-M, 부모 보존·index 재작성 |
| Tree-enabled 실행의 검증 | 기존 q 기반 p/q | 각 문맥에서 target sampling 후 자식 포함 여부 확인 |
| Miss | chain JIT draft | chain JIT 유지. tree-enabled run의 miss도 새 target sampler로 검증 |
| B=1 경로 | 기존 single-request 구현 | B>1과 같은 공용 tree 실행 경로 |
| Pure chain mode | p/q | 변경 없음 |
| KV·scheduler·출력 집계 | v1.4 | 기존 로직 유지 |

**주의:** 새 검증에는 실제 rejection residual이 없다. `e(1-q)`와 기존 위치식은 우선순위 휴리스틱으로만 유지한다. 실제 correction 분포라는 기존 수학적 해석을 그대로 붙이면 안 된다. 새 verifier는 q와 p가 같아도 모든 draft 토큰을 항상 수락하지 않는다. AL/hit/TPS는 새로 측정해야 한다.

## 파일별 검토 순서

1. `score_tree.py`는 독립 CPU 참조다. serving의 `score_tree_export.py`는 같은 log q-path 순위·top-M·부모/형제 remap을 GPU graph로 계산한다. 안정적인 동점 처리를 포함한다.
2. `target_tree_sampling.py`: dense/flat target draw, GPU 경로 선택, recovery 반환. T=0은 argmax, stochastic 전용 경로에서는 greedy 연산 생략.
3. `batch_tree_draft.py`: 자연 생성 용량, GPU 최종 선택, 선택된 M노드 metadata의 한 번 bulk CPU export, 기존 token·q·topology wire 사용.
4. `batched_tree_executor.py`: staging-only 실행에서 사용하지 않는 거대한 root별 finalize 생략. 실제 parent cell의 logits를 유지한다.
5. `batch_tree_common.py`, `batch_tree_verify.py`, `batch_tree_warmup.py`: B1/B>1 공용 경로, phase별 M에 맞는 shape, 모든 shape 사전 준비, graph/eager 모두 새 verifier 적용.
6. `batch_tree_sampling.py`, `flat_tree_sampling.py`: 내부 `target_only=True` 연결. 공개 설정 옵션을 추가한 것이 아니다. 과거 verifier 함수는 pure-chain/참조 검사에 남아 있으나 점수로 pruning한 serving tree에 연결하지 않는다.
7. `runtime_manifest.py`, `config.py`: 실제 selection/verifier/search bound 기록. 공개 파라미터 이름은 v1.1 계약을 유지한다. 옛 max_nodes alias는 M의 deprecated alias이며 별도 G를 다시 설정하지 않는다.
8. `test_score_tree_v15.py` 및 수정 fixture: 편향 반례, exact 분포 열거, greedy, flat/dense, B1/B4 CUDA graph, 실제 arena export 검증.

## Latency를 고려한 구현

- Target 수락·경로 선택은 GPU CUDA graph 한 번으로 재생한다. depth별 `.item()`/`.tolist()` 왕복이 없다.
- Draft 생성은 기존 고정 forward 폭·graph를 유지한다. 새로운 tree를 다시 forward하거나 token을 재추출하지 않는다.
- 최종 점수 선택·부모/형제 index 재작성은 **GPU graph에서 수행**한다. CPU에는 root별 최종 M노드의 작은 metadata만 한 번 bulk readback한다. 모든 생성 노드를 CPU에서 정렬한 초기 구현은 latency 검사에서 더 느려 채택하지 않았다. CPU 포장·readback이 0이라고 주장하지 않으며 component 검사로 별도 측정한다.
- Staging-only 실행에서 사용하지 않는 root별 parent/backbone finalize를 생략한다. 생성 상한을 키웠는데 옛 finalize를 유지하면 불필요한 작업이 커진다.
- M1=23/M2=15이면 PS target forward와 glue도 15노드 shape를 사용할 수 있다. PS를 매번 P1의 23노드까지 padding하지 않는다. 이 차이 때문에 첫 최종 smoke에서 발견한 누락 import와 glue bucket 불일치를 수정하고 다시 검증했다.
- 사전 graph 준비는 inference 집계 밖이다. 선택법의 latency와 실제 AL/TPS 효과를 구분한다. component 결과만으로 전체 step 개선을 주장하지 않는다.

## 연속 성능 측정 조건

| 항목 | 값 |
|---|---|
| Target/Draft | LayerSkip-Llama2-7B / JackFram llama-68m, BF16 |
| Batch·온도 | B=1, target/draft 모두 0.7, nucleus 없음 |
| Exit | 21 (0-based) |
| P1 | K1=10, 위치당 root4, continuation 입력 cap56, M1=23 |
| P2 | K2=5, root24, forward 폭28, M2=15 |
| C·확장 | C=3, last |
| 비동기 옵션 | proxy send만 ON, exit replica·proxy side stream OFF |
| 데이터 | Spec-Bench 480문항/560 raw 턴 전체, 원래 입력 ID·순서 유지 |
| 문맥·출력 | 2048 / min(1024,2048−입력 길이), 자연 EOS, 기존 context guard |
| Warmup | 각 seed 첫 3개 raw 턴을 성능에서 제외, 중복 실행 안 함 |
| Profiling | OFF. 독립 CPU wall 기록·원시 step 기록 유지 |
| Prefix | cold prefix, 과거 연속 seed와 같은 실험용 처리 |
| 집계 | 턴→문항→seed 평균. cap/clip·부분 batch·같은 요청/batch/source mix 내 2×중앙값 초과 step 제외. 제외 수와 raw도 저장 |
| Seed 배정 | 42→123→2026→0→1→2→…, 중복 없이 사용자 중단까지 |
| GPU | 로컬 orion의 비어 있는 허가된 쌍 (4,5),(6,7),(1,2),(0,3). 외부 작업은 건드리지 않음 |

이 설정은 v1.4 최종 실험의 파라미터를 승계한 것이다. 새 verifier에 최적인지 찾은 결과가 아니다. v1.4 seed를 v1.5 결과로 재사용하지 않는다.

## 다른 서버에서 받기

```bash
git fetch origin MLSys-ver1.5
git worktree add -b review/mlsys-ver1.5 ../PSD-MLSys-ver1.5-review origin/MLSys-ver1.5
cd ../PSD-MLSys-ver1.5-review
git log -1 --format='%H %s'
git status --short
```

GPU 모델·가중치·데이터·venv·CPU affinity·GPU 쌍·포트는 서버마다 확인한다. `tools/controller.py`는 실수로 원격 latency를 섞지 않도록 **로컬 orion 전용**으로 둔다. 다른 서버는 `run_once_helper.py`와 `PLAN.json`을 자신의 환경에 맞게 검토한 뒤 별도 raw/queue를 만든다. 기존 실행 중 파일을 바꾸지 않는다.

## 실행 결과 위치와 중단

- 최종 검증: [REPORT.md](results/mlsys_ver15/REPORT.md), `checks/`.
- 실행 도구: `results/mlsys_ver15/tools/`. 내부 adapter는 기존 성능 집계·입력 계약을 고정한 benchmark 도구이며 engine runtime 교체본이 아니다.
- 로컬 live 표: `results/mlsys_ver15/performance/REPORT.md`. 완료·검증된 seed만 반영한다.
- Raw: `/data/chokwans99/duet-ver15-score-seeds-20261019`, seed별 `runs/seedN/{PROVENANCE.json,run.json,status.json,process.log,telemetry.jsonl}`.
- `RELEASE.json`의 validation_passed와 pushed가 True이고 실제 HEAD와 일치해야 controller가 실행된다. 실행 중 source hash가 바뀌거나 결과 검증이 실패하면 새 seed를 멈춘다.
- 정상 정지: raw의 `STOP` 파일 생성. 진행 중 seed는 마치고 다음 seed를 배정하지 않는다. 즉시 종료 요청이면 소유 프로세스를 확인해서 우리 job만 종료한다.
- 디스크 여유 35GiB 미만이면 새 seed 배정을 보류한다. 과거 데이터를 임의 삭제하지 않는다.
- Live 결과는 계속 변하므로 Git ignore다. 향후 사용자 중단 시 완료 seed의 보고서·summary·해시를 별도 snapshot으로 커밋한다.

## 아직 주장할 수 없는 것

- 점수순 선택이 v1.4보다 AL/TPS를 높였다는 결론: 연속 full 평가 후 확인.
- 새 verifier에 최적인 root 수식·K/exit/width/M: 재탐색하지 않음.
- 13B/70B 또는 다른 GPU의 성능과 모든 TP 구성: 이번 7B 로컬 결과로 보장하지 않음.
- B>1 실제 모델 full 비교: CUDA graph 기능 검사는 포함하지만 이번 사용자 지정 성능 실험은 B=1.
- Raw 진단 데이터의 큰 target/draft full-vocab 로그: 이미 별도 compact 데이터로 보존했으며 이 브랜치에 불필요하게 포함하지 않음.
