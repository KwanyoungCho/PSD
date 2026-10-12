# 논문 브랜치에 선택적으로 옮길 패치

기준 commit은 `a82f7d24fb36827a9a81a3567f344dccb71f193e`다.
이 파일은 적용 방법이며 채택 판정은 최종 REPORT를 따른다. 원 논문 브랜치에
자동 merge하지 않았다. 각 패치는 서로 분리했고 `git apply --check`를 통과했다.

| 파일 | 실제 변경 | 적용 범위 |
|---|---|---|
| `patches/01_chain_fast.patch` | CPU 요청 정보로 target/draft 온도가 모두 양수인지 판단하고, JIT chain의 불필요한 greedy 계산·GPU 조건 readback·여러 결과 복사를 제거 | SSD와 DUET의 공통 verifier. Tree cache-hit walk는 변경하지 않음. Greedy·mixed/non-JIT 일반 경로는 원본 유지 |
| `patches/02_tree_graph_candidate.patch` | 원래 B1 tree executor에서 조건부 residual ladder만 graph로 계산, 원래 방문 순서의 CPU coin과 terminal multinomial 보존 | **실험 후보**. 새 B>1 executor 이식이 아님. `SSD_PAPER_TREE_ACCEPT_GRAPH=1`에서만 작동. 성능 근거 없이 기본 활성화하지 않음 |
| `patches/03_kv_correctness.patch` | Paged prefix 접근, 완전 cache prompt의 마지막 query, preemption 전 원래 출력 경계, 올바른 완료 block의 hash | 공통 정확성 수정. Tree glue/accepted-path commit/defragmentation/promotion 변경 없음 |

01은 양의 온도 JIT 경로만 가져오는 더 좁은 패치다. Systems branch의 greedy
최적화/수정, packed queries, 새 proxy graph나 mixed-miss 정책까지 묶어 오지 않는다.
실험 adapter의 stochastic 경로와 동일한 `verify_stochastic`를 사용한다.
02에는 graph cache/metadata 준비 비용이 있으므로 GPU kernel 구간만으로 채택하면
안 된다. 모든 최종 판단은 전체 verification과 step TPS를 함께 본다.

02의 측정된 구현은 `prototypes/tree_v2_measured.py`다. 원 verifier의 wire
validation은 `0 <= parent_q_ref < u_valid <= valid`를 요구한다. 별도 점검에서
이 범위를 벗어나는 예는 기존 validation이 이미 거부함을 확인했고, 필요 없는
추가 경로/방어 조건은 넣지 않았다. 채택 추천 여부는 전체 latency 결과로 정한다.

안전한 검토 예시(새 review worktree에서 실행):

```bash
git worktree add --detach --no-checkout /path/to/paper-review a82f7d2
git -C /path/to/paper-review sparse-checkout set --cone ssd/ssd
git -C /path/to/paper-review checkout --detach a82f7d2
git -C /path/to/paper-review apply --check /absolute/path/01_chain_fast.patch
git -C /path/to/paper-review apply --check /absolute/path/03_kv_correctness.patch
```

이후 승인한 파일만 적용한다. 다른 서버에서 원 논문 코드가 이미 수정됐다면
강제 적용하지 말고 patch의 함수 단위 diff를 대조한다. 이 실험의 공통
benchmark adapter(AMD dtype, SHM 이름, cold-prefix 제어, 집계)는 이 세 패치에
숨겨 넣지 않았다. 실제 온도·G>M·집계·scheduler·wire의 기존 승인도 별도다.

`/home/chokwans99/PSD-paper-port-review`에는 통합 smoke 검증을 위해 세 패치를
실제 적용한 detached checkout을 준비했다. 이것은 논문 branch 변경이나
02의 채택 승인을 뜻하지 않는다. 원본 비교 checkout은 clean 상태로 유지한다.

추가로 `/home/chokwans99/PSD-paper-chain-review`에는 **01만** 물리적으로
적용했다. 최종 `tree_chain_b1` full480 대조는 이 checkout을 직접 실행한다.
02/03 또는 runtime 함수 교체가 섞이지 않아 실제 최소 패치의 tree workload
latency를 확인할 수 있다. 역시 논문 branch 자체를 변경한 것은 아니다.
