# B1 tree PS 준비 최적화 — 직접 breakdown 검증

2026-10-15. 기준 `MLSys-ver1.1` @ `0916871`. 아래 runtime 변경은 현재 작업 트리 반영이며 새 커밋·push는 하지 않았다.

## 판정

**PS 전체 5.269→4.928ms, 6.48% 감소.** 준비·검사 네 구간 합은 1.947→1.607ms (17.47% 감소). 실제 DS/PS graph와 target forward 시간은 거의 같다. 감소는 모델 연산량이나 AL 변경에서 온 것이 아니라, PS 입력 준비·상태 검사의 비용 감소다.

- 최종 runtime 회귀 **382개 / 40 suites 통과**, skip/error/failure 0. 검사 전후 source hash 동일.
- Profile: 동일 24턴 × 세 쌍, 출력 **72/72 일치**. 각 쌍의 step별 source/출력/AL/hit 일치.
- Profile OFF: 별도 동일 48턴 **48/48 출력 일치**, AL/hit 일치, 양 arm 측정 중 graph 준비 영향 step **0**.
- 세 쌍의 2,391개 일치 step 비교. 24턴 중 너무 짧아 첫2/마지막 제외 후 표본이 없는 2턴을 제외한 **22개 문항**이 latency 통계에 기여한다. 반복을 독립 문항으로 세지 않았다.
- PS 평균 차이의 문항 단위 paired bootstrap 95% 구간: **-0.390~-0.290ms**. 동일 입력 step 단위로 정렬한 뒤 문항을 재표집했다.

## 직접 볼 그림

1. [구간별 변경 전후 비교 PDF](BREAKDOWN_COMPARISON.pdf), [PNG](BREAKDOWN_COMPARISON.png): 평균의 가산 분해. 대표 단일 step을 평균처럼 사용하지 않는다.
2. [기존 방식의 native timeline 전후 비교](native/NATIVE_BEFORE_AFTER.pdf): DS hit / PS hit / miss 각 한 쌍, **동일 UID·동일 step**. 기준 arm에서 source별 PS 중앙값에 가까운 step을 선택하고 수정 arm의 같은 step을 표시했다.
3. [전체 수치·CI](BREAKDOWN.json), [step별 대응 원시 추출](MATCHED_STEPS.json).

## 어떤 코드가 왜 바뀌었나

### 1. 같은 root 위치 매핑을 다시 만들던 계산 제거

기존 B1 tree-hit PS 구성은 `_update_phase2_layout_inplace`가 만든 `fan_idx_hit`와 같은 값을 `repeat_interleave`로 한 번 더 만들었다. 이미 있는 매핑을 그대로 재사용한다. 남은 `repeat_interleave`에는 이미 확정된 출력 크기 `B*MQ_LEN`을 주어 GPU에서 출력 크기를 확인하는 동기화를 줄인다. 선택된 위치·token·점수·순서는 바꾸지 않는다.

### 2. Tree 조상 마스크의 GPU→CPU→GPU 왕복 제거

기존에는 GPU의 `_p1_context_glue_rows`에서 선택한 마스크를 `.cpu().numpy()`로 내리고, NumPy→pinned host→executor GPU 입력으로 다시 올렸다. 이제 GPU에 유지한 채 executor 입력에 직접 복사한다. 마스크의 내용·shape·유효 열 수는 같다. 참조/eager fallback·진단은 기존 NumPy 계약을 유지하도록 그 경로에서만 CPU로 변환한다. 첫 pilot의 executor OFF 전후 24턴도 출력 24/24가 일치하여 fallback 연결을 확인했다.

### 3. PS 유효성 검사를 한 GPU kernel로 묶음

기존 `_p2_step_state`는 작은 stack, 비교, any, dtype 변환, cat 연산을 여러 번 실행한 후 한 번 CPU로 읽었다. 새 경로는 **동일한 여섯 검사**를 한 Triton integer kernel에서 수행하고, 같은 작은 snapshot을 한 번 읽는다. CPU readback 자체를 없앴다는 뜻은 아니다.

검사는 `no_roots → ctx → rope → slot → ctxlen → pages` 순서와 실패 우선순위를 그대로 유지한다. 실제 root R만 RoPE 검사, 모든 forward lane W는 slot 검사, 실제 prefix page만 검사하는 계약도 같다. 잘못된 metadata를 허용하거나 검사를 생략하지 않는다. CPU/list 진단 입력은 기존 참조 경로를 유지한다.

고정 입력 component 검사 F=3/W=15/R=10: **0.205→0.042ms**. F=6/W=26과 F=8/W=32도 감소했다. 100회 AB/BA pair, 각 sample 8회, warmup25회; [STATE_COMPONENT.json](STATE_COMPONENT.json). 이 microbench 값과 online dispatch gap 전체는 범위가 다르므로 합산하지 않는다.

새 사용자 옵션, 후보 수식 변경, calibration, pruning, K1/K2 변경은 없다. graph 실행체/forward 수/RNG 소비도 바꾸지 않았다.

## 측정 설정

- Target: `/home/chokwans99/models_mlsys/layerskip-llama2-7b`; draft: `/home/chokwans99/models_mlsys/AMD-Llama-135m`.
- RTX4090, GPU4 target / GPU5 draft, target TP1; CPU affinity16–23/48–55. B1, BF16, T0.7, seed9101.
- exit20 (0-based), K1/K2=6/3, M1/M2=G1/G2=12/8, C3, DS roots/position2, DS continuation cap26, PS forward width15/root10, frontier. Root/position 수식은 현재 ver1.1 기본값 유지.
- **파라미터 최적화 실험이 아니다.** 같은 비용 경로를 전후 비교하기 위한 고정 설정이다.
- SpecBench full 입력 파일에서 균등 간격으로 24턴 선택; warmup4턴, 출력 상한96, 입력 상한512, 문맥2048. 480문항 전체 AL 평가로 해석하지 않는다. Profile OFF 확인은 같은 방식의 48턴이다.
- `SSD_TREE_EXEC=1`, graph prewarm=all, `SSD_ASYNC_PROXY_SEND=1`; 양 arm 동일 cold-prefix. Profile OFF 결과의 준비 step0도 확인했다.
- 기존 3090/68M 그림과는 **모델과 GPU가 다르다**. 그 그림의 6.87ms와 여기의 값을 직접 비교하지 않는다.
- 기준 checkout은 `/data/chokwans99/duet-ps-prep-20261015/baseline` clean detached0916871. 수정 arm은 위 commit + `RUNTIME.patch`다. 실행 도중 runtime은 변경하지 않았다.

## 구간별 결과

| 구간 | 이전 ms | 수정 ms | 차이 ms | 변화 |
|---|---:|---:|---:|---:|
| PS root/layout 구성 (`phase2_build`) | 0.877 | 0.771 | -0.105 | -12.00% |
| 슬롯 계산·상태 검사·dispatch 빈 구간 | 0.638 | 0.417 | -0.221 | -34.65% |
| PS 준비 A: root 메타데이터 | 0.193 | 0.202 | +0.009 | +4.65% |
| PS 준비 B: graph 입력 버퍼 | 0.239 | 0.216 | -0.023 | -9.45% |
| PS graph 재생: forward+tree 선택 | 2.861 | 2.856 | -0.005 | -0.19% |
| 나머지 PS 간격·출력 변환·병합 | 0.461 | 0.465 | +0.004 | +0.91% |
| **PS 전체** | 5.269 | 4.928 | -0.341 | -6.48% |
| Target forward: pre+post+final logits | 15.522 | 15.514 | -0.008 | -0.05% |
| DS graph 재생 | 6.591 | 6.581 | -0.010 | -0.16% |

- PS 전체는 **proxy 수신 완료부터 cache merge 완료까지**의 같은 draft process 시간이다.
- Dispatch는 `phase2_build` 끝과 첫 `p2_prepare` 시작 사이의 직접 관찰 간격이다. 코드상 persistent metadata copy·slot 도출·유효성 검사·dispatch를 포함한다. CPU 단독 시간이나 GPU kernel 단독 시간으로 부르지 않는다.
- 준비 A/B는 원래 같은 `p2_prepare` label인 두 event를 실행 순서로 분리했다.
- Graph에는 K2 forward 외에도 tree 선택·sampling이 포함된다. 순수 Transformer 시간으로 부르지 않는다.
- 나머지는 graph 전후 간격·출력 변환·merge를 포함한 잔여 시간. 전체에서 서로 겹치지 않는 구간을 빼서 계산하며 부모/자식 event를 이중 합산하지 않는다.
- **모든 칸이 줄지는 않았다.** 준비 A와 잔여 비용의 소폭 증가도 표에 남겼다. 채택 근거는 준비 전체와 PS 전체의 순감소다.

### 반복과 source별 확인

| 반복 | 실행 순서 | 이전 PS ms | 수정 PS ms | 감소 |
|---|---|---:|---:|---:|
| 1 | before→after | 5.331 | 5.009 | 6.03% |
| 2 | after→before | 5.300 | 4.975 | 6.14% |
| 3 | before→after | 5.176 | 4.800 | 7.28% |

반복1·2는 GPU6 회귀 검사와 일부 시간대가 겹쳤다. GPU4/5에는 다른 compute process가 없었지만 CPU 부하 가능성을 배제하려고 **반복3은 회귀 종료 후 inference pair만 실행**했다. 반복3에서도 감소가 유지되는지 별도로 확인했다. 온도/clock 요약은 [TELEMETRY_SUMMARY.json](TELEMETRY_SUMMARY.json), raw는 archive에 있다. GPU clock 고정 실험은 아니며 서브 % graph 시간 차이를 이 최적화의 효과라고 주장하지 않는다.

| 현재 step source | 일치 step 쌍 | 이전 PS ms | 수정 PS ms |
|---|---:|---:|---:|
| miss | 333 | 5.153 | 4.905 |
| DS/P1 hit | 813 | 5.272 | 4.914 |
| PS/P2 hit | 1245 | 5.298 | 4.943 |

이 source는 **현재 verification에 제공된 tree/chain의 출처**다. 표의 PS는 그 step에서 다음 cache를 준비하는 PS 시간이다. 세 source 모두 개선 여부를 확인한다.

## Profile OFF 확인

| 항목 | 이전 | 수정 |
|---|---:|---:|
| 출력 일치 | 48턴 | 48/48 동일 |
| AL | 2.155954 | 2.155954 |
| Cache hit | 83.0593% | 83.0593% |
| Step 평균 ms | 19.395 | 19.184 |
| 준비 영향 step | 0 | 0 |
| 참고 TPS | 111.16 | 112.38 |

[UNPROFILED.json](UNPROFILED.json). 단일 비교쌍의 보조 점검이며 모델의 최적 TPS나 전체 데이터 개선율 주장은 아니다. PS 일부가 target과 겹치므로 PS 절감분이 timestep에서 그대로 절감될 필요도 없다.

## 검증·재현·남은 범위

- [CHECKS.json](checks/CHECKS.json), [source provenance](checks/CHECKS_PROVENANCE.json): 최종 코드 382 tests, source unchanged.
- 새 검사는 여섯 실패 원인·우선순위, padded root/모든 lane의 slot, page tail, non-contiguous stride, int32/int64, 재사용 버퍼, 무작위 metadata 손상, GPU/host 마스크 내용 동등성, CPU/list fallback을 확인한다. 실제 eager fallback 전후 출력도 동일하다.
- [RUNTIME.patch](RUNTIME.patch): runtime 두 파일만의 변경. benchmark/plot/report와 분리해 검토할 수 있다.
- [PROVENANCE.json](PROVENANCE.json): 실행별 설정·소스 해시·실제 commit/dirty 구분. raw log/profile/run JSON과 실행 script는 `RAW_RESULTS.tar.gz` 및 원래 `/data/chokwans99/duet-ps-prep-20261015/`에 보관한다.
- Full regression: `CUDA_VISIBLE_DEVICES=6 /home/chokwans99/PSD/ssd/.venv/bin/python results/mlsys_ver11/check_suites.py --output-dir <new-dir>`.
- 재분석: `/home/chokwans99/PSD/ssd/.venv/bin/python results/mlsys_ver11/ps_preparation_20261015/analyze.py --raw /data/chokwans99/duet-ps-prep-20261015/runs --output results/mlsys_ver11/ps_preparation_20261015`.
- 여전히 root 선택, 일부 scalar/fanout readback, slot matrix 생성, executor 입력 buffer 갱신, cache merge 비용이 남는다. 이번에 전체 overhead를 없애지는 않았다.
- 이 결과는 **7B+135M / B1 / 4090** 검증이다. 70B·Blackwell·다른 모델과 B>1의 같은 개선율은 주장하지 않는다. tree 알고리즘·lossless 검증법은 그대로이고 새 경로는 metadata 계산에 한정된다.
