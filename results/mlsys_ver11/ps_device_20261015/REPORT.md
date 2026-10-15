# PS의 불필요한 CPU readback 제거 — 추가 최적화

2026-10-15. MLSys-ver1.1 기반의 추가 후보 실험. **이번 후보는 미채택하고 직전 PS 최적화 runtime을 유지한다.** 새 커밋·push 없음.

## 결론

**미채택: PS 전체와 graph 외 overhead가 함께 줄었다는 증거가 부족하다.** 직전 최적화 버전과 직접 비교해 PS 전체 **4.965→4.948ms (-0.34%)**.
준비 네 구간 합은 **1.634→1.622ms**다.
PS 전체 차이의 문항 단위 paired bootstrap 95% 구간은 **-0.050~+0.028ms**다.

기능 검증은 통과했지만, `phase2_build`의 감소가 PS 전체와 graph 외 overhead의 확실한 감소로 이어지지 않았다. 평균만 보고 이 후보를 성능 개선으로 추가하지 않았다. `candidate_source/`와 patch에 구현을 보존하고 현재 runtime은 baseline 소스와 일치하도록 복원했다. [복원 소스 검증](RESTORED.json)을 확인한다.

기준은 clean `0916871`이 아니라 [앞선 PS 준비 최적화](../ps_preparation_20261015/REPORT.md)까지 포함한 runtime이다.
두 단계 실험의 다른 실행에서 얻은 절대 시간을 이어 붙여 누적 개선율을 계산하지 않는다.

## 원인과 변경

1. B1 tree hit의 PS 준비에서 GPU fan-out을 `.tolist()`로 읽고 Python seed 위치 목록을 만들었다. 실제 captured executor는 이 목록의 **길이만** 사용했다. 이제 root 수는 이미 확정된 tensor shape에서 가져오고, 위치 매핑과 조상 mask는 GPU에 유지한다.
2. 같은 request의 문맥 길이는 DS 실행 준비에서 이미 CPU로 읽었다. PS에서 같은 tensor를 다시 `int(...)`로 읽던 부분을 없애고 DS에서 보관한 Python 정수를 전달한다.
3. Eager fallback이나 진단 도구가 실제 위치 목록을 필요로 하면 그 경로에서만 host fan-out과 seed 위치를 만든다. 매 step 목록을 `None`으로 초기화하므로 이전 요청의 값이 남지 않는다. 정상 captured 경로를 막는 새 사용자 옵션은 없다.

작은 데이터를 읽는 데에도 GPU의 선행 작업이 끝나기를 기다리는 비용이 포함될 수 있다. 따라서 제거한 데이터 크기만으로 절감 시간을 예측하지 않는다. 또한 `phase2_build`만 줄어도 뒤 dispatch로 대기가 이동할 수 있으므로 **PS 전체 완료 시간까지** 비교했다.

`metadata_ints.F`는 이 P2 tree 경로에서 사용하지 않아 0을 전달한다. Legacy chain 경로는 host fan-out과 원래 F를 유지한다. Graph 거절 후 arena fallback도 목록을 복원한다. 후보 점수, 순서, budget, tree 정책, graph 내부 수학 연산, RNG 소비, 여섯 metadata 유효성 검사는 변경하지 않았다.

## 직접 breakdown

| 구간 | 직전 최적화 ms | 이번 수정 ms | 변화 |
|---|---:|---:|---:|
| PS root/layout 구성 | 0.787 | 0.768 | -2.43% |
| Slot 계산·유효성 검사·dispatch 간격 | 0.423 | 0.428 | +1.16% |
| PS root 메타데이터 준비 | 0.206 | 0.204 | -1.02% |
| Graph 입력 버퍼 준비 | 0.218 | 0.222 | +2.00% |
| PS graph: forward + tree 선택 | 2.864 | 2.857 | -0.27% |
| 기타 간격·출력 변환·cache merge | 0.466 | 0.469 | +0.57% |
| **PS 전체** | 4.965 | 4.948 | -0.34% |
| Target forward | 15.512 | 15.513 | +0.01% |
| DS graph | 6.598 | 6.587 | -0.17% |
| 준비 네 구간 합 (중복 합산 금지) | 1.634 | 1.622 | -0.74% |
| PS 전체에서 graph를 뺀 시간 (중복 합산 금지) | 2.100 | 2.091 | -0.45% |

PS 전체는 proxy 수신 완료부터 cache merge 완료까지의 draft process 구간이다. Dispatch는 build 끝부터 첫 prepare 시작까지의 관찰 간격이며, CPU 단독 계산 시간으로 부르지 않는다. Graph는 forward뿐 아니라 tree 선택/sampling도 포함한다. 중첩 event를 이중 합산하지 않는다.

| 반복 | 실행 순서 | 이전 PS ms | 이후 PS ms |
|---|---|---:|---:|
| 1 | A→B | 4.975 | 4.968 |
| 2 | B→A | 4.954 | 4.928 |

| 현재 검증 source | step 쌍 | 이전 PS ms | 이후 PS ms | 차이 95% 구간 ms |
|---|---:|---:|---:|---:|
| miss | 222 | 4.951 | 4.959 | -0.018~+0.038 |
| DS hit | 542 | 4.961 | 4.951 | -0.045~+0.035 |
| PS hit | 830 | 4.971 | 4.942 | -0.070~+0.034 |

Source는 현재 verification 입력의 출처이고 PS 시간은 다음 cache를 만드는 시간이다. 이번 readback 제거는 **B1 tree-hit** 경로에 적용되며 miss와 B>1 전체 개선을 주장하지 않는다.

## 검증과 측정 조건

- 후보 runtime의 **386 tests / 40 suites**, 실패·오류·skip 0. 검사 전후 runtime/bench/test hash 동일. [검사 결과](checks/CHECKS.json), [소스 증거](checks/CHECKS_PROVENANCE.json). 현재 유지한 runtime은 앞선 382-test 검증 소스와 정확히 일치하며, 이번 386개 검사는 보관된 후보 소스의 결과다.
- Profile: 동일 24턴 × AB/BA 두 쌍, 출력 **48/48 동일**, step source·AL·cache hit 동일. 1,594개 matched step 쌍, 통계 기여 문항 22개. Full dataset의 AL 개선 평가가 아니다.
- Profile OFF: 별도 48턴 출력 **48/48 동일**, AL 2.155954, hit 83.0593%로 전후 동일. Graph 준비 영향 step 양 arm 0. [보조 측정](UNPROFILED.json).
- Graph가 두 번에 한 번 거절되도록 강제해 실제 fallback **148회씩** 실행, 8턴 출력 **8/8 동일**. [fallback 증거](FALLBACK.json). 이 실행은 회귀 검사와 겹쳐 성능 근거로 사용하지 않는다.
- Llama2-7B + AMD135M, BF16, B1, T0.7, seed9101, RTX4090 GPU4/5. CPU16–23/48–55. Exit20, K1/K2=6/3, G=M=12/8, C3, DS 폭26, PS 폭15/root10, frontier. 파라미터 탐색 없이 고정했다.
- SpecBench raw turns 파일에서 균등 간격 선택, 입력 상한512/출력96/문맥2048, warmup4. 양 arm 동일 cold-prefix 및 graph prewarm all.
- 성능 arm은 전체 회귀 종료 후 순차 실행했다. GPU clock은 고정하지 않았으며 telemetry를 보존했다. 다른 GPU 작업의 시스템 자원 영향까지 완전히 통제한 실험은 아니다.

## 그림·재현·남은 비용

- [구간별 전후 PDF](BREAKDOWN_COMPARISON.pdf), [PNG](BREAKDOWN_COMPARISON.png), [수치·CI](BREAKDOWN.json).
- [동일 UID/step의 native timeline 전후](native/NATIVE_BEFORE_AFTER.pdf): DS hit, PS hit, miss 각각 한 쌍. Draft 점선은 forward 횟수 표시이며 실측 forward 경계가 아니다.
- [직전 최적화 대비 후보 runtime patch](RUNTIME_INCREMENTAL.patch), [실행별 provenance](PROVENANCE.json). 후보가 변경한 runtime은 `draft_runner.py`, `helpers/p2_prepare.py` 두 파일이다. 현재 runtime에 이 patch가 적용됐다는 뜻은 아니다.
- Raw는 `RAW_RESULTS.tar.gz`와 `/data/chokwans99/duet-ps-device-20261015`에 보관한다. 재분석: `python results/mlsys_ver11/ps_device_20261015/analyze.py --raw /data/chokwans99/duet-ps-device-20261015/runs --output results/mlsys_ver11/ps_device_20261015 --reps 1 2`.
- 남은 metadata 유효성 검사 readback, graph 입력 버퍼 복사, slot 구성, cache merge 등은 남아 있다. CPU–GPU 동기화가 모두 제거됐다는 결과가 아니다. 이 결과의 수치를 70B/Blackwell/B>1에 일반화하지 않는다.
