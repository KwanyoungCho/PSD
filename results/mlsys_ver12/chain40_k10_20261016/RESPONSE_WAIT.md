# PS 초과와 response wait의 관계

기존 원시 기록을 추가 분석했다. 새 inference나 runtime 변경은 없다.

`target_recv_response_wait`는 target이 요청을 보낸 뒤 draft의 응답 metadata/token과
q logits를 받을 때까지의 구간이다. PS 완료만 기다리는 구간은 아니다. 다음 순서로 동작한다.

1. Target step n의 응답 수신 → verification n.
2. 그와 병렬로 draft가 DS n, PS n을 수행하고 cache를 만든다.
3. Target이 step n+1의 후보를 요청한다.
4. Draft는 PS n과 정리를 마친 뒤 명령을 수신하고 cache를 조회한다.
5. Hit이면 준비한 chain/tree를 응답하고, miss이면 JIT draft 후 응답한다.
6. Target은 응답을 받은 뒤 verification n+1을 수행한다.

따라서 PS n의 초과 시간은 response wait n+1과 대응시켜야 한다. 현재 구현은 P1 hit라도
PS를 중단해 먼저 응답하는 구조가 아니다. 또 기준 deadline은 target 요청 **시작**이며,
response wait는 요청 **송신 후** 시작하므로 두 지표의 시작 시각도 다르다.

## 실측 대조

이전 tree K3/W9 기준과 이번 chain K10/40후보의 동일한 보고서 모집단을 사용했다.
아래 값은 전체 source 평균이며 단위는 ms다.

| Target | 이전 tree response wait | 이번 chain response wait |
|---|---:|---:|
| 7B | 3.343 | 4.012 |
| 13B | 3.435 | 3.193 |

PS 초과가 없어진 것은 아니다. 같은 P1 hit끼리 관측 구간을 분리하면 다음과 같다.
`이전 PS가 아직 안 끝난 구간`은 response wait 시작부터 draft가 다음 명령 수신 준비를
마칠 때까지 겹친 시간을 뜻한다. 이후 구간에는 요청 처리·cache 조회·응답 구성·통신 등이
들어간다. 고립시킨 커널 비용이나 반사실적 개선량이 아닌 시간 구간 분할이다.

| Target / 방식 / P1 hit | 이전 PS가 아직 안 끝난 구간 | 이후 응답 처리 구간 | response wait 합 |
|---|---:|---:|---:|
| 7B tree | 0.004 | 3.136 | 3.140 |
| 7B chain | 1.941 | 1.882 | 3.823 |
| 13B tree | 0.007 | 3.257 | 3.264 |
| 13B chain | 0.855 | 1.916 | 2.770 |

Chain에서 PS tail 대기는 늘었지만 tree의 KV 복구·구조/parent-q 구성 등이 포함된 요청
처리 비용은 줄어드는 방향이다. 예를 들어7B P1 hit의 draft request receive는
0.939→0.287ms, hit-cache 처리 구간은1.865→1.012ms다. 서로 다른 GPU의 중첩 구간을
단순 합쳐 wall time으로 해석하지 않는다. 구현 근거는 `draft_runner.py`의
`_service_spec_request`와 `hit_cache_and_respond`다.

전체 평균에는 miss 비중도 영향을 준다.7B miss는14.49%→4.60%,13B는16.11%→6.47%다.
반면 miss 조건부 response wait는7B4.138→8.790ms,13B4.219→7.569ms로 늘었다.
이번에는 K와 후보 수·검증 입력까지 바뀌었으므로 tree/chain 구조만의 A/B 실험은 아니다.

이번 chain 안에서 P1 hit를 이전 PS 완료 여부로 나누면:

| Target | 이전 PS가 요청 전에 끝남 | 이전 PS가 요청보다 늦음 |
|---|---:|---:|
| 7B | 1.185ms (169step) | 4.590ms (581step) |
| 13B | 1.234ms (153step) | 3.179ms (575step) |

입력/상태가 다른 관측 집단이므로 이 차이 자체를 고립된 causal penalty로 사용하지 않는다.
다만 PS 초과가 무해하거나 다음 verification과 전부 겹쳐 숨겨졌다는 해석은 맞지 않는다.

7B P1 hit 대표 timeline에서는 step62의 PS가 다음 요청 시작보다2.612ms 늦고,
step63의 response wait는4.206ms다. 같은 그림의 step62 response wait와 비교하면
인과 순서가 한 step 어긋난다.

코드:
- `ssd/ssd/engine/speculator_async.py`: `_speculation_request`의 request send / response recv.
- `ssd/ssd/engine/step.py`: speculate 응답을 기다린 뒤 verify.
- `ssd/ssd/engine/draft_runner.py`: `draft_loop`는 DS/PS/cache 완성 뒤 다음 명령을 처리.

재현: `python -B results/mlsys_ver12/chain40_k10_20261016/diagnose_response_wait.py`.
모든 source/완료 여부 집계와 출처 해시는 `RESPONSE_WAIT_SUMMARY.json`, 개별 step은
`RESPONSE_WAIT_DIAGNOSIS.json.gz`에 있다. CUDA→host 시각 정렬 오차의 한계는 기존과 같다.
