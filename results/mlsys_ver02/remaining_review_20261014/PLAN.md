# ver0.2 이후 남은 시스템 비용 검토

2026-10-14. 원격 `MLSys-ver0.2`는 사용자가 요청한 검증 완료 commit `0d5374289b8112db53c8f82ea7eafadc0c30a696`으로 push했다. 이 폴더는 그 이후의 검토이며, 기존 release의 `MANIFEST.json`/검사/원시는 수정하지 않는다.

## 검토 범위

- 다른 서버의 root·위치·tree 선정 알고리즘은 합치지 않는다. 같은 입력/후보/수식에서 CPU 준비, 전송, 확률 후처리 비용을 비교한다.
- 기존 채택: NumPy draft 입력 준비, 유효 context만 proxy/수락 계산, graph 사전 준비. 기존 보고서의 component/full 증거를 다시 구분해 평가한다.
- 이번 추가 후보: B>1 target verification 진입 시 draft q를 임시 padding tensor에 복사한 뒤 proxy 고정 buffer로 다시 복사하는 중복 제거. Dense와 compact 모두 고정 입력 buffer는 보존한다. Empty/폭 불일치 입력은 기존 padding 경로를 유지한다.
- 미채택 재검토: full target packed, mixed-miss AR, replica/stream 자동ON. 새 근거가 없으면 이전의 조건부/미채택 결론을 유지한다. 개선을 보이기 위해 알고리즘이나 파라미터를 바꾸지 않는다.

## 실행 순서 및 판단

1. 고정 logits/topology 대조. V=32000/128256, B=2/8 및 부분B3, N=4/8, payload 폭4, dense/compact. 매 shape seed10회 proxy 출력·수락·RNG·입력 buffer exact 일치. AB/BA 교대로 준비/전체 proxy+accept 시간을 측정한다. 반복 process도 수행한다.
2. 실제 `batch_tree_verify.verify` 경로 검사. Capture/eager × T0/T0.7, 긴/짧은/빈/불일치 폭, dense/compact 및 부분 batch, buffer 재사용. 이전 padding 계약과 직접 입력 경로를 비교한다.
3. 전체 25 suite를 최종 source로 실행한다. 실패·skip0 필요. 소스 변경 시 재검사한다.
4. Push된 clean checkout과 새 runtime의 실모델 전체480문항×2 pass를 같은 GPU pair에서 비교한다. 첫 턴 평가이며 main 서버의560턴과 다르다. 동일 후보·온도·seed·파라미터를 고정한다. 전체 TPS는 보조다. 수십µs component 개선을 전체 TPS 개선으로 과장하지 않는다.
5. 기준은 논문 a82f7d2 → ver01 f12d0f1 → 이미 검증된 ver02 0d53742 → 이번 단일 변경이다. 이번 incremental 실험을 원 논문 전체 대비 개선률로 쓰지 않는다.

## 자원·재현

- Runtime 기준 checkout: `/data/chokwans99/PSD-ver02-pushed-20261014`, clean detached `0d53742`.
- Candidate: `/home/chokwans99/PSD-MLSys-ver0.1`, `MLSys-ver0.2`, 변경 runtime은 `batch_tree_verify.py` 한 개.
- 현재 GPU4/5 full Llama2, GPU6 component/회귀. GPU7은 타 사용자의 작업으로 사용하지 않는다.
- 이전 원격 eslab18 SSH control socket은 없다. 원격 계산을 수행한 것으로 보고하지 않는다.
- 실험 계획: `l2_full_plan.json`, `l3_full_plan.json`. 실행 raw는 `/data/chokwans99/ver02-payload-*20261014*`.
- 단일 component 표본 CI는 반복 호출 변동이다. 온라인 문항 bootstrap CI 또는 여러 장비 검증과 구별한다.
