# P1 direct-child exclusion fix

동적 P1의 다음 배열 token 하나 제외를 같은 prefix의 기존 직접 자식 전체 제외로 수정했다. 기본값 수정 `4c7e4338`을 기준으로 한 로컬 실험 분기이며 새 알고리즘이나 verifier로 교체하지 않았다.

## 변경과 정확성

- context reach 계산에 이미 필요한 parent 배열을 후보 제외에도 공유한다. 다른 prefix의 동일 token은 제외하지 않고 leaf는 아무 token도 제외하지 않는다.
- U+P−1개 이하의 작은 top 후보 표만 검사한다. vocabulary 크기의 mask/copy는 추가하지 않는다. 한 fused Triton kernel이 모든 직접 자식을 제외하고 U개를 선택한다.
- root의 q는 원분포에서 그대로 구한다. 제외 후 재정규화하지 않으며 residual ladder, WOR sampling, 수락/복구 수식과 모델은 변경하지 않았다. 관련 핵심 파일 SHA는 RESULTS.json에 있다.
- 기존 scalar-index 대입은 GPU fill로 바꿔 pageable CPU→GPU copy를 없앴다. 실제 CPU readback은 추가하지 않았다. profiler의 일반 scalar 연산 이름만으로 GPU readback이라고 판정하지 않고 DtoH 전송 여부를 확인했다.
- P/K를 실행 입력으로 받아 native context1..24가 같은 kernel specialization을 재사용한다. context6 warmup 뒤1..24를 모두 실행해 JIT cache가 늘지 않는 것을 검사했다. 새 cold-start JIT 자체는 여전히 warmup에 필요하며 새 크기마다 steady-state JIT를 추가하지 않는다.
- 작은 vocabulary에서 U개 대안을 채울 수 없으면 valid=false padding으로 처리하고 PS dedup에서 제외한다. 실제32k vocabulary의 hot path에서는 모든 root가 유효해 기존처럼 mask=None이다.

## 동일 입력의 준비 함수 시간

BF16/V32000/U4, persistent 출력 버퍼. CPU affinity/단일 CPU thread 설정, 수정 전후 순서를 교대로 바꿔 각각16블록×100회를 측정했다. 아래는 eager 함수 전체의 CPU 제출부터 GPU 완료까지의 wall 평균이다. 측정 profiler는 별도 호출에만 켰다. 전체 step 시간이 아니다.

| Context / roots | 기존 µs | 수정 µs | 차이 µs |
|---|---:|---:|---:|
| 6 / 24 | 513.66 | 508.06 | -5.60 |
| 16 / 64 | 564.52 | 549.64 | -14.88 |
| 24 / 96 | 598.27 | 579.68 | -18.59 |

원시 블록별 분산은 MICROBENCH.json에 보존했다. 작은 µs 차이를 일반적인 시스템 성능 향상으로 확대 해석하지 않는다. 세 형상의 별도 프로파일에서 compute kernel 수는65→59였고, 수정 후 DtoH/HtoD 전송이 없으며 기존 pageable HtoD1회는 제거됐다.

## 수정 전후 실제 step 비교

로컬 RTX4090의 PIX [1,2]·[4,5]·[6,7], seed42·123·2026. 각 seed의 전후는 같은 GPU 쌍, seed123은 수정군 먼저 실행했다. 양 phase tree on/last, BF16, B1, exit21, P1 K10/W56/위치당4/G=M23, P2 K5/R24/W28/G=M15, C3, async send on, graph path, profile off로 고정했다.

Spec-Bench의 같은84 raw턴·원본 token ID와 순서, 출력 상한128, 문맥2048, Ttarget=Tdraft0.7, top-p1, 자연 EOS/기존 문맥 보호. 첫3턴 warmup, 나머지81턴 측정이다. 원래 helper의 frozen 사본만 선택하는 대조군과 수정 helper를 같은 runtime/harness로 비교했다. ROOT_IMPLEMENTATION_target/draft.json으로 각 process의 함수 선택을 확인했다.

6개 실행 공통 유효 61문항·70턴의 문항 평균 → 동일 가중3-seed 평균이다. 전체480문항/560턴·출력1024의 논문 성능 재평가는 아니다.

| 군 | P1 비율 % | P1 AL | P2 비율 % | P2 AL | 전체 hit % | 전체 AL | Step ms | TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 기존 | 43.180 | 3.160 | 44.804 | 2.669 | 87.983 | 2.859 | 19.122 | 147.847 |
| 수정 | 46.953 | 3.034 | 41.716 | 2.570 | 88.669 | 2.798 | 19.095 | 144.683 |

전체 step 변화 -0.0269ms (-0.141%). TPS 변화 -2.140%. root 구성 변경으로 hit 경로·실제 tree 크기·stochastic 출력도 바뀌므로 이 값에는 workload 변화가 포함된다. 새 후보 연산의 순수 비용은 위 동일 입력 microbenchmark로 분리했다.

| Seed | 군 | P1 step ms | P2 step ms | Miss step ms | 전체 step ms | 긴 step 제외 |
|---|---|---:|---:|---:|---:|---:|
| 42 | before | 19.375 | 18.571 | 20.633 | 19.173 | 0 |
| 42 | after | 19.334 | 18.474 | 20.441 | 19.090 | 0 |
| 123 | before | 19.347 | 18.550 | 20.466 | 19.137 | 0 |
| 123 | after | 19.330 | 18.526 | 20.456 | 19.126 | 0 |
| 2026 | before | 19.294 | 18.505 | 20.428 | 19.057 | 1 |
| 2026 | after | 19.306 | 18.466 | 20.346 | 19.070 | 0 |

기존 cap/clip 및 요청별·source 구성별2×중앙값 초과 step 제외를 시간·토큰·AL·hit에 함께 적용했다. 제외 수는 위 표, 원시/필터 없는 지표와 모든 분모는 RESULTS.json에 있다. 출력2토큰 미만/무생성 지표를0으로 대체하지 않았다.

## 검증과 재현

CHECKS.log 및 JIT_REUSE_CHECK.log: 직접 자식/leaf/다른 prefix/원래 q와 reach/무작위 topology/동점/고갈 padding/최대 fanout/CPU·CUDA/CUDA graph 재생 및 persistent 버퍼/커널 재사용을 검사했다. 기존 P1·ordered sibling·기본값·인자 동치 검사도 포함한다.

ROOT_CHECKS.log는 개발 중 초기 통과 기록, CHECKS.log는 최종 코드의 통합 검사다. 중간 A/B는 kernel specialization 점검 때문에 중단해 ab_superseded_shape_specializations에 보존했고 최종 결과에는 사용하지 않았다.

재현: 기존 SSD 환경에서 `PYTHONPATH=ssd:ssd/tests SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp python -B -m unittest -v test_p1_child_exclusion test_p1_dynamic_tree test_tree_live_siblings test_tree_policy_defaults test_args_cleanup_equiv`. GPU 선택은 해당 서버에 맞춘다. 성능 실험의 정확한 명령/env/SHA는 각 runs/*/PROVENANCE.json, 전체 조건은 PROTOCOL.json에 있다.

[요약 수치](SUMMARY.json) · [함수 측정](MICROBENCH.json) · [통합 검사](CHECKS.log) · [조건](PROTOCOL.json)

원시 결과와 전체 문항별 자료는 /home/chokwans99/PSD-ver16-p1-exclusion-review-20261023/results/mlsys_ver16/p1_child_exclusion_fix_20261023 에 보존되어 있다. 이 디렉터리는 검증·성능 요약만 배포한다.
