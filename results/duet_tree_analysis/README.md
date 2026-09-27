# DUET tree 분석 재개

- 먼저 [REPORT.md](REPORT.md): 논문·코드 차이, 정확성 문제, 새 실험 결과.
- 다음 [THEORY.md](THEORY.md): 목적함수, 형제 순서, causal 구성, DP, 편향 반례.
- 연구 전체: [DUET_PROGRESS.md](../DUET_PROGRESS.md).

현재 결론: 원래 q 경로 곱은 논문과 구현에 있다. 수락 목적함수와는 다르다.
Calibration reach 점수는 holdout MSE를 개선했지만 실제 tree 생성 AL 우위는
확정하지 못했다. 분포 저장 없는 2-seed 실행의 AL은 1.905→2.092,
TPS는 68.968→70.286(+1.91%)였으나 CI에 0을 포함한다. Full-p/q 계측 실행은
반대 방향이었다. 두 결과를 구분하며 production 정책으로 채택하지 않는다.
G>M q rerank의 기존
lossless 주장은 실제 함수의 완전 열거 반례로 반박되었다.

실험은 모두 종료되었으며 GPU background job은 남기지 않는다.
원시 결과는 세 campaign의 manifest와 NPZ에 있다. 기존 결과를 덮어쓰지 않는다.

| 파일 | 역할 |
|---|---|
| `core.py` | 정확한 ladder/reach, ancestor+sibling-prefix DP |
| `audit_math.py`, `math_checks.json` | Production 함수와 비교, 완전 열거 편향 반례 |
| `collect.py`, `run_probe.py`, `probe_plan.json` | 실제 unpruned tree full-p/q 수집 |
| `analyze_probe.py`, `analysis.json` | Cal/val 분리, 예측 오차 및 subset 진단 |
| `calibration_frozen.json` | Cal 8 prompts만 사용한 scalar 추정치 |
| `score_hook.py` | 점수 한 문장을 실험 process에서 교체; G=M 강제 |
| `run_rollout.py`, `rollout_plan.json` | 검증 전 고정한 2-policy 실제 구성 비교 |
| `analyze_rollout.py`, `rollout_analysis.json` | Hit AL·step 진전 비교, TPS 제외 |
| `bench_score.py`, `run_benchmark.py`, `benchmark_plan.json` | 추가 verifier observer 없는 2-seed 실제 실행 |
| `analyze_benchmark.py`, `benchmark_analysis.json` | Prefill 포함 output TPS, AL, prompt-cluster CI |
| `comparison.png`, `comparison.pdf` | 예측/사후 subset/실제 구성을 분리한 그림 |
| `source_manifest.json` | 실행·분석 source와 주요 결과 checksum |

다음 단계는 scalar estimator를 그대로 배포하는 것이 아니다. Causal admission,
전 frontier, continuation gain, 가변 fanout, root prior, graph 비용을 분리해서
검증한다. 새 score/정책을 이 validation에 맞춰 고르면 다음 confirmation에는
새 prompt와 여러 seed가 필요하다. 사용자 수식 설명 TODO는 미완료로 유지한다.
