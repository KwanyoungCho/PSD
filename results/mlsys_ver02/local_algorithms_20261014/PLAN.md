# 로컬 연구 알고리즘 이식 및 비교 계획

기준은 MLSys-ver0.2@18e912f. 다른 서버의 연구 코드는 가져오지 않는다.

- Root: complement e(1-q), 허용 vocabulary 전체 합 정규화, terminal mass .75 observed + .25 overlap.
- Tree: historical calibration reach + gain allocation + unexpanded frontier. 표 출처와 지원 범위를 로그/문서에 명시한다.
- 실제 proposal q, ordered-WOR sampling/verification, G>M prefix, KV 계약은 유지한다.
- B1/B>1, eager/graph, dense/flat의 정책을 맞춘다. T0은 soft budget score, target argmax 검증 유지.
- 운영 기본 조합은 하나. 실험에서는 baseline/root-only/tree-only/combined를 분리해 같은 예산으로 비교한다.
- 먼저 독립 수학 참조/분포 보존/경계/전체 회귀, 다음 실제 모델 smoke, 마지막 full480 첫 턴 비교를 수행한다.
- Root 목적은 cache hit, tree 목적은 AL. 부가 proxy/P1/P2/verification latency와 TPS를 기록한다.
- 기존 70B scalar table은 calibration 학습이 없는 정리와 다르다. 이식 모델에서도 보편적인 우위를 주장하지 않는다.
- 미완료/실패/간섭 실행은 기록하고 완료나 성능 근거로 세지 않는다.
# Profile 표본 고정 메모

Full 비교와 별도로 수행하는 48문항 profile은 Spec-Bench의 6개 group에서 원래 순서의 0,10,...,70번째 문항을 각각 뽑는다. 앞의 MT-bench 48문항에 치우치지 않도록 profile 실행 전에 고정했다. PROFILE_QUESTIONS.json에 보존한다. Profile은 비용·phase 관측용이며 primary 품질/TPS 표와 합산하지 않는다. 장시간 event 보존의 간섭을 막기 위해 process별 최대 18,000개 event를 기록하고 실제 누락은 별도 보고한다.
