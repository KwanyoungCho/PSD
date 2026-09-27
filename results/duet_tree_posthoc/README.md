# DUET tree posthoc 재개 안내

먼저 [FINDINGS.md](FINDINGS.md)의 해석을 읽고, [REPORT.md](REPORT.md)의 전체 수치와
[THEORY.md](THEORY.md)의 증명을 연결한다. 사용자 이해 확인 TODO는 별도로 미완료다.

전체 480문항/560턴 × 두 정책을 실행했다. 모든 served tree를 진단했고 원본 분포는
매32번째 tree에 보관했다. 기존 3-seed 성능 결과와 이번 계측 자료를 혼합하지 않는다.

- `features.py`, `observe.py`, `launch.py`: 기존 전체-corpus harness에 수동적 observer를 추가.
  생성/verification 정책은 바꾸지 않는다. 기존 데이터가 있으면 재실행으로 덮어쓰지 않는다.
- `calibrate.py`, `analyze.py`: 기존8문항과 question-grouped cross-fit 보정, exact reach,
  예산별 sampled-pool DP, observed-node 위치 선택/종료 구조 분석.
- `fanout.py`: g1/g2의 정확한 O(V log V) 적분, 원래8문항의 phase gain curve,
  두 부모·새 node 예산2의 사전 배분 의사결정 평가.
- `sibling_bound.py`: frozen table이 같은 부모의 첫 형제를 항상 우선한다는 증명/완전 열거.
- `examples.py`: 실제 입력/history로 복원한 실패 사례와 feature-bin 충돌.
- `validate.py`, `audit_math.py`, `check_statistics.py`: 전체 이벤트 대조, raw 재계산,
  실제 float32와 참조 산술, 기대 AL/분산, 집계/feature 사용 검사.
- `finalize.py`: 두 수집이 끝난 뒤 독립 분석을 실행. `freeze.py`: 해석 검토 후 checksum 동결.
- `q_path_*`와 `smoke_*` 파일은 중간 확인 자료다. 최종 결론은 prefix 없는
  `analysis.json`, `fanout_analysis.json`, `audit.json`에 근거한다.

현재 inference AL 상승을 새로 확인한 정책은 없다. 유망한 결과는 같은 관측 대안과
국소 예산에서의 gain 기반 fanout 배분이며, 전체 C=3/다중 round online 확인이 다음 단계다.
사후 subset DP는 진단용이고 token-dependent pruning으로 배포하지 않는다.

재집계만 할 때 GPU 수집은 다시 돌릴 필요가 없다.

```bash
ssd/.venv/bin/python results/duet_tree_posthoc/validate.py
ssd/.venv/bin/python results/duet_tree_posthoc/analyze.py
ssd/.venv/bin/python results/duet_tree_posthoc/fanout.py
ssd/.venv/bin/python results/duet_tree_posthoc/examples.py
ssd/.venv/bin/python results/duet_tree_posthoc/sibling_bound.py
ssd/.venv/bin/python results/duet_tree_posthoc/make_report.py
```

보고서/분석을 수정하면 동결 manifest와 차이가 생긴다. 원시 기록과 기존 실험을
보존하고 수정 이유를 남긴 다음 `freeze.py`로 새 연구 산출물만 재동결한다.
