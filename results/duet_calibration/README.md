# DUET Calibrator

2026-09-13 사전 calibration 기반 파라미터 선택 feasibility 실험.
기존 calibration 추천은 참고만 하고 시간 모델과 선택 절차를 새로 만들었다.

- [전체 연구 진행 기록과 재개 안내 — 2026-09-21](../DUET_PROGRESS.md)
- [결론과 상세 실험](REPORT.md)
- [수식과 적용 조건](THEORY.md)
- [주요 숫자](NUMBERS.txt), [전체 집계 JSON](NUMBERS.json)
- [추천 설정](recommended.json), [CLI 인자](recommended_args.txt)
- [동결 모델의 19개 검증](VALIDATION.md), [독립 confirmation](CONFIRMATION.md)
- [재현 및 후속 작업](HANDOVER.md)
- [이전 수식 설명의 미완료 TODO](../residial_dist/training_free/TODO.md)

핵심은 비용으로 후보를 줄인 뒤 실제 품질/처리율을 확인하는 것이다.
시간식만으로 모든 파라미터의 전역 최적을 보장하는 도구는 아니다.
