# DUET 문서 안내

현재 P2 동적 트리의 설계, 구현, 검증 결과는 다음 문서 하나를 기준으로 한다.

- [TREE_IMPLEMENTATION.md](TREE_IMPLEMENTATION.md): 논문 작성과 구현 감사를 위한 현재 기준 문서

기존 DUET 일반 문서 01--14와 16은 이 폴더에 원래 이름으로 유지한다.
`internal/`에는 P2 tree 연구 과정에서 작성한 15번과 17--29번만 보관한다.
이 tree 문서들은 가설, 폐기된 정책, 당시의 중간 수치와 정정 전 결론까지
포함하는 역사 기록이다. 현재 tree 동작을 판단할 때는 직접 인용하지 말고,
위 기준 문서의 이력 절에서 원문 근거가 필요할 때만 참고한다. 충돌하면 코드와
기준 문서가 우선한다.

주요 실행 정책은 다음과 같다.

- 기본값 `eagle`: proxy 확률과 draft confidence로 매 라운드 확장할 부모를 동적으로 선택
- `off`: tree 도입 전과 같은 chain 비교군
- `coverage`: 모든 root의 깊이 4 backbone을 유지하는 고정형 품질 비교군
- `adaptive`, `confidence`, `level`, `frontier`: 연구 재현용 정책

`eagle`을 기본값으로 둔 것은 동적 정책을 주 연구 경로로 삼기 위한 결정이다.
현재 저장된 장기 실험만으로 `eagle`이 모든 workload에서 `coverage`나 chain보다
우월하다고 확정한 것은 아니다. 재현 결과와 남은 제한은 기준 문서에 함께 기록한다.
