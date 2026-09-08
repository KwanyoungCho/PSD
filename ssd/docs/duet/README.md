# DUET 문서 안내

| 목적 | 문서 |
|---|---|
| 새 서버에서 실험을 돌린다 | [00-server-setup.md](00-server-setup.md) |
| 현재 tree 동작·실험 이력·논문 주장 경계 | [TREE_IMPLEMENTATION.md](TREE_IMPLEMENTATION.md) |
| 방법 수준의 DUET 명세 | 저장소 루트 [MESA-SSD.md](../../../MESA-SSD.md) |

기존 DUET 일반 문서 01–14와 16은 이 폴더에 원래 이름으로 유지한다.
`internal/`에는 tree 연구 과정에서 작성한 15번과 17–30번만 보관한다. 이
문서들은 가설, 폐기된 정책, 당시의 중간 수치와 정정 전 결론까지 포함하는
역사 기록이다. 현재 동작을 판단할 때는 직접 인용하지 말고, 기준 문서의 이력
절에서 원문 근거가 필요할 때만 참고한다. 충돌하면 코드와 기준 문서가 우선한다.

## 현행 실행 정책

공개 스위치는 phase별로 `off|on` 두 값만 쓴다. 두 phase를 하나의 스위치로
묶지 않는 이유는 P1만 또는 P2만 tree로 켜는 분해 실험이 필요하기 때문이다.

```bash
--duet_p1_tree_policy off|on      # proxy 도착 전 draft-source phase
--duet_p2_tree_policy off|on      # proxy 도착 후 proxy-source phase
```

| 설정 | 의미 |
|---|---|
| `off` / `off` | **chain. 현재 성능 기준선이자 champion** |
| `on` / `on` | P1·P2 모두 동적 tree. 논문의 token-axis/구조 실험용 |
| `off` / `on` | P2만 tree. phase 분해용 |
| `on` / `off` | P1만 tree. phase 분해용 |

동적 tree는 첫 forward에서 모든 root를 평가하고, 이후 라운드의 부모를 누적
`proxy × confidence` 점수로 전역 선택한다. P1은 proxy가 없으므로 같은 selector에
`도달확률 × 시작 token q`를 시작 점수로 넣는다.

`duet_p1_tree_allocation_policy`는 P1 root 사이의 예산 배분을 정한다.

| 값 | 의미 |
|---|---|
| `dynamic` | 전역 frontier 경쟁 (기본값) |
| `backbone` | P1 root마다 full-depth continuation을 보장. **P1 tree 확대 실험은 이 값을 명시한다** |
| `hybrid` | 위 둘의 절충 |

P1이 만드는 것은 *미래 cache key의 forest*라, P2식 전역 frontier는 나중에 실제로
hit할 root를 굶길 수 있다. 그래서 `backbone`이 따로 필요하다.

## deprecated 입력

`--duet_tree_policy`(`eagle`, `coverage`, `confidence`, `level`, `frontier`,
`adaptive`, `hybrid`)와 `--duet_tree_nv`, `duet_tree_beta`는 **과거 실험 재현
용도로만 남아 있다.** 새 실험에서는 쓰지 않으며, 공개 정책 이름으로도
사용하지 않는다.

## 현재 판정

성능 champion은 chain이다. 동적 tree는 accepted length와 step당 token을 올리지만
이 GPU 배치에서는 target verify row 증가가 이를 상회해 TPS가 낮다. target 검증이
상대적으로 싸거나 draft/target GPU 비율이 다른 서버에서는 캘리브레이션 후 다시
판정해야 한다. 근거와 조건은 `TREE_IMPLEMENTATION.md` §8.16, §14를 본다.
