# SSD 프로젝트 문서

이 디렉토리는 SSD 엔진에 추가된 두 가지 큰 작업 — **DUET (early-exit proxy
기반 비동기 speculative decoding)** 과 **AWQ W4A16 양자화** — 의 설계, 구현
이슈, 실험 결과를 한글로 정리한다.

## 어디부터 볼 것인가

| 목적 | 문서 |
|------|------|
| **새 서버에서 실험을 돌린다** | [`duet/00-server-setup.md`](duet/00-server-setup.md) |
| 현재 tree 동작·실험 이력·논문 주장 경계를 확인한다 | [`duet/TREE_IMPLEMENTATION.md`](duet/TREE_IMPLEMENTATION.md) |
| 방법 수준의 DUET 명세를 본다 | 저장소 루트 [`MESA-SSD.md`](../../MESA-SSD.md) |

`MESA-SSD.md`는 초기 연구명이 파일명에 남아 있을 뿐 현재 기법 이름은 **DUET**
이다. 문서 폴더도 과거 `mesa/`에서 `duet/`으로 이름이 바뀌었다.

## 카테고리

### `duet/` — DUET (early-exit proxy 비동기 SD)

| 파일 | 내용 |
|------|------|
| [`00-server-setup.md`](duet/00-server-setup.md) | **새 서버 셋업·실행 가이드.** 외부 의존 자산, GPU 요구량, 환경 변수, 정본 실행 명령, 캘리브레이션, 측정 방법론, 함정 |
| [`TREE_IMPLEMENTATION.md`](duet/TREE_IMPLEMENTATION.md) | 동적 tree의 기준 문서. 알고리즘, CUDA Graph 실행기, target 검증, 번호가 붙은 실험 결과, 지원 범위, 논문에서 주장 가능/불가능한 것 |
| [`README.md`](duet/README.md) | duet 폴더 안내와 현행 정책 이름 |
| `01-design.md` … `03-results.md` | 초기 설계·구현 이슈·결과 |
| [`04-split-k1k2-design.md`](duet/04-split-k1k2-design.md) | split-K1/K2 실행 계약 (현재 유일한 DUET 경로) |
| `05-policy-b-fix.md`, `06-timeline-cleanup-plan.md` | Policy B 확정, 타임라인 정리 |
| [`07-qwama-draft-support.md`](duet/07-qwama-draft-support.md) | Qwama draft 지원 검토 |
| `08-proxy-overlap-experiment.md`, `09-beat-sd-best-plan.md` | proxy overlap 실험, SD 대비 계획 |
| [`10-swiftspec-analysis.md`](duet/10-swiftspec-analysis.md) | SwiftSpec(ASPLOS 2026) 분석 |
| `11-kv-promo-design.md`, `12-experiment-summary.md` | KV promotion 설계, 실험 요약 |
| [`13-b-gt-1-design.md`](duet/13-b-gt-1-design.md), [`14-b-gt-1-code-review.md`](duet/14-b-gt-1-code-review.md) | **`B>1` 확장 설계와 코드 리뷰 (미구현)** |
| `16-args-config-cleanup.md` | CLI/config 정리 |
| `internal/` | P2/P1 tree 연구 과정의 15, 17–30번 노트 |

`internal/`은 가설, 폐기된 정책, 정정 전 중간 수치를 포함한 **역사 기록**이다.
현재 동작을 판단할 때 직접 인용하지 말고, 기준 문서의 이력 절에서 원문 근거가
필요할 때만 참고한다. 충돌하면 코드와 기준 문서가 우선한다.

### `quantization/` — AWQ W4A16

| 파일 | 내용 |
|------|------|
| [`01-plan.md`](quantization/01-plan.md) | 양자화 통합 계획. INT8/INT4(torchao) v1 → AWQ Marlin v2 전환 history와 v2 phase별 설계 |
| [`02-impl-issues.md`](quantization/02-impl-issues.md) | 구현 이슈 트래커 (v1/v2 시간순) |
| [`03-final-report.md`](quantization/03-final-report.md) | 최종 리포트. 34B/70B + draft AWQ 비교 + dense vs AWQ breakdown |

## 현재 상태 (2026-09)

- **성능 champion은 chain**이다. `--duet_p1_tree_policy off --duet_p2_tree_policy off`.
  동적 tree는 accepted length를 올리지만 이 GPU 배치에서는 target step 증가가
  이를 상회해 TPS가 낮다. 자세한 판정 근거와 조건은 `TREE_IMPLEMENTATION.md`
  §8.16과 §14를 본다.
- **지원 범위는 `B=1` + `temperature > 0` + Llama 계열 + `vocab ≤ 32768`**이다.
  `B>1`과 `temp=0`은 chain fallback으로 동작한다.
- **AWQ Marlin**이 양자화 main path다. torchao 경로는 internal fallback으로만
  유지한다(legacy CLI 제거 완료).
- 논문용 실행 드라이버와 지표 스크립트는 이 저장소 밖에 있다. 반드시
  [`duet/00-server-setup.md`](duet/00-server-setup.md) §1을 먼저 읽는다.

## 영문 문서

저장소 루트의 `README.md`는 upstream SSD의 영문 가이드다.
