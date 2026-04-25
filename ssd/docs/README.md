# SSD 프로젝트 문서

이 디렉토리는 SSD 엔진에 추가된 두 가지 큰 작업 — **AWQ W4A16 양자화**와
**MESA-SSD (early-exit proxy 기반 speculative decoding)** — 의 계획,
구현 이슈, 실험 결과를 한글로 정리한다. 기존에 root 에 흩어져 있던 18개
.md 파일을 카테고리별로 통합한 결과이다.

## 카테고리

### `quantization/` — 양자화 (AWQ W4A16)

| 파일 | 내용 |
|------|------|
| [`01-plan.md`](quantization/01-plan.md) | 양자화 통합 계획. INT8/INT4 (torchao) v1 → AWQ Marlin v2 로 방향 전환한 history 와 v2 의 phase별 설계 |
| [`02-impl-issues.md`](quantization/02-impl-issues.md) | 구현 중 발생한 이슈 트래커. v1 (torchao 기반) 과 v2 (AWQ 기반) 양쪽의 시간순 기록 |
| [`03-final-report.md`](quantization/03-final-report.md) | 최종 결과 리포트. 34B/70B 실험 + draft AWQ 비교 + dense vs AWQ breakdown |

### `mesa/` — MESA-SSD (early-exit proxy)

| 파일 | 내용 |
|------|------|
| [`01-design.md`](mesa/01-design.md) | MESA-SSD 설계 — TreeLayout, Budget Split, Split CudaGraph, Rev1 (Policy A/B) |
| [`02-impl-issues.md`](mesa/02-impl-issues.md) | 구현 이슈 트래커 + Rev1 수정 항목 (B=1 assert, proxy_top_k 확대, fallback 제거 등) |
| [`03-results.md`](mesa/03-results.md) | MESA 실험 결과 — 8B/7B 초기 측정, Rev1 Policy A 결과, parameter sweep, 34B Rev1 final |

## 영문 문서

`README.md` (이 폴더 외부, repo root) 는 upstream SSD 의 영문 가이드.

## 문서 상태

- **AWQ Marlin** 은 main path. torchao 경로는 internal fallback 으로만 유지
  (legacy CLI 제거 완료).
- **MESA-SSD Rev1** 은 B=1 + non-EAGLE 범위에서 검증 완료. 34B 실험 결과
  parameter sweep 안정.

## 원본 위치 (참고용)

이 docs 가 통합한 원본 파일들 — 실제 파일은 정리 후 삭제됨:

```
ssd/
├── INT8-WEIGHT-ONLY-PLAN.md           → 01-quantization/01-plan.md
├── INT8-WEIGHT-ONLY-PLAN-v2.md        → 01-quantization/01-plan.md
├── INT8-WEIGHT-ONLY-PLAN-v2-KR.md     → 01-quantization/01-plan.md (한글본 통합)
├── INT8-IMPL-ISSUE.md                 → 01-quantization/02-impl-issues.md
├── INT8-v2-IMPL-ISSUE.md              → 01-quantization/02-impl-issues.md
├── INT8-v2-IMPL-ISSUE-KR.md           → 01-quantization/02-impl-issues.md
├── FINAL_REPORT.md                    → 01-quantization/03-final-report.md (v1 부분)
├── AWQ-v2-FINAL-REPORT.md             → 01-quantization/03-final-report.md
├── AWQ-v2-FINAL-REPORT-KR.md          → 01-quantization/03-final-report.md
├── MESA-IMPL-PLAN.md                  → 02-mesa/01-design.md
├── MESA-BREAKDOWN-PLAN.md             → 02-mesa/01-design.md (profiling 부분)
├── MESA-rev1.md                       → 02-mesa/01-design.md (Rev1 부분)
├── MESA-rev1-problems.md              → 02-mesa/02-impl-issues.md
├── IMPL_ISSUE.md                      → 02-mesa/02-impl-issues.md
├── MESA-RESULTS.md                    → 02-mesa/03-results.md
├── MESA-rev1-RESULTS.md               → 02-mesa/03-results.md
├── MESA-SWEEP-RESULTS.md              → 02-mesa/03-results.md
└── past_plans/                        → 위 항목들의 중복 백업 (삭제됨)
```

각 실험의 raw 결과 (`tmp/final_exp*/REPORT.md`) 는 해당 실험 디렉토리에
유지된다 — 실험 raw data 와 plot 이 함께 있는 게 자연스럽기 때문.
