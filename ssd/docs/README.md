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
| [`01-design.md`](mesa/01-design.md) | MESA-SSD 설계 — TreeLayout, Budget Split, Split CudaGraph, Rev1 (Policy A/B), **Phase 2 Hybrid v1** (single batched cont+proxy forward, 8 CG buckets, HybridPhase2Plan) |
| [`02-impl-issues.md`](mesa/02-impl-issues.md) | 구현 이슈 트래커 — v1, Rev1 (B=1 assert, proxy_top_k 확대, fallback 제거), **Phase 2 Hybrid (Step 1..9D)** + sync fix + per-depth label fix + 9D build opt |
| [`03-results.md`](mesa/03-results.md) | MESA 실험 결과 — 8B/7B 초기 측정, Rev1 Policy A, parameter sweep, 34B Rev1 final, **Phase 2 Hybrid (8B 검증, 70B both-AWQ A/B/C)** |

## 영문 문서

`README.md` (이 폴더 외부, repo root) 는 upstream SSD 의 영문 가이드.

## 문서 상태

- **AWQ Marlin** 은 main path. torchao 경로는 internal fallback 으로만 유지
  (legacy CLI 제거 완료).
- **MESA-SSD Rev1** 은 B=1 + non-EAGLE 범위에서 검증 완료. 34B 실험 결과
  parameter sweep 안정.
- **MESA Phase 2 Hybrid** 는 Step 9D (build hot-path 최적화) 까지 완료. 8B
  검증 환경 + 70B both-AWQ 200×256 prompts 측정에서 hybrid TPS +2.4% (vs
  pre-hybrid baseline) 확인. split fallback 회귀는 sync fix 로 사실상 닫힘
  (-4.4% → -0.2%). 자세한 내용은 `mesa/03-results.md` Part 5.

## 원본 위치 (참고용)

이 docs 가 통합한 원본 파일들 — 실제 파일은 정리 후 삭제됨:

```
ssd/
├── INT8-WEIGHT-ONLY-PLAN.md              → quantization/01-plan.md
├── INT8-WEIGHT-ONLY-PLAN-v2.md           → quantization/01-plan.md
├── INT8-WEIGHT-ONLY-PLAN-v2-KR.md        → quantization/01-plan.md (한글본 통합)
├── INT8-IMPL-ISSUE.md                    → quantization/02-impl-issues.md
├── INT8-v2-IMPL-ISSUE.md                 → quantization/02-impl-issues.md
├── INT8-v2-IMPL-ISSUE-KR.md              → quantization/02-impl-issues.md
├── FINAL_REPORT.md                       → quantization/03-final-report.md (v1 부분)
├── AWQ-v2-FINAL-REPORT.md                → quantization/03-final-report.md
├── AWQ-v2-FINAL-REPORT-KR.md             → quantization/03-final-report.md
├── MESA-IMPL-PLAN.md                     → mesa/01-design.md
├── MESA-BREAKDOWN-PLAN.md                → mesa/01-design.md (profiling 부분)
├── MESA-rev1.md                          → mesa/01-design.md (Rev1 부분)
├── MESA-rev1-problems.md                 → mesa/02-impl-issues.md
├── IMPL_ISSUE.md                         → mesa/02-impl-issues.md
├── MESA-RESULTS.md                       → mesa/03-results.md
├── MESA-rev1-RESULTS.md                  → mesa/03-results.md
├── MESA-SWEEP-RESULTS.md                 → mesa/03-results.md
├── MESA-PHASE2-HYBRID-IMPLEMENTATION-PLAN.md → mesa/01-design.md (Part 5)
├── MESA-PHASE2-HYBRID-ISSUE.md            → mesa/02-impl-issues.md (Part 3)
├── MESA-PHASE2-HYBRID-REPORT.md           → mesa/03-results.md (Part 5)
├── MESA-PHASE2-HYBRID-FINAL-REPORT.md     → mesa/03-results.md (Part 5)
└── past_plans/                            → 위 항목들의 중복 백업 (삭제됨)
```

각 실험의 raw 결과 (`tmp/final_exp*/REPORT.md`) 는 해당 실험 디렉토리에
유지된다 — 실험 raw data 와 plot 이 함께 있는 게 자연스럽기 때문.
