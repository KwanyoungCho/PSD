# DUET P1/P2 on/off champion timeline matrix

수집 시각: 2026-08-08

## 고정 설정

- target: Layerskip Llama-2-70B AWQ, TP4 (GPU 0--3)
- draft: TinyLlama-1.1B AWQ (GPU 4)
- `K1=9`, `K2=4`, temperature 0.7, exit layer 56
- P1 dynamic: 18 node 생성 / 최대 14 node 검증
- P2 dynamic: 8 node 생성 / 8 node 검증
- P2 `R=W=10`, `C=3`
- `SSD_CHAIN_PROXY_GRAPH=1`, tree phase에서는 전체 phase CUDA Graph와
  all-page warmup 사용
- 네 조합 모두 seed 42, dataset별 1 prompt(총 4), output 256
- `SSD_PROFILE_DUET=1`, `SSD_PROFILE_DUET_DETAIL=1`

이 실행의 TPS는 상세 event가 켜진 진단값이며 champion 성능 수치로 사용하지
않는다. 목적은 같은 설정에서 target/draft 시간축의 구조를 비교하는 것이다.

## 생성한 그림

각 조합에 다음 13장을 생성했다.

- 기존 중앙 대표 그림: `hit_k1`, `hit_k2`, `miss` 각 1장
- 여러 대표 사례: 각 상태의 full-step duration p25/p50/p75, 총 9장
- 위 9장을 한눈에 보는 3×3 contact sheet 1장

Contact sheet 행은 `hit_k1 / hit_k2 / miss`, 열은 `p25 / p50 / p75`다.
개별 step id와 duration은 각 폴더의 `representatives.tsv`에 기록했다.

| P1 | P2 | 폴더 | contact sheet |
|---|---|---|---|
| off | off | [`p1_off_p2_off/`](p1_off_p2_off/) | [`timeline_representatives_contact_sheet.png`](p1_off_p2_off/timeline_representatives_contact_sheet.png) |
| on | off | [`p1_on_p2_off/`](p1_on_p2_off/) | [`timeline_representatives_contact_sheet.png`](p1_on_p2_off/timeline_representatives_contact_sheet.png) |
| off | on | [`p1_off_p2_on/`](p1_off_p2_on/) | [`timeline_representatives_contact_sheet.png`](p1_off_p2_on/timeline_representatives_contact_sheet.png) |
| on | on | [`p1_on_p2_on/`](p1_on_p2_on/) | [`timeline_representatives_contact_sheet.png`](p1_on_p2_on/timeline_representatives_contact_sheet.png) |

## 중앙 duration과 실행 확인

| P1/P2 | hit_k1 p50 | hit_k2 p50 | miss p50 | executor evidence |
|---|---:|---:|---:|---|
| off/off | 58.26ms | 49.51ms | 56.98ms | chain |
| on/off | 74.60ms | 49.36ms | 55.37ms | P1 replay 251 |
| off/on | 57.93ms | 61.26ms | 55.01ms | P2 replay 263 |
| on/on | 75.63ms | 61.21ms | 57.74ms | P1/P2 replay 262/262 |

이는 선택된 p50 representative의 full target step wall span이다. 서로 다른 token
trajectory의 작은 profile이므로 AL/TPS formal 비교로 해석하지 않는다. 다만 P1을
켰을 때 hit_k1 구간, P2를 켰을 때 hit_k2 구간이 선택적으로 길어지고, on/on에서
두 executor가 모두 실제 replay된다는 구조 확인에는 사용할 수 있다.

각 폴더에는 다음도 포함한다.

- `duet_profile_target_rank0_*.json`, `duet_profile_draft_*.json`: 원 profile
- `proxy_summary.txt`: cache 상태별 proxy 세부 구간 p50/p95
- `metrics.txt`: 해당 진단 run의 요약
- `images.txt`: 생성 그림 목록
- `representatives.tsv`: p25/p50/p75 step 선택 근거

## 재현

```bash
cd /home/chokwans99/PSD/ssd
bash experiments/proxy_async_overlap/tree_sweep/run_phase_combo_timelines_20260808.sh
```

이미 profile이 있으면 재실험 없이 그림만 다시 만들 수 있다.

```bash
RESUME=1 \
  bash experiments/proxy_async_overlap/tree_sweep/run_phase_combo_timelines_20260808.sh
```
