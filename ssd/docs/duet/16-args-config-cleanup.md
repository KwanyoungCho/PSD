# 16 — DUET 인자/config/env 정리 계획 (승인됨, 2026-08-03)

**상태**: 사용자 승인 완료 — 구현 순서는 P2-tree P0/T1 **이전** (v4의 새
노브를 깨끗한 네임스페이스에 얹기 위함). 전수 조사 근거는 워크플로우
duet-args-inventory (2026-08-03) — CLI/config/env 3면.

## 0. 진단 요약

1. **pfo는 유령**: CLI·config 필드가 아니라 `__post_init__`이
   `async_fan_out − duet_draft_fan_out`으로 만드는 동적 속성
   [config.py:337]. budget을 바꾸려면 `--f`와 dfo를 동시 조작해야 하고,
   산식 `pfo×(K1+1)`이 5개 파일에 인라인 중복 (config:157,241,426 +
   scheduler:58 + model_runner:310 + draft_runner:194,248 +
   verifier:421).
2. **`--k`/`--f`의 의미 전환**: 비-DUET에선 트리 형상, DUET에선 합계
   검증용(k=K1+K2)·파생 원천(f→dfo 기본값·pfo·miss JIT 폭)으로 격하 —
   같은 정보를 3중 입력.
3. **필수 env 모순**: split-K1/K2가 유일 경로인데
   `SSD_FORCE_SPLIT_K1K2=1`이 여전히 필수, 11개 지점 분산 읽기.
4. **데드/상수**: `--fl`(미사용), `--duet_split_phase2_fan_out_list`
   (지정=에러), `--duet_policy a`(거부), `--backup`(DUET 무시);
   env B==1 3종+TOPM(null 판정), `SSD_DUET_JIT_SUBSET`(무이득 판정),
   `SSD_ASYNC_PROXY_SEND`/`SSD_PROXY_STREAM`(미사용). 반면
   `SSD_DUET_JIT_SHORT`는 champion 표준인데 env 게이트.
5. **내부 이중 유도**: fan_out_list 기본값+MQ_LEN을 config와
   model_runner[:199-208]가 각자 계산; K2≤K1 검증이 config가 아닌
   draft_runner[:201-206]에 위치.

## 1. Tier 1 — CLI 재편 (별칭 추가, 구 플래그 유지 + deprecation 경고)

```bash
# 현재                                          # 제안
SSD_FORCE_SPLIT_K1K2=1 ... --duet \             ... --duet \
  --k 13 --f 3 \                                  --duet_k1 9 --duet_k2 4 \
  --duet_phase1_k 9 --duet_phase2_k 4 \           --duet_p1_fanout 2 \
  --duet_draft_fan_out 2 --duet_policy b \        --duet_p2_budget 10 \
  --duet_exit_layer 56  # +SSD_DUET_JIT_SHORT=1   --duet_exit_layer 56
```

| 구 | 신 (canonical) | 비고 |
|---|---|---|
| `--duet_phase1_k/phase2_k` | `--duet_k1` / `--duet_k2` | 별칭 |
| `--k` | 생략 가능 | k=K1+K2 자동 유도, 명시 시 일치 검증 |
| (없음: f−dfo 간접) | **`--duet_p2_budget N`** | 직접 지정; 미지정 시 기존 산식 폴백 (재현성 100%) |
| `--f` | DUET에선 생략 가능 | p1_fanout + ⌈budget/(K1+1)⌉ 유도 (miss JIT 폭 역할은 P2-tree T2에서 재검토) |
| `--duet_draft_fan_out` | `--duet_p1_fanout` | 별칭 |
| `--duet_split_phase1_fan_out_list` | `--duet_p1_fanout_list` | 별칭 |
| `--duet_policy` | deprecated (b 고정) | a는 이미 거부됨 |
| `--fl`, `--duet_split_phase2_fan_out_list` | 삭제 | 데드 |

## 2. Tier 2 — env 처분

| env | 처분 |
|---|---|
| `SSD_FORCE_SPLIT_K1K2` | **은퇴** — `--duet`이 곧 split (config 단일 필드화, 11개 읽기 정리; 과도기 무해) |
| `SSD_DUET_JIT_SHORT` | config 필드 승격 + **기본 ON** (`--duet_no_jit_short`로 해제) |
| `SSD_DUET_JIT_SUBSET` | 유지 (기본 OFF, A/B 재현용 — 무이득 판정 문서화) |
| B==1 전용 3종 + PROXY_TOPM | 동결 명시 (제거 안 함 — B=1 이력 보존) |
| `SSD_ASYNC_PROXY_SEND`/`PROXY_STREAM` | 동결 명시 |
| TRACE/PROFILE 계열, `SSD_DIST_PORT` | 유지 |
| 실험 스크립트의 죽은 export (MESA/KV_PROMO 등) | 불변 (역사 기록) |

## 3. Tier 3 — config 내부 단일화 (동작 불변)

1. `duet_p2_budget` 필드 신설; `duet_proxy_fan_out`은 호환 property로.
2. budget 산식 5중복 → `duet_proxy_total_budget` property 단일 소스.
3. fan_out_list 이중 유도(model_runner:199-208) 제거.
4. K2≤K1 검증을 config `__post_init__`로 이동 (명시 raise).

## 4. 검증 계획

- 동일성 유닛테스트: 신 인터페이스 Config ≡ 구 인터페이스 Config
  (champion + B별 승자 형상 전부, 전 필드 비교).
- B=1 champion 스모크: 신·구 명령 동일 TPS 밴드.
- 기존 테스트 43개 (M1-M6 38 + jit_subset 5) 회귀.
- 커밋 단위: Tier 1 → 2 → 3 각각 별도 커밋 + 스모크.
