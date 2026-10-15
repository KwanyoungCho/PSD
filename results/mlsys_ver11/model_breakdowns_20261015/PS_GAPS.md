# PS 시작 준비·빈 구간·다음 명령 대기 분석

> 그림은 [모델별 timeline 정본](../../mlsys_ver12/model_timelines/REPORT.md)으로 통일했다. 아래 해석과 수치는 과거 측정 기록이다.

2026-10-15. 기존 원시 profile 재분석이며 신규 inference/런타임 수정은 없다.

사용자가 첨부한 그림은 `l2_7b_final_profile`, UID `244_t0`, request epoch2,
step78(P1 hit)의 실제 이벤트를 그린 것이다. **막대가 잘못 늘어난 것은 아니지만,
기존 그림은 계측되지 않은 준비 구간을 빈칸으로 남겼고, 다음 요청 대기를 생략했다.**
따라서 이전 그림만 보고 빈칸을 GPU idle로 해석하면 안 된다.

- [번호를 붙인 해당 step](../../mlsys_ver12/model_timelines/llama2_7b_llama68m/TIMELINE.pdf)
- [모델별 최신 timeline PDF](../../mlsys_ver12/model_timelines/ALL_MODELS.pdf)
- [그림 모음](../../mlsys_ver12/model_timelines/index.html)
- [1,309개 step별 수치](PS_GAPS_AND_WAITS.csv), [통계·근거·원본 해시](PS_GAPS_AND_WAITS.json)

## 1. 첨부 step에서 각 구간의 의미

시간 기준은 기존 그림과 같은 target step 시작이다. 아래 값은 모두 ms다.

| 구간 | 시작→종료 | 길이 | 해석 |
|---|---:|---:|---|
| Proxy 수신 대기 종료 | 17.793 | — | 이 이후부터 PS 전체 구간을 계산 |
| ① 연두색 점 박스 `phase2_build` | 17.990→19.561 | 1.571 | PS root 선택/중복 처리, layout·실행 입력 구성 |
| ② 바로 뒤의 빈칸 | 19.561→20.738 | 1.177 | 별도 label이 없는 PS dispatch 준비. 아래 코드 참조 |
| 첫 번째 `p2_prepare` | 20.738→21.094 | 0.355 | root/seed 입력·가시성 등 rollout 입력 준비 |
| 두 번째 `p2_prepare` | 21.154→22.005 | 0.850 | executor의 고정 입력 버퍼 준비 |
| `p2_graph_replay` | 22.023→24.075 | 2.052 | PS 전체 graph replay; forward뿐 아니라 graph 내부 선택 연산도 포함 |
| 변환·merge 종료 | 24.665 | — | `merge_cache` 종료로 PS envelope 종료 |
| ③ merge 이후 계측되지 않은 구간 | 24.665→24.957 | 0.292 | key 무효화 등 후처리와 루프 전환이 있는 구간 |
| ④ `draft_recv_cmd` | 24.957→26.103 | 1.146 | 다음 target 명령 수신: 통신/대기를 포함 |
| 명령 수신 끝→요청 본문 수신 시작 | 26.103→26.126 | 0.023 | 이벤트 사이의 짧은 미계측 구간 |
| 다음 `draft_recv_request` | 26.126→27.208 | 1.082 | 다음 step79의 요청 수신/처리가 시작됨 |

PS 전체는 **6.872ms**, 그중 graph는 **2.052ms**, graph 밖 elapsed는
**4.820ms(70.1%)**다. 연두색 박스 1.571ms를 PS 전체라고 부르면 틀린다.
표에서 생략한 작은 이벤트 사이 간격도 전체 6.872ms에는 포함된다.
부모/자식 이벤트는 중첩될 수 있으므로 표에 있는 모든 막대를 합산해서 전체 시간을 만들지 않는다.

PS 종료→다음 요청 본문 수신 시작은 **1.461ms**이며,
`0.292 + 1.146 + 0.023 = 1.461`로 분해된다.
다음 명령을 기다리는 구간은 표시할 수 있지만 이 1.461ms 전체가 GPU idle이라는 뜻은 아니다.
`draft_recv_cmd`에도 통신이 포함되며 GPU kernel activity를 별도로 수집한 기록은 없다.
다음 step의 cache 응답·glue까지 기다린 시간을 idle로 포함하지 않았다.

### 추가 질문: 주황색 빗금, forward 횟수, 마지막 회색을 계산에 사용할 수 있는가

PS graph 앞의 주황색 빗금 두 개는 위 표의 `p2_prepare`다.
첫 번째 **0.355ms**는 rollout의 root·가시성·입력 구성, 두 번째 **0.850ms**는
executor 고정 버퍼에 token·점수·온도·KV slot·page 정보를 채우는 단계다.
같은 함수가 중복 실행됐다는 뜻은 아니다. 새 그림에서는 각각 `PS prepare A`와 `PS prepare B`로 구분했다.
DS 쪽 glue 안에 보이는 주황색 빗금은 별도 `tree_prep`이며 같은 PS 준비가 아니다.

DS/PS graph에는 K1/K2만큼 등분 점선과 번호를 추가했다. 이 step은 DS6/PS3회다.
**등분은 횟수 안내일 뿐, 각 forward의 실측 경계나 같은 latency를 뜻하지 않는다.**
준비·merge를 포함한 phase 전체를 forward로 나누지 않았다.

마지막의 `draft_recv_cmd`는 다음 요청의 명령을 기다리고 받아오는 구간이다.
그 뒤의 `draft_recv_request`는 다음 step의 실제 요청 수신이므로, 새 그림에서는 파란색으로 변경했다.
현재 target step 종료는 **25.673ms**(빨간 점선), 다음 target 요청 전송 시작은 **25.980ms**,
draft의 요청 본문 수신 시작은 **26.126ms**다. 회색이 현재 step을 넘는 이유에는 다음 step 통신이 포함된다.

**다음 요청 전에 준비가 끝난 여유만 추가 연산에 쓸 가능성이 있다.** 명령이 도착한 뒤에도
draft가 계산 중이면 다음 cache 응답을 늦춰 target 대기를 늘릴 수 있다. 이 예시에서 기존
명령 수신 시작24.957ms와 다음 요청 전송25.980ms 사이 차이는 약1.024ms다.
이 값도 시계 정렬·통신·정리 비용과 실행 변동을 고려하지 않은 관측 여유이지 보장된 예산은 아니다.

또한 현재부터 target 대기가 전혀 없는 것도 아니다. 다음 step79의 `target_recv_response_wait`는
**26.326–29.723ms, 약3.396ms**이며, 이미 준비한 cache를 찾아 응답하는 처리·통신도 포함한다.
이 대기 전부가 이전 PS가 늦었기 때문이라고 해석해서는 안 된다. 이전 PS는24.665ms에 끝났다.
즉 회색 전체1.461ms를 추가 계산에 써도 target 영향이 없다는 결론은 낼 수 없다.

## 2. 연두색 박스와 빈칸에서 실제로 하는 일

**현재 소스만 보고 추측하지 않고, 측정 당시 commit `82d964e0`의
`ssd/ssd/engine/draft_runner.py`를 확인했다.** 관련 원문은
[MEASURED_PS_SOURCE.txt](MEASURED_PS_SOURCE.txt)에 commit·행 번호와 함께 보관한다.

① `phase2_build`, `_tree_step_p1p2_dynamic` 2123–2162행:

- proxy 후보에서 PS root 선택, DS root와의 중복 처리.
- 고정 입력 폭에 맞춘 padding과 위치별 fan-out/layout 갱신.
- tree 실행 입력과 RoPE 위치 구성.
- glue 가시성 행을 GPU에서 CPU NumPy 배열로 읽는 작업.

따라서 이 박스는 **root와 실행 정보를 준비하는 실제 비용**이다.
proxy 확률 계산 자체는 target 쪽이고, 이 박스에서 draft Transformer를 실행하는 것은 아니다.

② `phase2_build` 종료와 첫 `p2_prepare` 사이:

- `_run_p2_tree_step` 2970–3008행에서 입력 metadata를 고정 주소 버퍼에 복사.
- `_compute_step_positions_and_slot_maps`에서 각 forward의 위치·KV slot·문맥 길이 계산.
- `_p2_step_state`에서 RoPE, KV slot, 문맥 길이, page 범위 등 실행 가능 여부 검사.
- 검사 결과를 `torch.cat(...).cpu().tolist()`로 읽어 CPU에서 실행/skip 결정(2911행).

이 구간의 CPU 경계 간격도 **1.177ms**로 확인됐다. 따라서 단순한 렌더링 여백이 아니다.
그러나 metadata 계산, copy, readback, Python 처리 각각의 시간을 따로 계측하지는 않았으므로
**1.177ms 전부가 CPU 연산, 전부가 readback, 또는 전부 제거 가능한 비용이라고 단정하지 않는다.**

고정 버퍼는 주소/수명 안전성을, 검사 코드는 잘못된 KV 접근 방지를 위한 계약을 지킨다.
최적화한다면 재사용·중복 계산 제거·graph 안 처리 가능성을 확인하되, 정확성 검사를 단순 삭제해서는 안 된다.

③ PS 종료 뒤에는 key 무효화와 다음 명령 수신으로 돌아가는 경로가 있다.
이 구간도 내부 label이 없으므로 그림에는 `Post-PS gap (unprofiled)`로 표시했다.
graph replay 2.052ms와 CPU 변환 span을 단순 더하지 않는다. CPU 변환 span에는 앞선
비동기 graph 완료를 기다리는 시간이 포함될 수 있기 때문이다.

## 3. 이전에도 알려져 있던 비용인가

**PS 전체에서 graph 밖 준비·정리 비용이 크다는 것은 이전에도 측정했다.**
[round8 component review §3](../../../../PSD-mlsys-coverage/results/mlsys_coverage/round8_component_review/REPORT.md)에
다음 값이 이미 있다.

| 당시 구현 | P2 전체 중앙값 | P2 graph 중앙값 |
|---|---:|---:|
| 논문 tree | 3.804ms | 1.677ms |
| 통합 실행기 | 2.402ms | 1.692ms |

당시에도 `phase2_build`가 tree replay 전에 끝나므로 PS 전체로 쓰면 안 된다고 명시했다.
따라서 준비 비용을 몰랐던 것은 아니다. **다만 이번의 빈 1.177ms를 이 코드 경계로
분리해 보여주고 해결했었다는 근거는 없으며, 준비 비용이 모두 제거된 상태도 아니다.**

과거 표는 7B+135M, RTX4090, FP16, T=1, K1/K2=4/2이고, 이번 그림은
7B+68M, RTX3090, BF16, T=0.7, K1/K2=6/3이다. 2.402ms와 6.872ms를 직접
비교하여 특정 수정 때문에 느려졌다고 결론 내릴 수 없다. 이번 값은 계측 실행의 elapsed이며
비계측 production에서 같은 양을 그대로 절약할 수 있다는 뜻도 아니다.

## 4. 한 step에만 나타난 현상인가

아래는 기존과 같은 유효 step 집합의 중앙값이다. 모두 실제 **4턴 profile**이며
full Spec-Bench 성능 결과는 아니다. 중앙값 열끼리 더하거나 빼면 행별 중앙값과 다를 수 있다.

| 모델·설정 | 유효 step | PS root/layout | 그 뒤 미계측 준비 | PS 전체 | PS graph | 다음 명령 수신 대기 |
|---|---:|---:|---:|---:|---:|---:|
| Llama2-7B+68M, 6/3 | 321 | 1.573 | 1.173 | 6.381 | 2.037 | 1.582 |
| Llama2-13B+135M, 6/3 | 444 | 2.035 | 1.537 | 9.313 | 3.623 | 1.432 |
| Llama3-8B+340M, 4/1 | 387 | 1.539 | 1.111 | 6.239 | 2.120 | 3.142 |
| CodeLlama-34B+135M-code, 12/6 | 73 | 1.567 | 1.208 | 11.866 | 7.359 | 7.695 |
| 같은 34B, 16/6 대안 | 84 | 2.054 | 1.614 | 13.507 | 7.408 | 1.032 |

7B의 연두색 구간 p95는 1.640ms, 뒤의 빈 구간 p95는 1.227ms다.
첨부 step 하나의 큰 이상치로 보이지 않는다. 다만 321개 step이 독립된 321개 문항인 것은 아니다.

이번 step에서 PS는 target 최종 head보다 0.419ms 늦지만, **다음 target 요청 전송보다
1.316ms 먼저 끝난다.** 즉 준비 비용은 상당하지만, 이 step에서는 준비 비용 전부가
다음 요청의 대기나 TPS 손실로 노출되는 것은 아니다. 비용을 줄이면 여유가 늘어나고 더 많은
draft 연산에 쓸 가능성이 있지만, AL이나 TPS 향상은 별도 대조 없이 주장하지 않는다.

## 5. 그림 수정과 검증

- 기존 원시 이벤트와 그림은 보존했다. 신규 PDF에서는 같은 대표 step과 원래 두 줄 renderer를 사용했다.
- 이름 없는 준비 구간을 사선으로 표시하고, 기존 raw에 있던 `draft_recv_cmd`와 다음 요청 수신을 추가했다.
- `draft_recv_cmd`는 이전 step ID가 붙는다. **step ID로 단순 join하지 않고**, 같은 요청 안의
  현재 merge 종료와 다음 요청 수신 사이 CPU 시각으로 연결했다. 요청 경계를 넘지 않는다.
- 1,309개 step 모두 연결 성공, 음수 구간 없음, PS 이후 세 구간 합과 전체 간격 일치,
  원본 SHA256 불변을 확인했다.
- 22개 CPU 검사 통과: [WAITS_CHECKS.log](WAITS_CHECKS.log).
- PDF/PNG 및 원시 구간 보존 검증: [WAITS_VALIDATION.json](WAITS_VALIDATION.json).
- 신규 도구: `ssd/tools/duet_timeline/plot_model_wait_timelines.py`.

```bash
/home/chokwans99/PSD/ssd/.venv/bin/python -B \
  ssd/tools/duet_timeline/plot_model_wait_timelines.py \
  --source /data/chokwans99/duet-ver10-eslab18-review-20261015/received/duet_ver10_parameter_search_20261015 \
  --output results/mlsys_ver11/model_breakdowns_20261015
```

원인별로 얼마를 줄일 수 있는지까지 확정하려면 위 metadata 계산·검사·readback 구간을
세분하여 동일 입력 A/B로 검증해야 한다. 이번 작업은 원시 기록과 코드를 대조한 진단이며,
추가 최적화 구현이나 성능 향상 검증을 완료한 것으로 기록하지 않는다.

## 6. 기존 최적화 시도와 지금 남은 B1 준비 비용

2026-10-15 추가 코드·이력 검토. 근거:
[PS_OPTIMIZATION_REVIEW.json](PS_OPTIMIZATION_REVIEW.json).
이번 검토에서는 런타임 수정이나 새 GPU 성능 실험을 하지 않았다.

### 이미 시도하고 반영한 것

| 최적화 | 확인된 내용 | 이번 B1 그림과의 관계 |
|---|---|---|
| P1/P2 CUDA graph·버퍼 재사용 | B1 초기화에서 P1/P2 준비. 이번 로그는 P1 replay426, P2 replay426이며 capture 통계 항목 없음 | 이미 적용됨. 매 step graph 생성이 현재 주황/빈 구간의 원인은 아님 |
| P1 root 고정 버퍼 재사용 | `e1ddefd`에 구현·검사 | P1에 적용된 최적화이며 PS 모든 준비 비용 제거를 뜻하지 않음 |
| 온도 조건의 GPU→CPU 검사 제거 | `3d51664`: CPU 요청 계약으로 P1/P2 온도 지원 여부 결정 | B1에도 적용. 이번 설정은 `stochastic_only=True` |
| Phase2 layout 재사용·boolean index 대체 | layout 재생성 대신 기존 필드 갱신, boolean take 대신 고정 크기 gather | B1 코드에도 반영돼 있지만 fan-out readback 등은 남아 있음 |
| NumPy 입력 준비 | `703b3a7`, `BatchedArenaInputs.replay`; 이전3090/B8 측정1.69–1.80→1.30–1.40ms/호출 | **B>1 실행기 경로의 근거**. 현재 B1의 두 주황색 구간이 이 최적화 후0.4ms 줄었다고 말할 수 없음 |
| 결합 kernel·bulk metadata | B>1 staging에서 작은 연산 결합·metadata 일괄 D2H. 기존 동일 입력 대조에서 P1/P2 감소 확인 | B1 기본 경로를 이 실행기로 교체한 것이 아님 |

수치 출처는 [ver0.1 보고서](../../mlsys_ver01/REPORT.md)의
`C. 직접 측정한 비용 개선`과 `각 최적화도 따로 제거해 확인` 표다.
이전 B8의 Llama2 bulk metadata 대조는 P1 23.585→22.477ms, P2 9.940→9.458ms다.
이는 NumPy 대조와 다른 설정·실험이므로 개선량을 합산하지 않는다.

### 왜 이번에는 여전히 큰가

1. **B1과 B>1의 준비 경로가 다르다.** `batch_tree_common.enabled`는 최대 B>1 또는
   명시한 내부 실험 override에서만 묶음 실행기를 사용한다. 이번 기록은 B1이고 override가 기록되지 않았다.
   현재 보고 있는 `DraftRunner._tree_step_p1p2_dynamic`/`_run_p2_tree_step`는 그와 별도 경로다.
2. **CUDA graph는 현재 주로 tree 생성 본체를 감싼다.** root·fan-out·KV 정보가 바뀔 때의
   Python 준비와 GPU→CPU 판정, graph 입력 갱신은 그 밖에서 매번 실행된다.
3. **작은 데이터도 동기화 비용이 든다.** fan-out `.tolist()`, 위치 scalar 변환,
   glue 행 `.cpu().numpy()`, 실행 상태 `.cpu().tolist()`가 남는다. 복사 byte 수가 작다는 이유로
   지연도 작다고 단정할 수 없다. 앞선 GPU 결과를 기다린 뒤 CPU가 후속 연산을 예약하는 과정이 포함된다.
4. **같은 정보를 여러 형태로 만드는 비용이 있다.** 고정 버퍼 복사, 위치/KV 계산,
   CPU NumPy 변환, executor 입력 복사 단계가 이어진다. 예를 들어 위치별 root mapping은
   layout 갱신에서 만든 뒤 tree-hit PS에서 `seed_ctx`로 다시 계산한다.

그림의 B1 준비 경로7개 함수는 측정 commit `82d964e0`와 현재 `0916871`의 AST가 동일했다.
즉 ver1.1으로 이름이 바뀌었다는 이유로 이 비용이 이미 해결됐다고 볼 근거는 없다.

### 줄일 수 있는 구체적인 후보와 검증 순서

| 우선 | 변경 후보 | 유지해야 하는 것 | 현재 판정 |
|---|---|---|---|
| 1 | B1에서 이미 만든 `layout.fan_idx_hit`을 `seed_ctx` 계산에 재사용. 곧 덮어쓸 위치 계산도 생성 단계부터 중복 제거 검토 | 같은 root 순서·RoPE·fan-out, 입력 버퍼 수명 | 동일 정보 중복 생성 확인. 실제 절감량은 아직 미측정 |
| 2 | GPU glue 가시성 행을 CPU로 내렸다 executor에 다시 올리는 왕복을 직접 GPU 버퍼 갱신으로 변경 | 유효 행과 padding 처리, ancestor mask 동일성, eager fallback | 구조적 개선 후보. 단순 `.cpu()` 삭제로 끝나지 않음 |
| 3 | KV 위치·slot·문맥 길이·page 갱신을 고정 버퍼에 직접 생성하고 작은 연산을 graph/결합 kernel로 묶기 | 비연속 page, 문맥 경계, 모든 root/width 조건 | B>1에 유사 접근의 실측 근거 있음. B1 이식 효과는 별도 확인 필요 |
| 4 | CPU에 이미 있는 정적 정보 재사용, 동적 유효성 검사를 한꺼번에 처리해 불필요한 readback 줄이기 | 잘못된 KV/RoPE 접근 차단·skip 계약 | 안전 검사를 삭제하는 최적화는 아님 |

고정 버퍼 복사는 이전 allocator/주소 수명 문제를 방지하기 위한 것이고, 상태 검사는
잘못된 KV slot/page 접근을 막는다. 이 목적을 유지하면서 계산/전달 방식을 바꿔야 한다.

검증은 같은 B1 입력·tree·root 순서·K1/K2를 고정하고 각 후보를 단독 대조하는 순서가 맞다.
root/layout, dispatch 준비, prepare A/B, graph, PS 전체를 따로 보고 출력·proposal·KV·RNG
동일성을 확인한다. Graph 밖4.820ms 전체를 제거 가능한 비용으로 잡거나, 다른 모델의
TPS 차이를 이 준비 최적화의 효과로 사용하지 않는다.

## 후속 직접 검증

2026-10-15 현재 ver1.1 코드에서 세 준비 경로를 수정하고 동일 GPU/모델/입력의 전후 breakdown을 측정했다. [구현·결과·회귀 증거](../ps_preparation_20261015/REPORT.md), [직접 구간 비교](../ps_preparation_20261015/BREAKDOWN_COMPARISON.pdf), [native timeline 전후](../ps_preparation_20261015/native/NATIVE_BEFORE_AFTER.pdf). 이 문서의 이전 3090/68M 원시 측정은 그대로 보존한다.
