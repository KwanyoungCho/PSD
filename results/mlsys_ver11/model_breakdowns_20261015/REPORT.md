# 모델 조합별 상세 latency breakdown — 2026-10-15

18번 서버가 최종 선택한 **4개 모델 조합, 5개 설정**의 원시 profile을 재분석했다.
용도는 target/draft 구간, DS/PS 시간 예산, cache 응답·준비·수락 비용을 직접 확인하는 것이다.
**새로운 inference나 파라미터 탐색은 수행하지 않았다.**

- **PS 준비 빈칸과 다음 요청 대기까지 표시한 최신본: [NATIVE_EVENTS_WITH_WAITS.pdf](NATIVE_EVENTS_WITH_WAITS.pdf) — 14쪽**
- 연두색 박스·빈칸·PS 이후 대기 설명: [PS_GAPS.md](PS_GAPS.md), [그림 모음](with_waits.html)
- 기존 최적화의 B1/B>1 적용 범위와 남은 개선 후보: [PS_GAPS.md §6](PS_GAPS.md#6-기존-최적화-시도와-지금-남은-b1-준비-비용)
- **준비/변환/merge까지 구분한 기존 상세 timeline: [NATIVE_EVENTS_MODELS.pdf](NATIVE_EVENTS_MODELS.pdf) — 14쪽**
- 상세 timeline 그림 모음: [native_events.html](native_events.html)
- 상세 timeline p25/p50/p75 전체: [NATIVE_EVENTS_ALL.pdf](NATIVE_EVENTS_ALL.pdf) — 41쪽
- Phase 요약: [ALIGNED_MODELS.pdf](ALIGNED_MODELS.pdf) — 모델당 한 페이지, 총5쪽
- 두 줄 timeline 모음: [aligned.html](aligned.html)
- 두 줄 p25/p50/p75 전체: [ALIGNED_REPRESENTATIVES.pdf](ALIGNED_REPRESENTATIVES.pdf) — 41쪽
- 이벤트별 세부 확인: [index.html](index.html), [ALL_MODELS_DETAILED.pdf](ALL_MODELS_DETAILED.pdf) — 52쪽
- 한 장 요약: [OVERVIEW.png](OVERVIEW.png)
- 전체 수치와 근거: [SUMMARY.json](SUMMARY.json)

### Forward 횟수 표시와 마지막 회색 구간 구분

`NATIVE_EVENTS_MODELS.pdf`/`NATIVE_EVENTS_ALL.pdf` 및 `NATIVE_EVENTS_WITH_WAITS.pdf`의
DS·PS graph 막대를 K1/K2 횟수만큼 흰 점선으로 등분하고 번호를 붙였다.
**개별 forward 경계는 수집하지 않았으므로 점선은 횟수 안내이며 실측 시간 분할이 아니다.**
전체 graph에는 forward 외에 선택·sampling 등도 포함된다.

대기 포함 그림에서는 주황색 준비를 A(root·가시성)와 B(graph 입력 버퍼)로 구분했다.
마지막 회색의 `draft_recv_cmd`와 다음 step의 `draft_recv_request`가 비슷해 보였으므로
다음 요청 수신을 파란색으로 변경했다. 현재 target step 종료는 빨간 점선으로 표시한다.
다음 요청 수신은 추가 draft에 쓸 수 있는 idle 시간이 아니다.

`244_t0`/step78은 target 현재 step 종료25.673ms, 다음 요청 전송25.980ms,
draft 다음 요청 수신26.126ms이다. 다음 target 응답 대기는26.326–29.723ms에 존재한다.
따라서 마지막 회색 전체가 target 지연 없이 사용할 수 있는 예산이라는 해석은 잘못이다.
상세 의미는 [PS_GAPS.md](PS_GAPS.md)를 따른다.

변경 전 중앙 PDF·audit는 `before_forward_dividers/`에 보존했다.
검증: [FORWARD_MARKERS_VALIDATION.json](FORWARD_MARKERS_VALIDATION.json).

### 추가 수정: 기존 최신 renderer로 이벤트별 두 줄 상세 그림

사용자가 DS/PS 내부 준비 과정도 구분하기를 요청하여, 이번에는 **기존
`plot_duet_aligned_timeline.py:plot_aligned_step`을 직접 호출**했다.
저장소의 관련 전체 branch 이력과 기존 PSD/coverage 작업 트리에서 확인한 마지막
갱신은 `6073966`(2026-09-08)이다. `plot_representatives.py`가 이 renderer를 부르고,
`plot_p1_p2_overview.py`와 `plot_paper_fig4_schematic_pct.py`는 별도의 요약/논문 도식이다.
앞서 만든 두 줄 phase 요약이 준비 과정을 충분히 나누지 못했으므로, 위 14쪽 상세본을 우선 본다.

상세본은 다음을 합치지 않고 원래 이름과 interval로 표시한다.

- DS: `p1_root_build`, `p1_slot_prepare`, `p1_prepare`, `p1_graph_replay`.
- PS: `phase2_build`, `p2_prepare`, `p2_graph_replay`, `p2_output_convert`, `merge_cache`, `p2_cache_merge`.
- 응답/glue: KV restore, tree 선택/pack, parent-q 전송, glue, tree 준비/replay.
- Target: 통신/검증 준비, exit 전후 graph, proxy, LM head, accept walk, KV commit.

원래 palette에 없던 P1 executor/통신/KV label에는 서로 다른 색·무늬만 추가했다.
구간을 phase 합계로 치환하지 않았고, parent span은 먼저 그려 그 안의 세부 구간이 보이게 했다.
현재 cache wait와 겹친 직전 draft step도 포함한다. 대표 step 및 원시 시각은 앞선 자료와 동일하다.

기존 renderer에는 **0.02ms 미만 이벤트의 최소 표시 폭**이 있다. 그대로 사용하되 그림
제목에 명시했고, [NATIVE_EVENT_INTERVALS.csv](NATIVE_EVENT_INTERVALS.csv)에 실제 시작·종료·길이를 저장했다.
겹친 부모/자식 막대는 합산하지 않는다. 공통 helper인 `p2_prepare`/`p2_output_convert`는
DS에서도 쓰일 수 있으므로 이름만으로 phase를 결정하지 않는다.

이 profile은 여러 forward를 묶은 **phase 전체 CUDA graph replay**를 기록한다.
따라서 준비/실행/변환은 나눌 수 있지만, graph 안의 forward1/2/3 실측 시간은 알 수 없다.
추가한 등분 점선은 횟수만 표시한다. 실제 경계를 확인하려면 별도 계측 실행이 필요하다.
이번 작업은 기존 raw의 재시각화다.

재현 adapter: `ssd/tools/duet_timeline/plot_model_native_events.py`.
위 source/output을 그대로 넘겨 실행한다. 입력/도구 해시와41개 대표 선택 근거는
[NATIVE_EVENTS_AUDIT.json](NATIVE_EVENTS_AUDIT.json)에 있다.

검증은 [NATIVE_VALIDATION.json](NATIVE_VALIDATION.json), [NATIVE_CHECKS.log](NATIVE_CHECKS.log)를 따른다.
19개 CPU 검사 통과,41개 대표 선택 동일, 표시한2,059개 원시 이벤트의 interval 일치,
현재 step 이벤트 전부 보존, 겹치는 직전 step 보존, 입력 SHA256 불변을 확인했다.
중앙 대표 PDF14쪽과 전체 PDF41쪽 및 PNG14장의 읽기 검사도 통과했다.

### 가독성 수정: 기존 aligned timeline 형식

사용자가 지정한 `ssd/bench/plot_duet_aligned_timeline.py`의 두 줄 Gantt 표현을
기준으로 다시 만들었다. **위 target, 아래 draft**, proxy 별도 stream만 작은 보조 줄이다.
한 모델 페이지에 P1 hit/P2 hit/miss를 세로로 배치하고, 세 상태의 x축 범위를 동일하게 맞췄다.

- DS/PS 전체 막대: 준비부터 cache 정리까지의 phase elapsed.
- DS/PS 아래 진한 띠: 그 안의 graph replay. graph 밖 준비/변환/merge 시간이 자연스럽게 보인다.
- 세로 점선: exit layer와 final LM head 완료. PS가 점선을 넘는지 바로 확인한다.
- 현재 요청 시점에 남은 직전 PS는 회색 가는 막대로 표시한다.
- 내부 helper 이름과 CPU dispatch는 세부 PDF에 남기고, 이 그림에서는 phase 중심으로 묶었다.
- 모든 막대는 **실제 한 step**이다. phase별 중앙값을 이어 붙인 가상 timeline이 아니다.
  대표 step 선정은 앞선 상세 자료와 동일하다. 따라서 대표 그림의 각 구간 값이 아래 전체 step
  중앙값 표와 정확히 같을 필요는 없다.
- 원본 시간, 데이터, 설정, 시계 정렬은 바꾸지 않았다. 신규 inference도 수행하지 않았다.
- Code34B 12/6은 miss 그림이 없고, 16/6은 miss 표본2개 중 하나가 대표이므로 일반화하지 않는다.

사용한 renderer는 ver1.1 작업 트리의 `plot_duet_aligned_timeline.py:draw_phase_lanes`이며,
기존 renderer/CLI 동작은 그대로다. Adapter는 `ssd/tools/duet_timeline/plot_model_aligned_timelines.py`다.
`ALIGNED_INTERVALS.csv`에서 표시한 구간을, `ALIGNED_AUDIT.json`에서 선택한 UID·step과
입력 및 도구 해시를 확인한다. 같은 source/output 인자로 adapter를 실행하면 재생성된다.

재검증: 기존 aligned 도구5개, 요청 정렬4개, 상세/두 줄 표현8개로 **17개 검사 통과**.
1,309개 step 모두에서 두 줄 그림의 DS/PS 길이가 기존 분석값과 일치하며,
41개 대표 step 선택과 원본 SHA256도 동일하다. [ALIGNED_VALIDATION.json](ALIGNED_VALIDATION.json),
[검사 로그](ALIGNED_CHECKS.log)를 보관했다. 검사는 `ssd/tests`에서 다음처럼 실행한다.

```bash
PYTHONPATH=.. /home/chokwans99/PSD/ssd/.venv/bin/python -B -m unittest -v \
  test_plot_duet_aligned_timeline test_model_breakdowns test_ver10_profile_analysis
```

## 1. 어떤 측정인가

현재 분석 작업 트리는 `MLSys-ver1.1`이다. 그러나 실제 측정은 eslab18의 **ver1.0 및
모델/자원 지원 commit**에서 수행됐다. 이전 런타임의 설정을 새로운 ver1.1 설정으로
바꾸어 표기하지 않았다.

- RTX3090, BF16, 실제 B=1, T=0.7, worker seed42.
- 설정별 profile 입력 **4턴**, 입력 상한512, 출력 상한256, 문맥 상한2048.
- 일반 모델은 원본 calibration 입력, CodeLlama는 원본 processed HumanEval code 입력.
- 대상은 인수인계 `selected.json`의 최종 선택이다. 전역 최적값 또는 full Spec-Bench 결과가 아니다.
- CodeLlama-34B는 시간 예산을 우선한 12/6 설정과, 별도 입력에서 처리량이 더 높았던
  16/6 대안이 함께 보존돼 있어 둘 다 그렸다.
- profile을 켠 4턴의 latency에, profile을 끈 별도 held-out 평가의 TPS/AL을 붙여서
  같은 모집단처럼 표시하지 않았다. 이 문서에서는 component latency를 중심으로 본다.
- target을 여러 GPU로 나눈 경우에도 기록은 **target rank0만** 존재한다. 다른 target GPU의
  시간축은 추정해 만들지 않았다. 표의 GPU 수는 `target + 별도 draft`다.

## 2. 모델별 설정과 PDF

Exit 숫자는 실제 config의 **0-based layer index**다. DS=P1, PS=P2다.

| 모델 조합 | Exit | K1/K2 | DS 위치당 root | G=M, DS/PS | 확장 경쟁 | GPU 수 | PDF |
|---|---:|---:|---:|---:|---|---|---|
| LayerSkip Llama2-7B + llama-68M | 20 | 6/3 | 2 | 12/8 | frontier | 1+1 | [상세](l2_7b/DETAILED.pdf) |
| LayerSkip Llama2-13B + AMD-Llama-135M | 23 | 6/3 | 3 | 12/8 | last | 2+1 | [상세](l2_13b/DETAILED.pdf) |
| LayerSkip Llama3-8B + Axcel Llama3.1 drafter-340M | 20 | 4/1 | 3 | 8/4 | last | 1+1 | [상세](l3_8b/DETAILED.pdf) |
| LayerSkip CodeLlama-34B + AMD-Llama-135M-code | 30 | 12/6 | 3 | 12/8 | last | 4+1 | [상세](code34b/DETAILED.pdf) |
| 같은 CodeLlama 조합, 처리량 대안 | 30 | 16/6 | 3 | 16/8 | last | 4+1 | [상세](code34b_throughput/DETAILED.pdf) |

실측 당시 공통 설정:

- 후보식 `complement` = e(1−q), `topm` 정규화, 위치 overlap 혼합0.25.
- `SSD_TREE_EXPANSION_POLICY=q_path`, sibling C=3, DS allocation=dynamic.
- DS forward scale=1.5, PS 입력 width=15, PS 활성 root 상한=10, threshold=0.
- proxy 위치별 top-L의 **resolved 값**은28. 단, 34B 16/6 대안은34다.
  요청값28만 보고 대안도28로 표시하면 틀린다.
- 실제 DS executor 입력 width는 source/shape에 따라 다르다. 아래 표 및 각 모델의
  `PARAMETERS.json → physical_tree_shapes`에 startup에서 기록된 값이 있다.

| 설정 | 기록된 DS round width 패턴 | PS round width |
|---|---|---|
| 7B | 8×6, 18×6, 첫26→이후21×5 | 15×3 |
| 13B | 12×6, 27×6, 첫39→이후32×5 | 15×3 |
| Llama3 8B | 6×4, 15×4, 첫27→이후23×3 | 15×1 |
| Code34B 12/6 | 21×12, 39×12 | 15×6 |
| Code34B 16/6 | 21×16, 51×16 | 15×6 |

이 표는 startup에서 기록한 **실행 shape 목록**이며 매 step의 실제 활성 노드 수가 아니다.
모든 설정은 G=M이지만 후보 수·source에 따라 forward의 입력 개수가 달랐다.
ver1.1의 `ds_width/ps_width`, 자동 top-L 설정으로 새로 실행한 결과는 아니다.

## 3. 그래프 읽는 순서

각 모델의 PDF는 다음 순서다.

1. **Components and deadlines**: 구간별 p50/p95, DS/PS 시간 여유의 누적분포,
   전체 step의 시간 추이. 긴 step을 임의로 제거하지 않았다.
2. **Cache source breakdown**: 현재 P1 hit/P2 hit/miss별 target/draft 구간,
   DS/PS graph와 graph 밖 구간, context 길이에 따른 시간 변화.
3. **실제 step의 상세 timeline**: 각 cache 상태의 p25/p50/p75 대표. target과 draft는
   같은 시간축을 쓰며 각 label을 별도 행에 보여준다. 대표마다 UID, 요청 번호,
   step 번호, proposal 노드 수, 해당 step의 AL을 표시한다.

Timeline의 기준시각0은 **현재 target의 exit 이전 forward 시작**이다.
그 이전 cache 응답/검증 준비는 음수 시간에 나타난다.

- 색 막대: CUDA event 사이의 elapsed. 순수 kernel 실행 시간만을 뜻하지 않는다.
- 검은 가는 선: CPU에서 해당 구간을 호출하고 반환하기까지의 dispatch 시간.
  동기화 대기를 포함할 수 있으므로 순수 CPU 연산 시간으로 해석하지 않는다.
- 세로선: target 시작, exit layer 완료, final LM head 완료, 다음 request 송신.
- 현재 cache wait와 겹친 **직전 step의 PS tail**이 있으면 별도 행으로 남겼다.
- `verify_sample_accept`와 내부 `tree_accept_walk`, `merge_cache`와 내부
  `p2_cache_merge`처럼 부모/자식 구간은 **서로 더하면 안 된다**.
- proxy side stream은 target after-exit과 겹친다. 이 둘 역시 순차 합계로 해석하지 않는다.
- 짧은 구간을 잘 보이게 만들려고 길이를 부풀리지 않았다. 세부 수치는 CSV/PDF 확대를 이용한다.

현재 P1/P2 hit는 **이전 step에서 만든 어느 cache를 지금 사용했는가**다.
같은 그림의 DS/PS는 **다음 verification outcome을 대비한 생성**이다.
따라서 현재 P1 hit라고 해서 현재 PS가 필요 없었다는 뜻이 아니다.

## 4. 관측된 시간

아래는 모든 유효 step을 합친 **각 구간의 중앙값(ms)**이다. 독립적인 중앙값이므로
표의 열을 더해서 총 latency를 만들지 않는다. 요청/토큰 수로 가중한 TPS 표도 아니다.

| 설정 | Target exit 전 | Target exit 후 | Final head | 수락 처리 | DS 전체 | PS 전체 | 다음 target 시작까지 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 7B 6/3 | 11.564 | 6.048 | 0.366 | 1.092 | 6.954 | 6.381 | 25.330 |
| 13B 6/3 | 15.651 | 9.012 | 0.374 | 1.069 | 12.008 | 9.313 | 33.291 |
| Llama3 8B 4/1 | 12.611 | 6.609 | 1.213 | 1.137 | 10.155 | 6.239 | 28.300 |
| Code34B 12/6 | 38.759 | 17.489 | 0.474 | 1.589 | 22.646 | 11.866 | 65.142 |
| Code34B 16/6 | 33.827 | 17.199 | 0.399 | 1.352 | 35.635 | 13.507 | 62.391 |

DS 전체는 root 준비부터 proxy 대기 직전까지다. PS 전체는 proxy 수신 완료부터
cache merge 완료까지다. 즉 draft transformer graph만 재서 budget이 충분하다고
판정하지 않았다.

| 설정 | DS 시간 여유 p50 | PS vs final head p50 | PS vs 다음 request p50 |
|---|---:|---:|---:|
| 7B 6/3 | +4.126 | +0.026 | +1.754 |
| 13B 6/3 | +2.229 | +0.020 | +1.634 |
| Llama3 8B 4/1 | +2.679 | +1.594 | +3.310 |
| Code34B 12/6 | +14.963 | +6.084 | +7.860 |
| Code34B 16/6 | −4.197 | −0.247 | +1.256 |

**양수는 먼저 끝남, 음수는 늦음**이다. 단, DS와 PS의 기준시각이 다르다.

- DS 여유 = 추정 proxy 도착 시각 − DS 종료. 실제 `proxy_wait` 종료만 사용하면,
  DS가 이미 늦어 proxy가 대기하던 상황을 숨기므로 send 시각과 transport 추정으로 보정한다.
- PS vs final head = final LM head 완료 − cache merge 완료.
- PS vs 다음 request = 다음 target request 송신 − cache merge 완료.
- 서로 다른 프로세스의 GPU anchor에 시계 오차가 있으므로 0.02ms 수준을 확정적인
  deadline 충족으로 해석하지 않는다. 그림의 ±1ms 음영은 경계 근처를 단정하지 않기 위한
  분류 구간이며, 계측오차를 통계적으로 추정한 신뢰구간이 아니다.

관찰할 점:

- **7B·13B**: DS에는 여유가 있지만 PS는 final head 종료에 거의 맞닿는다.
  다음 request까지의 여유와 final head까지의 여유를 구분해야 한다.
- **Llama3 8B**: K2=1이어도 PS 전체6.239ms 중 graph 중앙값은2.120ms다.
  graph 밖 elapsed 중앙값4.117ms가 있으므로 forward 횟수만으로 PS 시간을 설명할 수 없다.
  graph 밖 시간에는 준비·출력 변환·merge·대기가 포함되며, 이를 전부 CPU 계산으로 부르지 않는다.
- **Code34B 12/6**: 이 profile에서는 DS/PS 모두 여유가 있다.
- **Code34B 16/6**: DS graph가 커지고 glue/응답 비용도 달라진다.
  PS가 final head보다1ms 이상 늦은 step은46.43%, 다음 request보다1ms 이상 늦은 step은38.10%다.
  DS 여유 추정은 transport 표본이 **1개**라 특히 주의한다. DS 초과100%는 그 추정 아래의 값이다.
- 34B 두 설정은 K1뿐 아니라 DS M과 proxy resolved top-L도 다르고,
  출력·hit source·검증 shape 분포도 다르다. target exit 전 시간 차이38.759→33.827ms를
  **K1 변경이 target kernel을 가속했다는 인과 증거로 해석하면 안 된다**.
- Llama3 선택안은 별도 소규모 held-out에서 AR보다 빨랐던 설정도 아니다.
  여기서는 선택한 tree 연구 설정의 병목을 보여주는 것으로, 모든 모델에서 tree의 우위를 주장하지 않는다.

## 5. 표본 수와 검증

| 설정 | P1 hit | P2 hit | Miss | 유효 step | 대표 timeline |
|---|---:|---:|---:|---:|---:|
| 7B | 115 | 164 | 42 | 321 | 9 |
| 13B | 139 | 225 | 80 | 444 | 9 |
| Llama3 8B | 65 | 295 | 27 | 387 | 9 |
| Code34B 12/6 | 62 | 11 | 0 | 73 | 6 |
| Code34B 16/6 | 63 | 19 | 2 | 84 | 8 |
| 합계 | 444 | 714 | 151 | **1,309** | **41** |

- warmup 요청 전체, 각 측정 요청의 첫2단계와 마지막 단계만 제외했다.
- 위 counts는 profile의 이 필터를 적용한 모집단이다. 별도 성능 평가의 cap/clip 제외 규칙,
  full dataset의 cache hit율과 혼동하지 않는다.
- Code34B 12/6의 miss 그림은 표본이 없어 만들지 않았다. 16/6은 miss2개를
  중복하지 않고 둘 다 보여준다. 2개로 안정적인 분위수를 추정할 수는 없다.
- 원시 이벤트→현재 분석기의 **1,309개 단계별 수치 및 audit이 기존 timing.json과 전부 동일**하다.
- full request anchor로 요청을 구분했다. sparse tree label, event 기록 순서에 의존하지 않는다.
- outer `target_spec_wait`가 내부 request marker보다 먼저 시작하는 경우,
  특히 warmup이1step이고 다음 요청도 step1인 경우를 별도 검사했다.
- 같은 step에서 두 번 나오는 공유 helper를 버리지 않고 보존했다. 겹치는 interval은 union으로
  계산하고 parent/child latency를 이중 합산하지 않았다.
- 최신 정렬 분석기4개 + 상세 도구6개, **총10개 CPU 회귀 검사 통과**.
- 입력 SHA256은 처리 전후 동일하다. 엔진 runtime과 원시 실험 자료는 변경하지 않았다.

## 6. 도구 선택과 재현

원본 인수인계:

```text
/data/chokwans99/duet-ver10-eslab18-review-20261015/received/duet_ver10_parameter_search_20261015
```

원본 runtime: 7B=`82d964e0`, 13B/34B=`59f46511`, Llama3=`16dd4198`.
각 실행의 전체 commit/hash는 `SUMMARY.json` 및 모델별 `AUDIT.json`을 따른다.

기존 도구는 `ssd/tools/duet_timeline/`에 있었다. 논문용 비율 도식이나 과거 timeline의
자동 시계 이동을 그대로 이용하지 않고, **수정된 `request-anchor-v2` 정렬 분석기**에
상세 plotting wrapper를 연결했다.

- 도구: [plot_model_breakdowns.py](../../../ssd/tools/duet_timeline/plot_model_breakdowns.py)
- 정렬 분석기: [duet_profile_analysis.py](../../../ssd/bench/duet_profile_analysis.py)
- 사용법: [README](../../../ssd/tools/duet_timeline/README.md)

저장소 루트에서:

```bash
/home/chokwans99/PSD/ssd/.venv/bin/python -B \
  ssd/tools/duet_timeline/plot_model_breakdowns.py \
  --source /data/chokwans99/duet-ver10-eslab18-review-20261015/received/duet_ver10_parameter_search_20261015 \
  --output results/mlsys_ver11/model_breakdowns_20261015

/home/chokwans99/PSD/ssd/.venv/bin/python -B -m unittest -v \
  ssd.tests.test_model_breakdowns ssd.tests.test_ver10_profile_analysis
```

각 모델 디렉터리의 파일:

| 파일 | 내용 |
|---|---|
| `DETAILED.pdf` | 모델별 모든 그림 |
| `01_components_and_deadlines.png` | 구간별 분포와 시간 여유 |
| `02_cache_source_breakdown.png` | P1/P2/miss별 분해 |
| `timeline_*.png` | 실제 대표 step의 상세 시간축 |
| `PARAMETERS.json` | 요청 config, kwargs, resolved config, 실제 executor shape 목록 |
| `steps.csv` | 1행=1개 유효 step의 수치와 UID |
| `component_statistics.csv` | 전체/상태별 mean 및 분위수 |
| `all_label_statistics.csv` | label별 GPU/CPU dispatch 통계와 존재 표본 수 |
| `event_intervals.csv` | 모든 유효 step의 실제 이벤트 시작/종료, parent 관계 |
| `representatives.csv` | 대표 그림에 선택된 실제 UID/step 및 선택 기준 |
| `AUDIT.json` | 표본 수, transport 추정, 원본 해시, parser 검증 |

`all_label_statistics.csv`는 해당 label이 **있는 step**만 평균한다. label 부재를0ms로
합쳐 평균을 낮추지 않는다. 상단 component에서 조건부 helper 부재는0으로 표현하므로,
두 표를 비교할 때 `eligible/present` 분모를 함께 확인한다.
