# DUET timeline 도구

Target의 앞/뒤 모델 실행과 early-exit proxy 계산, draft의 P1/P2 실행을 같은
시간축에서 확인하는 도구다. Chain과 tree 모두 같은 명령을 사용한다.

## ver1.2: 상세 이벤트 + 연속 두 step + draft deadline

최신 tree 확인용 도구는 `plot_consecutive_steps.py`다. 같은 요청의 연속 두 step을
target/draft 두 줄의 **하나의 시간축**에 그린다. 준비·graph·병합·명령 수신을
기존 색/무늬로 구분하고, 세 번째 step의 요청 경계로 두 번째 deadline까지 계산한다.
이 도구는 `phase2_build`, `merge_cache`, handshake가 있는 DS/PS tree trace용이다.
PS가 없는 다른 모드는 위의 기존 single-step 도구를 사용한다.

이미 만든 그림은 저장소 루트의
`results/mlsys_ver12/timelines/historical_models/CONSECUTIVE_STEPS.pdf`(14쪽),
`results/mlsys_ver12/timelines/accepted_runtime/CONSECUTIVE_STEPS.pdf`에서 확인한다.
전자는 **과거 모델별 ver1.0 trace**, 후자는 **1.2에 채택한 런타임과 같은 소스의 trace**다.
두 파일의 절대 시간을 같은 하드웨어/모델 조건의 비교로 해석하면 안 된다.

다른 서버에서도 원시 window JSON만으로 다시 그릴 수 있다. GPU/모델/원본 서버 경로가 필요 없다.
저장소 루트에서 Python에 numpy, matplotlib, pypdf를 설치한 뒤 실행한다.

```bash
python ssd/tools/duet_timeline/plot_consecutive_steps.py \
  --window results/mlsys_ver12/timelines/accepted_runtime/req26_step7.json \
  --output /tmp/duet-two-steps
```

새 profile에서 선택하려면 다음과 같이 실행한다. 요청 epoch는 파일 전체를 시간순으로
정렬한 뒤 step ID가 다시 시작될 때 증가하는 **0-based 요청 순번**이다. 요청 끝의
두 step은 이후 deadline이 없으므로 선택할 수 없다. 다른 요청에 연결하지 않고 오류로 거부한다.
실행한 설정과 같은 K1/K2를 반드시 입력한다.

```bash
python ssd/tools/duet_timeline/plot_consecutive_steps.py \
  --profile-dir /path/to/profiles --request-epoch 2 --step-id 78 \
  --k1 6 --k2 3 --title 'Model / measured version / parameters' \
  --output /tmp/duet-two-steps-new
```

모델별 인수인계에서 한 번에 그리려면 `--source /path/to/duet_ver10_parameter_search_20261015`
와 `--output /path/to/output`을 사용한다. 같은 cache source에서 후속 두 anchor가 있는
step의 target 전체 길이 중앙값을 선택한다. 기존 single-step 대표와 선택 step이 다를 수 있다.

- **빨간 점선 D1/D2:** 다음 target `target_send_request` 시작. 이전 PS가 완료되고
  draft가 다음 명령을 받을 준비가 되어 있어야 하는 **관측된 시점 기준**이다.
- **초록 원:** `merge_cache` 종료, 즉 cache 준비 완료. **삼각형:** 후처리까지 끝나고
  다음 `draft_recv_cmd`가 시작한 시점. 이 두 시점은 같지 않다.
- **Ready margin = 다음 target send 시작 − draft receive-ready.** 양수는 준비가 먼저 끝남,
  음수는 늦음이다. 실제 더 넣을 수 있는 forward 예산이나 인과적 stall 시간을 뜻하지 않는다.
  서로 다른 GPU의 CUDA→host 시간 정렬 오차도 포함한다.
- **Next response wait:** 다음 step의 통신·cache 조회·miss draft 등도 포함하므로
  이전 PS 지연으로 모두 귀속하지 않는다. 표의 두 번째 next wait는 그림 이후 step의 값이다.
- PS가 시작되려면 proxy 수신도 필요하다. `proxy_wait`는 그대로 표시한다. Target의
  proxy send enqueue 완료를 실제 draft 도착 시각 또는 DS의 고정 deadline으로 간주하지 않는다.
- 흰 점선은 K1/K2 **횟수** 안내용 등분이며, 개별 forward를 계측한 경계가 아니다.
  사선 빈 구간은 미계측 elapsed이며 GPU idle의 증거가 아니다. 부모/자식 구간을 합산하지 않는다.
- 새 두-step 그림에는 최소 0.02ms 표시 폭을 쓰지 않는다. 원래 이벤트 폭/시각을 유지하며
  모든 raw window, source SHA256, `DEADLINES.csv`, `AUDIT.json`을 같이 저장한다.

출력에는 PDF, PNG, 브라우저용 `index.html`, 재현 가능한 JSON이 포함된다.
`--window` 재생은 보관된 window의 출처 해시를 유지하지만 원본 서버 파일을 다시 검증하지는 않는다.

## 2026-10-15: 준비 과정을 합치지 않는 기존 상세 timeline (최신 확인용)

준비/실행/출력변환/merge를 따로 보려면 이 경로를 쓴다.
이번 작업 전 저장소 이력에서 기존 도구의 마지막 갱신은 `6073966`(2026-09-08)이며,
`plot_representatives.py`가 `bench/plot_duet_aligned_timeline.py:plot_aligned_step`을
호출하는 방식이다. 아래 adapter도 **동일한 기존 renderer를 직접 사용**한다.

```bash
python ssd/tools/duet_timeline/plot_model_native_events.py \
  --source /path/to/duet_ver10_parameter_search_20261015 \
  --output results/mlsys_ver11/model_breakdowns_20261015
```

`NATIVE_EVENTS_MODELS.pdf`(14쪽), 모델별 `NATIVE_EVENTS.pdf`, `native_events.html`을 본다.
`NATIVE_EVENTS_ALL.pdf`는 같은 실제 p25/p50/p75 대표41개다. 원래 이벤트를 합치지 않고
두 줄에 색·무늬로 구분하며, 기존 palette에 없는 P1 준비·KV·통신 label만 색을 보완했다.
요청 정렬과 표본 선택은 `request-anchor-v2`로 검증한 결과를 유지한다.

- `p1_root_build`, `p1_slot_prepare`, `p1_prepare`, `p1_graph_replay`를 구분한다.
- `phase2_build`, `p2_prepare`, `p2_graph_replay`, `p2_output_convert`, `merge_cache`를 구분한다.
- 같은 helper(`p2_prepare`, `p2_output_convert`)가 DS에서도 실행될 수 있다. 이름만 보고
  항상 PS라고 해석하지 않는다.
- 기존 renderer의 0.02ms 최소 표시 폭은 유지하고 제목에 명시한다. 정확한 시간은 CSV를 따른다.
- 프로파일이 phase 전체 graph replay만 기록했다면 그 안의 forward별 시간을 만들어낼 수 없다.
- 부모·자식 막대는 겹쳐 표시하므로 더하지 않는다. 빈 구간도 임의로 phase에 합치지 않는다.

## 2026-10-15: 모델별 두 줄 phase 요약

기존 `ssd/bench/plot_duet_aligned_timeline.py`의 target/draft 두 줄 Gantt 구성을
기준으로 한 모델당 한 페이지를 만든다. Proxy 별도 stream만 작은 보조 줄로 분리한다.
각 페이지는 P1 hit/P2 hit/miss의 실제 중앙값 대표 step을 같은 시간축으로 비교한다.

```bash
python ssd/tools/duet_timeline/plot_model_aligned_timelines.py \
  --source /path/to/duet_ver10_parameter_search_20261015 \
  --output results/mlsys_ver11/model_breakdowns_20261015
```

`ALIGNED_MODELS.pdf`(5쪽), 모델별 `ALIGNED.png`, `aligned.html`을 먼저 본다.
`ALIGNED_REPRESENTATIVES.pdf`는 p25/p50/p75의 모든 실제 대표 step(41쪽)이다.
DS/PS 전체 막대는 준비/실행/정리를 포함하고, 아래 진한 띠는 graph replay 구간이다.
원시 시계와 단계 길이를 보존하며, 짧은 막대의 폭을 키우지 않는다. CPU 이벤트별 자세한
내용은 아래의 상세 진단을 이용한다. 단순화한 timeline도 아래와 같은 엄격한 정렬을 쓴다.

## 2026-10-15: 모델 조합별 상세 진단 (세부 확인)

18번 서버의 `duet_ver10_parameter_search_20261015` 인수인계 형식은 다음 명령으로
분석한다. `selected.json`의 최종 선택 4개와 CodeLlama 처리량 대안을 읽는다.

```bash
python ssd/tools/duet_timeline/plot_model_breakdowns.py \
  --source /path/to/duet_ver10_parameter_search_20261015 \
  --output results/mlsys_ver11/model_breakdowns_20261015
```

위 명령은 저장소 루트 기준이다. 출력 `index.html` 또는
`ALL_MODELS_DETAILED.pdf`에서 모델별 component 분포, DS/PS 시간 여유,
P1 hit/P2 hit/miss별 실제 p25/p50/p75 step을 확인한다. 파라미터·실행 commit·
모든 이벤트의 CPU/GPU 시간·입력 SHA256도 함께 저장한다. 원본은 변경하지 않는다.

이 도구는 `ssd/bench/duet_profile_analysis.py`의 `request-anchor-v2`를 사용한다.
요청마다 다시 시작하는 step 번호를 전체 request marker로 구분하며, tree 경로에서
간헐적으로만 나타나는 label로 요청 경계를 추측하지 않는다. 저장된 `timing.json`과
재계산한 단계별 수치 및 audit이 모두 같아야 그래프를 만든다.

- warmup, 요청별 첫 2개 및 마지막 step만 제외한다. 긴 step은 제거하지 않는다.
- DS는 root 준비를 포함하고, PS는 proxy 수신 완료부터 cache merge 완료까지다.
- target 종료는 마지막 LM head를 포함한다.
- CUDA-event elapsed와 CPU dispatch를 구분한다. 어느 쪽도 GPU busy time이나
  순수 CPU 계산 시간으로 해석하지 않는다. 부모/자식 및 서로 겹치는 구간을 합산하지 않는다.
- 서로 다른 프로세스의 시계를 인위적으로 이동하거나 짧은 막대를 늘리지 않는다.
- 대표 step은 현재 target request→postprocess 시간의 분위수로 선정한다.
  표본이 없는 cache 상태는 생성하지 않고, 작은 표본의 중복 대표도 제거한다.
- 실제 측정은 ver1.0 및 모델 지원 commit 기준이다. ver1.1에서 그림을 만들었다고
  ver1.1 성능으로 표시하지 않는다. profile 4턴과 별도 held-out 평가를 섞지 않는다.

기존 `plot.sh`와 `plot_representatives.py`는 아래의 개별 profile 탐색용 도구다.
과거 도구의 자동 시간축 보정이나 논문용 비율 도식 대신, **새 모델별 진단에서는 위의
원시 시간축과 엄격한 요청 정렬을 사용하는 도구를 우선한다.**

## 프로파일 수집

실험 실행 환경에 다음 값을 추가한다.

```bash
SSD_PROFILE_DUET=1
SSD_PROFILE_DUET_DETAIL=1
SSD_PROFILE_DIR=/absolute/path/to/profile_dir
```

Proxy CUDA graph는 기본으로 켜져 있다. 비교 진단이 필요할 때만
`SSD_CHAIN_PROXY_GRAPH=0` 또는 `SSD_TREE_PROXY_GRAPH=0`으로 끈다.

새 profile에는 다음 구간이 별도로 기록된다.

- `exit_logits`: early-exit hidden state에서 logits를 만드는 시간
- `exit_proxy_launch`: 기본 GPU 작업줄에서 proxy 작업을 예약하는 시간
- `exit_proxy_side`: 별도 GPU 작업줄에서 실제 logits/proxy/send를 수행한 시간
- `chain_proxy_graph_replay`: chain K1/K2 및 miss의 후보 계산 graph
- `tree_proxy_graph_replay`: tree hit의 후보 계산 graph
- `proxy_send_enqueue`: 계산된 proxy 정보를 비동기 통신에 넣는 시간

따라서 긴 막대가 실제 GPU 계산인지, CPU 예약 지연인지, 통신 enqueue인지
timeline에서 분리해서 볼 수 있다.

## 그래프 생성

`ssd` 디렉터리에서 다음처럼 실행한다.

```bash
bash tools/duet_timeline/plot.sh \
  experiments/proxy_async_overlap/tree_sweep/my_run/profile_tree
```

대표 `hit_k1`, `hit_k2`, `miss` step의 PNG가 입력 디렉터리에 생성된다.
특정 step만 보고 싶으면 plotter 인자를 그대로 덧붙인다.

```bash
bash tools/duet_timeline/plot.sh PROFILE_DIR \
  --step-id 120 --out PROFILE_DIR/timeline_step120.png
```

한 상태의 중앙값 한 장만으로 run의 분산을 숨기지 않으려면 p25/p50/p75 대표
step을 함께 만든다.

```bash
python tools/duet_timeline/plot_representatives.py PROFILE_DIR
```

각 `hit_k1`, `hit_k2`, `miss` 상태에서 target full-step duration의
p25/p50/p75를 고르며, draft response marker까지 저장된 step만 사용한다. 결과는
개별 PNG 9장, 선택 근거 `representatives.tsv`, 3×3
`timeline_representatives_contact_sheet.png`다. 행은 cache 상태, 열은 duration
quantile이다. 기존 JSON은 변경하지 않는다.

## Proxy 구간 숫자 요약

```bash
python tools/duet_timeline/summarize_proxy.py PROFILE_DIR
```

cache 상태와 구간별 표본 수, GPU p50/p95, CPU dispatch p50을 출력한다.
구버전 profile의 `proxy_compute_send` 항목도 읽을 수 있지만, 실제 계산과
예약 지연을 구분하려면 이번 변경 이후 새 profile을 수집해야 한다.

## 모델별 PS 준비·다음 명령 대기 표시

`plot_model_wait_timelines.py --source HANDOFF_DIR --output OUTPUT_DIR`는
`plot_model_native_events.py`와 같은 대표 step에 다음 내용을 추가한다.

- `phase2_build` 뒤부터 첫 PS `p2_prepare`까지의 미계측 준비 구간.
- `merge_cache` 이후 미계측 구간, `draft_recv_cmd`, 다음 step의 `draft_recv_request`.
- PS 전체/graph/graph 밖 elapsed 및 위 구간의 전체 표본 통계.

`draft_recv_cmd`는 이전 step ID를 가진다. 같은 요청 안의 merge 종료와 다음
request 수신 사이 CPU chronology로 연결해야 하며 step ID 단독 join을 하면 안 된다.
그림의 빈칸은 이름 없는 실행 구간일 수 있어 GPU idle로 부르지 않는다.
명령 수신에는 통신·대기가 섞인다. 기존 원시 파일과 이전 그림은 수정하지 않는다.

2026-10-15 산출물은 `results/mlsys_ver11/model_breakdowns_20261015/PS_GAPS.md`,
`NATIVE_EVENTS_WITH_WAITS.pdf`, `PS_GAPS_AND_WAITS.csv`다. 22개 CPU 검사와
1,309개 step 경계/원본 해시 검증은 `WAITS_VALIDATION.json`에 기록했다.

모델별 native/waits 그림의 흰 점선은 DS/PS forward 횟수만큼 graph를 등분한
횟수 안내다. 개별 forward 시간은 계측되지 않았다. `PS prepare A/B`는 각각
root/가시성 준비와 graph 입력 버퍼 준비이며, 다음 요청 수신은 대기 회색과
구분하는 파란색이다. 추가 표시 검증은 `FORWARD_MARKERS_VALIDATION.json`에 있다.
