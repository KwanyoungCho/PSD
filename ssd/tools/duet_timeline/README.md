# DUET timeline — 현재 사용할 도구

현재 배포 기준은 [MLSys-ver1.3](../../../MLSYS_VER13_HANDOVER.md)이다. 동봉된 과거 그림의
측정 버전은 보존한다. 아래 기본 모델 모음 외에 로컬 tree/chain 진단은
`results/mlsys_ver12/local_phase_latency_20261015/`와 `chain40_k10_20261016/`에 있다.
새 runtime이 반영됐다고 과거 timeline이1.3 실측이 되는 것은 아니다.

이번 도구는 tree의 optional `phase_probe_request_*.json`에서 실제 forward1회 시간을 읽고,
`preparation_*.jsonl`의 compile/capture 영향 step을 제외한다. Input spec 최상위
`skip_initial_steps`는 요청당 초기 제외 수(기본2)다. Probe가 없으면 graph/K를 유지한다.
Native chain의 portable window는 실제 `phase1_replay`/`phase2_replay`를 각각 번호로 표시한다.
Generic 새-profile exporter는 tree용이고, chain 변환은 동봉된 chain 진단 adapter를 참고한다.

모델별 그림은 **`results/mlsys_ver12/model_timelines/REPORT.md`**에서만 확인한다.
모델 조합당 `TIMELINE.pdf` 하나, 전체 모음 `ALL_MODELS.pdf`, 브라우저 `index.html`로 통일했다.
기존 ALIGNED/NATIVE/WAITS/별도 예시의 중복 PDF·PNG·HTML은 삭제했다.
원시 JSON·CSV·검사 기록과 실험별 전후 비교 증거는 보존한다.

## 모델별 그림 다시 만들기

저장소 루트에서 실행한다. numpy, matplotlib, pypdf가 필요하며 GPU/모델은 필요 없다.

```bash
python ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest results/mlsys_ver12/model_timelines/SOURCES.json \
  --output results/mlsys_ver12/model_timelines
```

`SOURCES.json`과 `data/`의 portable window JSON 및 `PHASE_SAMPLES.json.gz`가 있으면 다른 서버에서도 다시 그릴 수 있다.
이전 서버의 원시 profile 경로를 읽지 않는다. 출력은5개 모델 PDF/전체17쪽이며,
CodeLlama의 두 설정은 같은 PDF 안에 묶는다. 제목과 설명에 실제 측정 버전을 보존한다.
이것은 기존 기록의 재시각화이며1.2에서 모든 모델을 새로 측정했다는 뜻이 아니다.

다른 서버에서 저장된 그림만 확인할 때는 `ALL_MODELS.pdf` 또는 `index.html`을 열면 된다.
Python은 재생성할 때만 필요하다. 도구는 torch/CUDA/모델을 import하지 않는다.
별도 분석 환경이 필요하면 아래처럼 설치한다(추론용 환경을 바꿀 필요 없음).

```bash
python3 -m venv /tmp/duet-timeline-venv
/tmp/duet-timeline-venv/bin/python -m pip install numpy matplotlib pypdf
```

페이지마다 다음을 표시한다.

- 같은 요청의 **연속 두 step**, target와 draft의 공통 시간축.
- 준비·graph·변환·merge·명령 수신의 기존 상세 색/무늬.
- **빨간 점선:** 다음 target `target_send_request` 시작. Draft 준비의 관측된 deadline 기준.
- **초록 원 / 삼각형:** PS cache 완성 / 후처리까지 끝낸 draft 수신 준비.
- **Ready margin:** deadline−수신 준비. 음수도 그대로 표시한다.
- **흰 점선:** DS/PS forward 횟수. 실측 개별 forward 시간 경계가 아니다.
- **하단 표:** 현재 P1 hit / P2 hit / Miss / Overall별 모든 유효 step의 평균.
  대표 두 step의 평균이 아니다. Step 길이는 현재 target 요청 시작부터 다음 시작까지다.
  DS spare는 proxy 전송 준비 완료−DS 완료, PS spare는 다음 target 요청 시작−draft 수신 준비다.
  Graph/K1·K2는 샘플링·선택을 포함한 round당 평균이고 순수 transformer latency가 아니다.
- **Rate·AL·TPS 표:** P1/P2/Miss의 Rate는 전체 유효 step 중 발생 비율이다.
  Overall 행은 전체 hit rate `(P1+P2)/전체`를 표시한다. AL은 recovery/bonus token을
  포함한 `accepted_len` 평균이다. Profile TPS는 같은 유효 step의 **실제 반환 토큰 합 / 시간 합**이다.
  Per-step TPS의 평균이나 full dataset의 비계측 TPS가 아니다.

평균은 `PHASE_AVERAGES.csv/json`으로도 저장한다. Overall은 실제 step 수로 가중하고
빈 source는 N/A로 표시한다. 원시 시각이 담긴 `data/PHASE_SAMPLES.json.gz`로 재계산하며,
원본 profile 경로 없이도 그림과 평균을 모두 재생성할 수 있다.

현재 동봉된 모델별 trace의 `exit_proxy_side`와 `graph_post`는 겹치지 않는다. 새 기록에서는
중첩 건수·시간을 다시 계산해 표시한다. 이벤트 이름은 별도 stream 사용을 보장하지 않는다.
비동기 전송 enqueue는 실제 통신 완료 계측이 아니다.

미계측 간격은 GPU idle의 증거가 아니며, 다음 응답 대기 전체를 이전 PS 지연으로 귀속하지 않는다.
여유 시간을 전부 추가 forward에 쓸 수 있다는 보장도 아니다. 시간축 이동과 최소 막대 폭 없이
원시 시각을 사용하고, GPU 간 시각 정렬 오차를 인정한다. 중첩 부모/자식 구간은 합산하지 않는다.

## 새 실험에서 평균 표와 모델별 그림 만들기 — 권장

**현재 지원 범위는 B=1, DS와 PS가 모두 실행되는 tree profile이다.**
B>1의 request별 hit가 섞인 batch를 임의로 P1/P2/miss 하나로 분류하지 않는다.
추론을 먼저 완료하고 profile flush가 끝난 뒤 다음 후처리를 실행한다. 기존 그림 재생성과
새 profile의 평균 계산은 서로 다른 작업이며, 새 실험에는 이 exporter를 사용한다.

`timeline_inputs.json`을 만든다. 아래 상대 경로는 이 JSON이 있는 디렉터리 기준이다.
같은 모델·설정의 반복은 `runs`에 추가하고, 다른 K1/K2·tree 예산·온도·하드웨어는
별도 dataset으로 작성한다. `measurement`에 실제 commit·GPU·입력 수·온도 등을 적는다.

```json
{
  "schema": "duet-timeline-inputs-v1",
  "datasets": [{
    "model_id": "llama2_7b_amd135m",
    "model_name": "LayerSkip Llama2-7B + AMD-Llama-135M",
    "setting": "exit20 K1/K2=6/3 DS/PS M=12/8 C=3",
    "measurement": "실제 측정 commit / GPU / B1 BF16 T0.7 / 입력 범위",
    "runs": [{"outcomes": "run1/run.json", "profile_dir": "run1"}]
  }]
}
```

```bash
python ssd/tools/duet_timeline/export_profile_timelines.py \
  --spec /path/to/timeline_inputs.json \
  --output /path/to/new_timeline_bundle
```

이 명령 하나로 유효 step 집계 → source별 대표 선택 → 연속 두 step 추출 → 평균 표·PDF·HTML까지 생성한다.
출력은 새/빈 디렉터리를 지정한다. 이미 export한 묶음은 위의 `build_model_timelines.py`로 재생성한다.

입력 형식:

- 기존 benchmark의 `run.json` (`status=complete`, `args`, `warm_groups`, `groups`)를 직접 읽는다.
  `args.k1/k2`, group의 단일 request, step별 `events[0].source` 및 `cap/clipped`를 사용한다.
- 다른 실행기로 측정했다면 아래 **표준 outcome JSON**으로 변환한다. source는 반드시 실제
  verification의 `0=miss, 1=P1 hit, 2=P2 hit`다. 파일명·막대 길이로 추정하지 않는다.

```json
{
  "schema": "duet-timeline-outcomes-v1", "status": "complete",
  "batch_size": 1, "k1": 6, "k2": 3,
  "requests": [{
    "epoch": 0, "uid": "warmup_or_measured_uid", "warmup": true,
    "steps": [
      {"step_id": 1, "source": 0, "cap": false, "clipped": false,
       "accepted_len": 2, "emitted_len": 2, "context_tokens": 128, "verify_width": 3},
      {"step_id": 2, "source": 1, "cap": false, "clipped": false,
       "accepted_len": 4, "emitted_len": 4, "context_tokens": 130, "verify_width": 12}
    ]
  }]
}
```

예시는 필드 설명용이다. 실제로는 **warmup까지 포함한 모든 decode 요청·step**을 순서대로
기록해야 한다. epoch는 target/draft profile에서 step ID가 1로 재시작하는 순서(0부터)다.
생성 없이 끝나 profile anchor가 없는 요청에는 epoch를 부여하지 않는다. Target rank0와
draft의 anchor 집합이 outcome JSON과 완전히 일치하지 않으면 중단한다.

AL/TPS를 표시하려면 각 step의 `accepted_len`과 `emitted_len`을 제공한다.
`emitted_len`은 scheduler가 실제 추가한 토큰 수이며 `accepted_len`에서 추측하지 않는다.
필드가 없는 과거 기록도 latency·rate는 표시하지만 AL/TPS는 N/A로 남긴다.
선택 필드 `context_tokens`는 step 시작 전 확정된 KV 문맥 길이(이전 생성 토큰 포함),
`verify_width`는 root를 제외한 실제 검증 proposal 수다. 논리적 target query 수는
`verify_width+1`이며 CUDA graph bucket의 padding 후 행 수와 구분한다.
기존 benchmark `run.json` adapter는 이 필드들을 자동으로 가져온다.

평균에서는 warmup, 요청마다 처음 두 step, 마지막 step, 출력 상한/cap·clipped step을 제외한다.
대표 그림은 그 중 같은 요청의 후속 두 anchor가 있는 step에서 source별 step 길이 중앙값에
해당하는 실제 step을 고른다. **평균 표는 그 대표만이 아니라 모든 유효 step을 사용한다.**
표본이 없는 source는 N/A, 그림만 선택할 수 없는 source는 평균 표에 남고 페이지는 없다.
제외 수와 대표 선택 가능 표본 수는 `data/PHASE_SAMPLES.json.gz`의 `eligibility`에 보존한다.

출력에는 `SOURCES.json`, `data/`, `PHASE_AVERAGES.csv/json`, 모델별 PDF/PNG,
`ALL_MODELS.pdf`, `index.html`, `REPORT.md`, hash를 기록한 `INDEX.json`이 포함된다.
**다른 서버에는 출력 디렉터리를 통째로 복사하면 된다.** 새 서버에서는 원래 profile 경로에
접근하지 않고 `SOURCES.json` + `data/`로 평균과 그림을 모두 재생성한다. 각 window의 원본
파일 경로는 provenance이며 재생성 시 읽지 않는다.

`phase_averages.py --historical-source ... --accepted-runs ...`는 동봉된 과거 캠페인의
최초 변환용 adapter다. 새 실험에서는 모델명·반복 수가 고정된 그 adapter를 사용하지 않는다.

## 단일 대표 step만 빠르게 확인 — 전체 평균 표 없음

```bash
python ssd/tools/duet_timeline/plot_consecutive_steps.py \
  --profile-dir /path/to/profiles --request-epoch 2 --step-id 78 \
  --k1 6 --k2 3 --title 'Model / measured version / parameters' \
  --output /tmp/duet-profile-inspection
```

`request-epoch`는 전체 marker를 CPU 시간순으로 정렬해 step ID가 재시작할 때 증가하는
0-based 요청 순번이다. 같은 요청의 세 번째 anchor가 있어야 두 번째 deadline을 알 수 있다.
누락된 경계, 잘못된 command 연결, 역전된 시각은 조용히 보정하지 않고 거부한다.
실제 실행한 K1/K2를 입력해야 한다. `phase2_build`/`merge_cache`/handshake가 있는 DS/PS tree trace용이다.

특정 window만 재생하려면:

```bash
python ssd/tools/duet_timeline/plot_consecutive_steps.py \
  --window results/mlsys_ver12/model_timelines/data/accepted_source1_req26_step7.json \
  --output /tmp/duet-window-inspection
```

출력은 임시 분석용으로 사용한다. 모델별 정본을 갱신할 때는 `SOURCES.json`에 선택 근거·state·window hash를
반영하고 평균 입력도 함께 제공해야 한다. 권장 exporter가 이를 자동으로 생성한다. 옛 `plot_model_*` adapter는 내부 함수와 과거 분석
재현을 위해 남겨 뒀지만 현재 모델별 그림을 만드는 진입점은 `build_model_timelines.py` 하나다.

## 프로파일 수집

실험 실행 환경에 추가한다.

```bash
SSD_PROFILE_DUET=1
SSD_PROFILE_DUET_DETAIL=1
SSD_PROFILE_DIR=/absolute/path/to/profile_dir
```

`exit_logits`, `exit_proxy_launch`, `exit_proxy_side`, `tree_proxy_graph_replay`,
`proxy_send_enqueue` 등을 기록한다. Proxy enqueue 완료는 실제 draft 도착 시각이 아니다.
DS/PS graph replay에는 forward 외 tree 선택·sampling도 포함된다.

PS가 없는 chain 등의 기존 trace는 `plot.sh`/`plot_representatives.py`로 따로 탐색할 수 있다.
모델별 정본을 다른 형식으로 덮어쓰지 않는다. Proxy 구간 수치만 확인하려면
`python ssd/tools/duet_timeline/summarize_proxy.py PROFILE_DIR`를 사용한다.

## 도구 검사

```bash
python -B -m unittest discover -s ssd/tests -p test_consecutive_timeline.py -v
```

음수 여유, 경우별 가중 평균, hit/AL/TPS의 분모·누락 토큰 처리, 빈 표본, 요청 경계, 중첩 판별 및 **원본 profile 삭제 후 다른
경로로 옮긴 bundle 재생성**을 검사한다. 이번 전달의 전체 검사 기록은
`results/mlsys_ver12/model_timelines/checks/CHECKS.json`에 있다.
