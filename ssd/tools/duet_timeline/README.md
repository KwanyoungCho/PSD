# DUET timeline — 현재 사용할 도구

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

`SOURCES.json`과 `data/`의 portable window JSON만 있으면 다른 서버에서도 다시 그릴 수 있다.
이전 서버의 원시 profile 경로를 읽지 않는다. 출력은5개 모델 PDF/전체17쪽이며,
CodeLlama의 두 설정은 같은 PDF 안에 묶는다. 제목과 설명에 실제 측정 버전을 보존한다.
이것은 기존 기록의 재시각화이며1.2에서 모든 모델을 새로 측정했다는 뜻이 아니다.

페이지마다 다음을 표시한다.

- 같은 요청의 **연속 두 step**, target와 draft의 공통 시간축.
- 준비·graph·변환·merge·명령 수신의 기존 상세 색/무늬.
- **빨간 점선:** 다음 target `target_send_request` 시작. Draft 준비의 관측된 deadline 기준.
- **초록 원 / 삼각형:** PS cache 완성 / 후처리까지 끝낸 draft 수신 준비.
- **Ready margin:** deadline−수신 준비. 음수도 그대로 표시한다.
- **흰 점선:** DS/PS forward 횟수. 실측 개별 forward 시간 경계가 아니다.

미계측 간격은 GPU idle의 증거가 아니며, 다음 응답 대기 전체를 이전 PS 지연으로 귀속하지 않는다.
여유 시간을 전부 추가 forward에 쓸 수 있다는 보장도 아니다. 시간축 이동과 최소 막대 폭 없이
원시 시각을 사용하고, GPU 간 시각 정렬 오차를 인정한다. 중첩 부모/자식 구간은 합산하지 않는다.

## 새 profile에서 두 step 추출

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
반영하고 `build_model_timelines.py`를 사용한다. 옛 `plot_model_*` adapter는 내부 함수와 과거 분석
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
