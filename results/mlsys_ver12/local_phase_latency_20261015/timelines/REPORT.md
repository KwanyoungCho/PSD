# 모델 조합별 timeline — 현재 확인 경로

이 폴더는 입력 manifest에 지정한 모델별 timeline 모음이다.
원시 시각·평균 재현 입력을 함께 보존하며 새로운 inference를 실행한 것은 아니다.

- [전체 PDF](ALL_MODELS.pdf), [브라우저에서 모델별 보기](index.html).
- 모델당 PDF 하나, 페이지마다 **같은 요청의 연속 두 step**. P1 hit, P2 hit, miss 순서다.
- 같은 모델의 여러 설정은 같은 PDF에 묶는다. 유효한 연속 step 표본이 없는 source는 그림을 만들지 않는다.
- 그래프마다 측정 설명, 설정, 요청·step을 표시한다. 도구 버전과 실제 측정 runtime 버전을 구분한다.

| 모델 조합 | 쪽수 | 그림 | 측정 구분 |
|---|---:|---|---|
| LayerSkip Llama2-7B + llama-68m | 3 | [PDF](llama2_7b_68m/TIMELINE.pdf) | Fresh local orion RTX4090; B1 BF16 T0.7; 12 fixed turns x 2; first10/terminal/cap excluded; GPU forward markers enabled |
| LayerSkip Llama2-13B + llama-68m | 3 | [PDF](llama2_13b_68m/TIMELINE.pdf) | Fresh local orion RTX4090; B1 BF16 T0.7; 12 fixed turns x 2; first10/terminal/cap excluded; GPU forward markers enabled |

## 표시 읽는 법

- 빨간 점선: 다음 step의 `target_send_request` 시작. Draft 준비의 관측된 deadline 기준이다.
- 초록 원: PS cache merge 완료. 삼각형: 후처리까지 끝나고 다음 명령 수신을 시작한 시각.
- Ready margin = 다음 target send 시작 − draft 수신 준비. 음수는 준비가 늦었다는 관찰이다.
- 다음 response wait에는 cache 조회·통신·miss draft도 포함되므로 이전 PS의 지연으로 모두 귀속하지 않는다.
- 흰 점선은 DS/PS forward 횟수 안내다. Graph 내부 개별 forward 경계를 측정한 것이 아니다.
- 사선 빈 구간은 미계측 elapsed이며 GPU idle의 증거가 아니다. 부모·자식 구간을 합산하지 않는다.
- 시간축 이동과 최소 막대 폭 없이 원시 시각을 사용한다. GPU 간 시각 정렬 오차가 있고,
  관측된 여유를 전부 추가 계산에 사용할 수 있다는 보장은 없다.

## 경우별·전체 평균

그래프 하단의 절대 시각 표를 **P1 hit / P2 hit / Miss / Overall** 평균 표로 교체했다.
그림의 두 step은 실제 예시이며, 표는 같은 모델·설정의 유효 step 전체를 사용한다.
분류는 현재 step에서 사용한 cache source다. 그 step의 DS/PS는 다음 결과를 대비하는 생성이다.
전체 평균은 실제 step 수로 가중하며, 표본이 없는 경우는 N/A로 표시한다.

성능 표와 latency 표는 **동일한 유효 step 집합**이다.

- Rate: 해당 경우 step 수 / 전체 step 수. P1 hit·P2 hit·Miss 비율의 합은100%다.
  **Overall 행의 Rate는 전체 hit rate, 즉 (P1+P2)/전체**다.
  표본이 없는 source의 Rate는0%, 조건부 AL/TPS는N/A다.
- AL: 원시 `accepted_len` 평균. Correction/recovery 또는 bonus token을 포함한다.
- Profile TPS: **실제로 반환한 토큰 합 / 아래 Step 시간의 합**이다. Per-step TPS의 평균이 아니다.
  Profile을 켠 제한된 표본의 값이며 full-dataset/비계측 TPS로 인용하지 않는다.
- 반환 토큰 수가 없으면 AL로 대체하지 않는다. 일부 step의 기록이 빠진 행도 AL/TPS는N/A다.

- Step: 현재 `target_send_request.start`부터 같은 요청의 다음 `target_send_request.start`까지.
  요청/응답·검증·후처리·다음 요청까지의 간격을 포함한다. `graph_pre` 간격이나 AL이 아니다.
- DS total: root 준비 시작부터 `proxy_wait.start`까지. Glue와 proxy 대기는 제외한다.
- DS spare*: target의 `proxy_send_enqueue.end` − DS 완료. 실제 수신 도착 시점은 계측되지
  않았으므로 **전송 준비 시각 기준**을 사용한다. 과거 보고서의 transport를 더한 추정 gap1과 다르다.
  `proxy_wait.end`로 계산하면 DS가 늦어 이미 도착한 proxy를 기다리게 한 경우를 숨기므로 사용하지 않는다.
- PS total: `proxy_wait.end`부터 `merge_cache.end`까지.
- PS spare: 다음 target 요청 시작 − 다음 draft 명령 수신 준비. Cache 완료 후 정리 시간까지 고려한다.
- DS/PS graph ÷ K1/K2: forward round당 평균. Graph 안의 sampling·tree 선택을 포함하며,
  개별 transformer forward를 분리 측정한 수치가 아니다. Source에 따라 입력 width도 다르다.
- Spare는 음수를 유지한다. 추가 forward에 전부 사용할 수 있는 확정 예산이나 인과적 stall 측정값은 아니다.

새 profile exporter는 warmup·요청의 처음 두 step·마지막 step·cap/clip step을 제외한다.
기존 자료의 필터는 함께 보존된 `eligibility`를 따른다. 지연시간이 길다는 이유로 제외하지 않는다.
입력 수·반복·실측 환경은 설정별 측정 설명을 따른다. 반복을 서로 다른 입력으로 세지 않는다.

`PHASE_AVERAGES.csv/json`은 아래 평균과 일치한다. `data/PHASE_SAMPLES.json.gz`에는
모든 대상 step의 원시 시각·현재/다음 source·K1/K2와 입력 hash를 보존했다.
다른 서버에서도 이 파일로 평균을 다시 계산한다.

### LayerSkip Llama2-7B + llama-68m — Exit 21; K1/K2 3/3; DS roots/pos 3, PS roots 9; W1/W2 9/9; C3; M40

Fresh local orion RTX4090; B1 BF16 T0.7; 12 fixed turns x 2; first10/terminal/cap excluded; GPU forward markers enabled

| Current outcome | Steps | Rate (%) | AL (tokens/step) | Profile TPS (tokens/s) |
|---|---|---|---|---|
| P1 hit | 698 | 42.329 | 2.292 | 88.405 |
| P2 hit | 712 | 43.178 | 2.409 | 93.290 |
| Miss | 239 | 14.494 | 1.904 | 75.899 |
| Overall | 1649 | 85.506 | 2.286 | 88.754 |

| Current outcome | Steps | Step (ms) | DS forward (ms) | DS compute* (ms) | DS spare* (ms) | PS forward (ms) | PS compute* (ms) | PS spare (ms) |
|---|---|---|---|---|---|---|---|---|
| P1 hit | 698 | 25.929 | 0.191 | 7.580 | +7.008 | 0.168 | 2.153 | +1.648 |
| P2 hit | 712 | 25.819 | 0.200 | 6.266 | +5.666 | 0.168 | 2.488 | +1.983 |
| Miss | 239 | 25.083 | 0.169 | 7.864 | +7.359 | 0.169 | 2.478 | +1.972 |
| Overall | 1649 | 25.759 | 0.192 | 7.054 | +6.479 | 0.168 | 2.345 | +1.840 |

유효 step 평균 target query: 12.629개(padding 전), 기존 KV 문맥: 300.333 tokens. Query 수와 문맥 길이는 다른 양이다.

### LayerSkip Llama2-13B + llama-68m — Exit 26; K1/K2 3/3; DS roots/pos 3, PS roots 9; W1/W2 9/9; C3; M40

Fresh local orion RTX4090; B1 BF16 T0.7; 12 fixed turns x 2; first10/terminal/cap excluded; GPU forward markers enabled

| Current outcome | Steps | Rate (%) | AL (tokens/step) | Profile TPS (tokens/s) |
|---|---|---|---|---|
| P1 hit | 741 | 46.633 | 2.509 | 77.841 |
| P2 hit | 592 | 37.256 | 2.336 | 70.942 |
| Miss | 256 | 16.111 | 1.828 | 68.454 |
| Overall | 1589 | 83.889 | 2.335 | 73.885 |

| Current outcome | Steps | Step (ms) | DS forward (ms) | DS compute* (ms) | DS spare* (ms) | PS forward (ms) | PS compute* (ms) | PS spare (ms) |
|---|---|---|---|---|---|---|---|---|
| P1 hit | 741 | 32.229 | 0.192 | 11.425 | +10.848 | 0.168 | 4.263 | +3.760 |
| P2 hit | 592 | 32.930 | 0.200 | 10.981 | +10.381 | 0.168 | 4.381 | +3.876 |
| Miss | 256 | 26.706 | 0.169 | 9.009 | +8.503 | 0.169 | 2.866 | +2.361 |
| Overall | 1589 | 31.601 | 0.191 | 10.870 | +10.296 | 0.168 | 4.082 | +3.578 |

유효 step 평균 target query: 12.785개(padding 전), 기존 KV 문맥: 273.986 tokens. Query 수와 문맥 길이는 다른 양이다.


## Proxy와 target 후반 forward의 중첩 여부

`exit_proxy_side`와 `graph_post`의 원시 시간 구간이 실제로 겹치는지 계산했다.
`PHASE_AVERAGES.json`의 설정별 `proxy_overlap`에 전체 대상 step의 중첩 건수와 평균을 저장했다.
`exit_proxy_side`는 별도 stream을 사용하지 않는 분기에도 붙는 이벤트 이름이다.
별도 stream 사용은 runtime의 `SSD_PROXY_STREAM=1` 또는 exit replica 경로에서 결정된다.
전송 막대 `proxy_send_enqueue`는 비동기 전송 요청 구간이며 실제 통신 완료를 계측하지 않는다.
따라서 통신이 겹치는지·얼마나 겹치는지를 이 막대만으로 판단하지 않는다.
이는 별도 draft GPU의 DS/PS가 target verification과 병렬 실행되는 것과 다른 문제다.


- LayerSkip Llama2-7B + llama-68m / Exit 21; K1/K2 3/3; DS roots/pos 3, PS roots 9; W1/W2 9/9; C3; M40: Proxy 계산 ↔ target 후반 forward 중첩: 0/1649 step, 전체 step 평균 중첩 0.000ms. 전송 완료 시간은 미계측.
- LayerSkip Llama2-13B + llama-68m / Exit 26; K1/K2 3/3; DS roots/pos 3, PS roots 9; W1/W2 9/9; C3; M40: Proxy 계산 ↔ target 후반 forward 중첩: 0/1589 step, 전체 step 평균 중첩 0.000ms. 전송 완료 시간은 미계측.

## 측정 조건과 대표 선택

측정 버전·모델·입력 범위는 위 설정별 설명을 확인한다. 각 window의 `metadata.selection`에
대표 선택 근거가 있다. 새 exporter는 같은 source에서 후속 두 anchor가 있는 유효 step 중
현재 target 요청 시작→다음 요청 시작 시간의 중앙값에 해당하는 실제 step을 선택한다.
표는 대표 step만이 아닌 모든 유효 step의 평균이다. 이 도구는 파라미터 최적화나
전체 데이터셋 평가를 수행하지 않는다. 서로 다른 환경의 절대 시간을 직접 비교하지 않는다.

`SOURCES.json`과 `data/`의 window JSON 및 PHASE_SAMPLES.json.gz는 새 그림의 재현 입력이다. 원본 interval과
source hash를 포함한다. `INDEX.json`, 모델별 `AUDIT.json`, `DEADLINES.csv`에서
페이지별 source·step·정확한 시각을 확인할 수 있다.

저장소 루트에서 재생성:

```bash
python /path/to/repo/ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest /path/to/this_bundle/SOURCES.json \
  --output /path/to/redrawn_bundle
```

필요 패키지는 numpy, matplotlib, pypdf다. GPU·모델·이전 서버의 파일 경로는 필요 없다.
새 실험에서 만드는 명령과 지원 입력 형식은 도구의 README를 참고한다.
