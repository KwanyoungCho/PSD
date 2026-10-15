# 모델 조합별 timeline — 현재 확인 경로

이 폴더가 모델별 timeline의 단일 확인 경로다. 이전 ALIGNED/NATIVE/WAITS 및 별도 예시 PDF는 삭제했다.
원시 profile·수치·이벤트 JSON은 보존했으며 새로운 inference를 실행한 것은 아니다.

- [전체 PDF](ALL_MODELS.pdf), [브라우저에서 모델별 보기](index.html).
- 모델당 PDF 하나, 페이지마다 **같은 요청의 연속 두 step**. P1 hit, P2 hit, miss 순서다.
- CodeLlama 두 설정은 같은 PDF에 묶었으며, 12/6에서 유효한 miss 표본이 없어 그 페이지는 없다.
- 그래프마다 측정 버전, 파라미터, 요청·step을 표시한다. 일부 모델의 과거 trace를1.2 측정으로 취급하지 않는다.

| 모델 조합 | 쪽수 | 그림 | 측정 구분 |
|---|---:|---|---|
| LayerSkip Llama2-7B + llama-68M | 3 | [PDF](llama2_7b_llama68m/TIMELINE.pdf) | 과거 ver1.0 계열 / RTX3090 / 설정별 profile 4턴. 1.2 신규 측정이 아님. |
| LayerSkip Llama2-7B + AMD-Llama-135M | 3 | [PDF](llama2_7b_amd135m/TIMELINE.pdf) | 1.2 채택 소스와 동일 / RTX4090 / PS 최적화 A/B의 고정 설정. 최적 파라미터 결과가 아님. |
| LayerSkip Llama2-13B + AMD-Llama-135M | 3 | [PDF](llama2_13b_amd135m/TIMELINE.pdf) | 과거 ver1.0 계열 / RTX3090 / 설정별 profile 4턴. 1.2 신규 측정이 아님. |
| LayerSkip Llama3-8B + Axcel Llama3.1 drafter-340M | 3 | [PDF](llama3_8b_drafter340m/TIMELINE.pdf) | 과거 ver1.0 계열 / RTX3090 / 설정별 profile 4턴. 1.2 신규 측정이 아님. |
| LayerSkip CodeLlama-34B + AMD-Llama-135M-code | 5 | [PDF](codellama34b_amd135m_code/TIMELINE.pdf) | 과거 ver1.0 계열 / RTX3090 / 설정별 profile 4턴. 1.2 신규 측정이 아님. |

## 표시 읽는 법

- 빨간 점선: 다음 step의 `target_send_request` 시작. Draft 준비의 관측된 deadline 기준이다.
- 초록 원: PS cache merge 완료. 삼각형: 후처리까지 끝나고 다음 명령 수신을 시작한 시각.
- Ready margin = 다음 target send 시작 − draft 수신 준비. 음수는 준비가 늦었다는 관찰이다.
- 다음 response wait에는 cache 조회·통신·miss draft도 포함되므로 이전 PS의 지연으로 모두 귀속하지 않는다.
- 흰 점선은 DS/PS forward 횟수 안내다. Graph 내부 개별 forward 경계를 측정한 것이 아니다.
- 사선 빈 구간은 미계측 elapsed이며 GPU idle의 증거가 아니다. 부모·자식 구간을 합산하지 않는다.
- 시간축 이동과 최소 막대 폭 없이 원시 시각을 사용한다. GPU 간 시각 정렬 오차가 있고,
  관측된 여유를 전부 추가 계산에 사용할 수 있다는 보장은 없다.

## 측정 조건과 대표 선택

3090 모델별 자료는 설정별4턴의 ver1.0 계열 profile이다. 같은 source에서 후속 두 anchor가
있는 step의 target 구간 길이 중앙값을 선택했다. CodeLlama는 별도 code 입력이다.
7B+135M/4090은1.2 채택 소스의24턴 PS 최적화 비교 기록에서 기존 A/B 대표를 사용했다.
이는 파라미터 최적화나 전체 데이터셋 성능 평가가 아니다. 서로 다른 환경의 절대 시간을
직접 비교하여 모델/버전의 우열을 판단하지 않는다.

`SOURCES.json`과 `data/`의 window JSON은 새 그림의 재현 입력이다. 원본 interval과
source hash를 포함한다. `INDEX.json`, 모델별 `AUDIT.json`, `DEADLINES.csv`에서
페이지별 source·step·정확한 시각을 확인할 수 있다.

저장소 루트에서 재생성:

```bash
python ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest results/mlsys_ver12/model_timelines/SOURCES.json \
  --output results/mlsys_ver12/model_timelines
```

필요 패키지는 numpy, matplotlib, pypdf다. GPU·모델·이전 서버의 파일 경로는 필요 없다.
이전 그림은 Git 커밋 `1f87514`에서 복구할 수 있다. 현재 확인에는 이 폴더만 사용한다.
