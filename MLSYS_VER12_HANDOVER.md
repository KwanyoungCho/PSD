# MLSys ver1.2 — 채택한 PS 준비 최적화와 상세 timeline

2026-10-15 KST. **이 서버의 다음 작업 기준은 `MLSys-ver1.2`다.** 작업 폴더 이름은
`/home/chokwans99/PSD-MLSys-ver0.1` 그대로지만 실제 Git 브랜치는1.2다.
다른 서버에서 진행 중인 알고리즘은 이 기준 위에서 별도로 병합한다.

## 이번 전달에서 바뀐 것

| 분류 | 반영 내용 | 근거/주의점 |
|---|---|---|
| 실행 코드 | PS root 매핑 재사용, 조상 마스크 GPU 유지, 여섯 유효성 검사를 한 kernel로 통합 | PS 전체 5.269→4.928ms, -6.48%; Llama2-7B+AMD135M, B1/4090의 고정 설정 |
| 실행 코드로 미채택 | 추가 fan-out readback 지연·host 문맥 길이 재사용 | 추가 차이 -0.017ms의 CI가0 포함. 후보와386개 검사 기록은 연구 자료로만 보존 |
| 분석 도구 | 기존 상세 이벤트/색/무늬, PS 준비 A/B, 미계측 간격, 명령 수신 | 새로운 runtime 계측을 추가한 것이 아닌 기존 profile 후처리 |
| 두-step 그림 | 같은 요청의 연속 두 step, 다음 target 요청 시작 deadline, cache/receive-ready 여유 | 원시 시간 보존. 다른 서버에서 JSON만으로 재생 가능 |
| 평균 표·새 profile 이식 | P1 hit/P2 hit/miss/전체의 step·DS/PS 시간·여유·graph/K, 새 profile exporter | 모든 유효 step 평균, 표본 수 공개. 원본 서버 없이 재생성 검사 |

이번1.2는 **후보 수식, 위치 수식, tree 확장 정책, K1/K2, 입력 폭, G=M 계약을 바꾸지 않는다.**
기존1.1에 들어 있던 알고리즘을 유지한다. 새 사용자 옵션도 없다. 파라미터는
[ver1.1 문서](MLSYS_VER11_HANDOVER.md)를 따르며 이번 latency 비교 설정을 모델별 최적값으로 사용하지 않는다.

## 브랜치 받기와 검토 순서

저장소는 `https://github.com/KwanyoungCho/PSD.git`, 받을 브랜치는 **`MLSys-ver1.2`**다.
기준은 `MLSys-ver1.1@0916871`이다. 기존 서버 작업을 덮어쓰지 않도록 별도 worktree에서 검토한다.

```bash
git fetch origin MLSys-ver1.2
git worktree add -b review/mlsys-ver1.2 ../PSD-MLSys-ver1.2-review origin/MLSys-ver1.2
cd ../PSD-MLSys-ver1.2-review
git rev-parse HEAD
git log --oneline 0916871..HEAD
git diff 0916871..HEAD -- ssd/ssd
```

커밋은 실행 코드, 도구/검사, 보고서/실험 증거 순서로 나눴다. 실행 코드만 검토할 때는
**`9390e21`**을 본다. 바뀐 runtime 파일은 `draft_runner.py`, 새 `helpers/p2_prepare.py` 두 개다.
`test_p2_preparation.py`의12개 검사가 같은 커밋에 있다. runtime91파일 전체의 해시가
기존382개 회귀 통과 당시 소스와 일치한다. 외부 서버의 `draft_runner.py` 전체를 덮어쓰지 말고
이 변경 구간만 비교하여 충돌을 해결한다.

검토할 자료:

1. [최종 반영 보고서](results/mlsys_ver12/REPORT.md): 변경, 측정 범위, 남은 비용, 그림 해석.
2. [채택 근거](results/mlsys_ver11/ps_preparation_20261015/REPORT.md): 동일 step breakdown, 세 반복, profile OFF parity.
3. [추가 후보 미채택 근거](results/mlsys_ver11/ps_device_20261015/REPORT.md): CI·후보 소스·복원 증거.
4. [버전1.2 검사](results/mlsys_ver12/checks/CHECKS.json): runtime 동일성,31개 offline 검사,최초 전달의16쪽 두-step 그림 검증; 현재17쪽 모델별 모음은 `model_timelines/checks/CHECKS.json` 참조.
5. [timeline 사용법](ssd/tools/duet_timeline/README.md): 새 실험 exporter, 표준 outcome JSON, 저장된 그림 재생 명령.

과거 실험 보고서의 “작업 트리”, “새 커밋·push 없음”은 **측정·보고 당시 기록**이므로
원본/해시 manifest를 보존했다. 현재 채택 및 전달 상태는 이 문서를 기준으로 확인한다.
`ps_device_20261015/candidate_source` 아래 코드는 미채택 후보 snapshot이며 production import 경로가 아니다.
그 파일이나 `RUNTIME_INCREMENTAL.patch`를 현행 runtime에 덮어쓰지 않는다.

## 검증 재현

이미 GPU 회귀 **40 suites/382개, 오류·실패·skip0**을 통과한 runtime과 동일하다.
최초 전달의 timeline 검사는31개다. 평균 표·서버 이동 기능을 추가한 이번 전달에서는
동일성 해시 확인 및 timeline 관련 **39개 CPU 검사가 모두 통과했다**(오류·실패·skip 0).
실제 797 step의 새 exporter 재집계도 기존 네 행의 평균과 일치했다.
최종 기록은 `model_timelines/checks/CHECKS.json`과 `EXPORT_REPLAY.json`을 확인한다.
GPU와 CPU 검사에는 중복 suite가 있으므로 합산해서 독립 검사 수로 쓰지 않는다.
미채택 후보의386개 검사는 배포 runtime의 검사 수가 아니다.

```bash
export DUET_PY=/path/to/venv/bin/python
# GPU 불필요: numpy, matplotlib, pypdf가 있는 Python
"$DUET_PY" -B results/mlsys_ver12/validate_release.py --output /tmp/mlsys-ver12-offline

# 새 하드웨어로 이식한 후에는 해당 서버의 CUDA/FlashInfer 환경에서 실행
export PYTHONPATH="$PWD/ssd:$PWD/ssd/tests"
export PYTHONDONTWRITEBYTECODE=1
# CUDA_VISIBLE_DEVICES, SSD_CUDA_ARCH, SSD_HF_CACHE, SSD_DATASET_DIR는 서버에 맞게 설정
"$DUET_PY" -B results/mlsys_ver11/check_suites.py --output-dir /tmp/mlsys-ver12-cuda
```

`validate_release.py`는 이 릴리스의 runtime 동일성을 검사한다. 다른 알고리즘을 병합한
뒤에는 이 검사가 실패하는 것이 정상이며, 이후 소스는 새 provenance와 회귀 결과로 검증한다.
원래 checks/raw를 새 실행 결과로 덮어쓰지 않는다. 예전 검사 건수만 기대하는 대신
실제 suite/test/skip/실패·오류 수와 종료 코드, `source_unchanged`를 확인한다.

## 그림부터 확인하기

**[모델별 timeline 보고서](results/mlsys_ver12/model_timelines/REPORT.md)를 시작점으로 사용한다.**
[전체 PDF](results/mlsys_ver12/model_timelines/ALL_MODELS.pdf)와 모델 조합별 PDF5개로 통일했다.
CodeLlama의 두 설정은 같은 PDF에 묶었다. 이전 모델별 timeline/별도 예시의 중복 파일은 삭제했다.
원시 JSON·CSV와 실험별 A/B 증거는 보존했다. 새 그림은 기존 profile의 재시각화다.
최초 전달의 `EVIDENCE_MANIFEST.json`과 `checks/`는 Git `1f87514` 당시의 기록이다.
그림 정리 이후에는 `model_timelines/INDEX.json`, `CLEANUP.json`, `checks/CHECKS.json`을
사용한다. 삭제한 옛 그림이 최초 manifest에 남아 있다는 이유로 raw 손실로 판단하지 않는다.

빨간 점선은 **다음 target 요청 시작**이다. 초록 원은 PS cache 완성, 삼각형은
후처리를 끝내고 draft가 다음 명령을 받을 준비를 마친 시각이다. 두 시각을 구분한다.
회색 command 수신 구간 전체가 추가 forward를 넣어도 되는 여유 시간은 아니다.
표의 ready margin은 관측된 차이이며, cross-GPU 시각 정렬 오차 및 시스템 변동이 있어
그만큼 계산을 추가해도 target stall이 없다는 보장이 아니다. 다음 response wait 전체도
이전 PS에 귀속하지 않는다.

그림의 표는 이제 **P1 hit/P2 hit/miss/전체 평균**이다. 대표 두 step만의 평균이 아니며
전체는 실제 step 수로 가중한다. Step 길이, DS/PS 전체 latency, DS/PS 여유, graph/K1·K2와
표본 수를 표시한다. DS 여유는 proxy 전송 준비 완료 기준, PS 여유는 다음 target 요청
시작−draft 수신 준비 기준이다. Graph/K는 sampling·선택을 포함하고 개별 forward 실측은 아니다.
동봉된 기존 trace 3,700 step에서는 proxy 계산과 target 후반 forward의 중첩이 없었다.
새 profile은 중첩 건수와 시간을 다시 계산한다. `exit_proxy_side`라는 이름만으로 별도
stream을 사용했다고 판단하지 않는다. 전송 enqueue는 통신 완료 계측도 아니다.

## 다른 서버에서 새 timeline 만들기

저장된 그림은 다음 명령으로 재생성한다. GPU·모델·원본 서버 profile이 필요 없다.

```bash
python ssd/tools/duet_timeline/build_model_timelines.py \
  --manifest results/mlsys_ver12/model_timelines/SOURCES.json \
  --output /tmp/duet-model-timelines
```

새 실험의 profile과 실제 cache source 기록을 쓰려면 README의 `timeline_inputs.json`을
작성한 후 다음 명령 하나를 실행한다.

```bash
python ssd/tools/duet_timeline/export_profile_timelines.py \
  --spec /path/to/timeline_inputs.json \
  --output /path/to/new_timeline_bundle
```

현재 exporter는 **B=1, DS/PS tree trace**용이다. B>1의 혼합 source를 임의로 하나로
분류하지 않으며, 그 경우는 별도 집계 계약이 필요하다. 기존 benchmark `run.json` 또는
README에 명시한 표준 outcome JSON을 입력한다. 새 모델·반복 수·서버 경로는 코드 수정 없이
spec으로 지정한다. 측정 commit·환경·설정 설명을 실제 실험에 맞게 남긴다.
출력 폴더 전체가 이식 가능한 묶음이다. 원본 서버 경로는 출처로만 남고 재생성 시 접근하지 않는다.
이번 추가 변경은 분석 도구·그림·문서이며 inference runtime과 실행 옵션을 바꾸지 않는다.

## 다른 서버에서 후속 확인할 것

- 70B/Blackwell 및 B>1의 **PS 준비·PS 전체 latency**를 같은 입력/출력/phase 조건으로 재측정한다.
  이번6.48%를 그 환경의 수치로 사용하지 않는다.
- 변경된 tree 알고리즘과 병합하면 root 매핑·조상 마스크 shape, 실제 root R와 forward lane W,
  fallback 경로의 계약을 다시 검사한다. 실패 우선순위를 보존하는 유효성 검사를 생략하지 않는다.
- PS 전체는 proxy 수신 완료→merge 완료다. 현재 최적화는 **draft 측 준비 overhead** 개선이며,
  Transformer 자체/target forward 가속, TPS/AL 증가, 최적 파라미터 확정 결과가 아니다.
- 새 graph key 준비는 기존 prewarm 정책을 유지한다. 후속 비교에서도 준비 영향 step을 공개한다.
