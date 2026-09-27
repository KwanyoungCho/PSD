# 후속 작업 인계

2026-09-21: [전체 연구 진행 기록](../../DUET_PROGRESS.md)에 calibration까지의 결과,
수정 점수의 논문 근거 논의와 후속 과제를 통합했다. 아래는 기존 실험 인계이며,
full-DUET 보조 검증의 후속 결과는 통합 기록과 calibration 보고서를 함께 본다.

2026-09-13: 사용자는 수식 설명을 아직 이해하지 못했다고 명시했다.
전체 재설명과 이해 확인은 [TODO.md](TODO.md)에 미완료로 남겼다.
현재 후속 연구는 [DUET calibration](../../duet_calibration/README.md)이다.

전체 결과와 판단은 [REPORT.md](REPORT.md), 증명은 [THEORY.md](THEORY.md).
사용자는 MLSys를 위해 DUET 개선 방향의 수식과 실제 검증을 요청했으며, 새 모델/head
학습을 원하지 않는다. 현재 엔진 파일을 변경하지 않고 실험 hook으로 검증했다.

## 핵심 결과

- 확실한 후보 coverage 개선은 **위치 확률 혼합**에서 나왔다.
  $\bar\alpha_i=\sum_v\min(e_i(v),q_i(v))$에서 $\bar h$를 만들고
  $\widetilde h=.75\widehat h+.25\bar h$를 사용한다.
- Full-vocabulary 정규화도 비교했다. T=1에서 단독 +0.412%p, 낮은 T에서는 단독
  이득이 작다. 위치 가중치 변경은 약 +1.6~2.3%p였다.
- 원래 residual `[e-q]+`는 세 T에서 모두 기존 proxy-score 기준선보다 낮은 coverage.
- 부드러운 할인 `e*(1-q)^beta`, floor `max([e-q]+,rho*e)`는 원래 차감의 큰 손실을
  줄이지만, 같은 배분의 proxy보다 강건하게 우월하다는 결과는 없다.
- **초기 T=1 +0.125%p의 source 추가 이득은 확정 결과로 사용하면 안 된다.**
  같은 동결 정책에 실제 wire(topk17 후 앞15)를 적용하니 +0.011%p, CI가 0 포함.
  `runtime_snapshot_checks.json`이 최종 구현 민감도 결과다. 기존 frozen 결과는
  provenance 때문에 보존했으며, 좋은 쪽으로 재선택하지 않았다.
- `numerical_case.json`: T=1 step331, token285/437의 동일한 경계 score를 고르는
  방식으로 한 step의 실제 coverage가 0.982 달라진 사례. BF16 logits tie가 중요하다.

## 데이터 분할과 동결

- 기존 source rows1..64/dataset의 모든 프롬프트는 개발 데이터다. 종전의 seed7
  “confirmation”도 이번에는 개발 자료로 포함했다.
- 새 실제 T=.7/.5 개발 생성은 그중 16개/dataset=64개 프롬프트, seed101.
- 새 확인은 source rows65..88/dataset, 24개씩 총96개, T=1/.7/.5, seed271.
  같은96개에서 288 generation이다. 기존개발과 text hash 중복0.
- 개발512 generation, sampled5796 steps. 확인288 generation, sampled3112 steps.
- 55 sources/rankings ×2 normalization ×24 h variants =2640 조합.
- `frozen.json`: 2026-09-12T13:43:02 UTC, 확인 정책 점수 관찰 전 생성.
  Replay source와 개발 NPZ hash가 동결되어 있다. 개발 freeze를 덮어쓰지 않는다.
- 확인 CSV의 top exploratory policy 순위는 새로운 독립 승자 주장의 근거가 아니다.

## 실행 구성

Target `/home/chokwans99/awq_calibrated/layerskip_llama2_70b`,
AWQ `/home/chokwans99/awq_artifacts/layerskip70b_awq_tp4`,
draft `/data/chokwans99/models/TinyLlama-1.1B-Chat-v1.0`.
`ssd/.venv/bin/python`을 사용했다. `source ssd/env.sh`의 기본 target은 다른 경로라서
실제 benchmark command에서 위 calibrated target을 명시적으로 덮어쓴다.

B1, PS-only chain (`--duet_only_proxy`), K4, exit56, topM17, roots15.
이 플래그는 source=e라는 뜻이 아니다. Source=e는 `--duet_proxy_source proxy`다.
실제 모델 생성은 GPU3..7, 계산 replay는 GPU2에서 실행했다. GPU0/1은 다른 작업 소유다.

현재 chain 후보는 T를 생략하고 raw softmax를 쓴다. 실제 sampler/verify/probe는 요청된
T를 적용한다. 이를 lossless 버그라고 부르지 않는다. Tree 코드에는 과거 속도 실험 후
candidate T 반영을 원복한 주석도 있다. 동일T 비교와 legacyT1 비교를 구분한다.

`runtime_policy.py`는 프로세스 안에서 후보 graph를 바꾼다. 검증/샘플링은 그대로다.
Hook 미호출/다른 eager 후보 경로는 오류로 처리한다. 실제 모든 candidate callback K=4.
GPU/TP/KV 전체 정확성 증명은 아니며, production CPU sampler+verify 30만 회의
현재 T 검사와 기존 코드 hash 비교, GPU graph/tensor 검사를 통과한 범위다.

## 속도 결과의 해석

- `live/`: 4 arms × seed801/802 ×16prompts, 이후 original residual 2회 추가.
  일부 구간이 GPU2 분석과 겹쳐 **탐색 screen**이다. `LIVE.md` 참조.
- `live_clean/`: 동일T의 원래 proxy 배분 vs 개선 proxy 배분, seed903/904, 같은16prompts.
  순서를 뒤집어 반복했고 GPU2가 비어 있으며 분석 작업 종료 뒤에 시작했음을 기록했다.
  최종 수치는 `LIVE_CLEAN.md`, `live_clean_summary.json`에 있다.
- 단독 실행 최종 결과: 실제 출력 TPS seed903 +4.72%, seed904 −7.81%, 합산 −1.61%.
  Cache hit 합산은 +0.55%p였지만 TPS는 높아지지 않았다. **속도 개선 확정/전체
  Mirror-SD 승리로 서술할 근거가 없다.** Coverage 개선 방향과 시스템 검증을 구별한다.
- 실제 출력은 각 실행4096개. 엔진의 decode-token counter에는 출력 한도에서 버린
  끝 토큰이 포함될 수 있어, 주요 시간 지표는 **실제 출력4096 / generate wall time**이다.
- 작은 2-seed screen이며 full Mirror-SD 구현 비교가 아니다. 토큰/step을 독립적인
  속도 반복 실험으로 간주하여 유의성을 부풀리지 않는다.

## 코드와 산출물

원시 새 실행은
`ssd/experiments/proxy_source_ablation/probe_training_free_20260912/`.
`datasets.json`에 dataset별 source row/text hash가 있다. 하나의 mixed Alpaca 파일에
프롬프트를 interleave했지만, 통계는 원래 dataset별로 다시 분리한다.
`--prompt_offset`은 기존 bench에서 실제 적용되지 않아 사용하지 않았다.

- replay.py / aggregate.py: 동결 정책과 독립 결과.
- diagnostics.py: 6개 주 비교 Bonferroni, 동일 배분의 source ablation, 교환 오차.
- budget_checks.py: 동결 규칙 B=5/10/15/25/40 민감도, 새 정책 선택 없음.
- runtime_snapshot_checks.py: 모든 확인 snapshot의 actual-wire 민감도. 예측 점수의
  최적성은 통과했지만 동점 token-ID/true-coverage 일치를 주장하지 않는다.
- audit.py: dataset/원시 snapshot/개발 freeze/엔진 source hash/실제 실행 기록 검사.
- replay_v1/v2 및 development_v1/v2: 확인 전 탐색 과정 보존용. Interval no-op의
  tie 보존을 고친 최종 replay만 독립 확인에 사용했다.

후속 우선순위는 deterministic tie 명세, DS가 켜진 전체 DUET에서 배분 변경 검증,
root 준비 완료 시각/재사용 suffix/실제 latency 목적함수의 연결이다. 추가 학습이
필요하다는 결론은 내리지 않았다. 원래 residual의 우위나 Mirror-SD 전체 TPS 승리를
이번 후보 coverage 결과만으로 주장하지 않는다.
