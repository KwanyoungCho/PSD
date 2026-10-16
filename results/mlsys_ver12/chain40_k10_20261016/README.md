# Chain40 / K1=K2=10 로컬 측정

사용자 지정 조건으로 native chain 경로를 측정한다. 직전 tree 진단과 같은
LayerSkip-Llama2-7B/13B + JackFram/llama-68m, 로컬 orion RTX4090만 사용한다.
7B는 target GPU4 + draft5,13B는 target4·5 + draft6이다.

- B1, BF16, T0.7, early-exit21/26(0-based), K1=K2=10.
- DS 총 root40개: K+1=11개 위치에 `[4,4,4,4,4,4,4,3,3,3,3]`을 배분한다.
  기존 설정 `duet_split_phase1_fan_out_list`를 사용하며 각 위치 draft top-k다.
- PS는 기존 후보·위치 점수로 DS와 겹치지 않는 root40개를 선택한다.
  `duet_p2_forward_width=40`, wire82개(40 + DS dedup 여유40 +2)다.
- 모든 phase의 실제 forward 입력은40행이며10회 실행한다. 후보마다 자손10개인 chain이다.
  P1/P2 tree policy는 모두 OFF이며 C와 tree 생성·검증 상한은 이 경로에 적용되지 않는다.
- Cache miss도 기존 JIT draft chain 길이10을 사용한다. target 실제 검증 폭10을 확인한다.
- 직전 진단의 사전 선택12턴×2회, 입력 상한512/출력 상한192, warmup3턴을 그대로 쓴다.
  seed 및 cold-prefix 처리도 같은 harness를 사용한다.12턴은 독립 입력 수이며24개의
  독립 문항으로 계산하지 않는다. full Spec-Bench 품질 비교가 아니다.

## 시간 측정

`phase1_replay`/`phase2_replay`는 native 코드가 매 forward에 기록하는 CUDA event다.
Graph에는 model+LM head와 출력 buffer 저장이 들어 있다. Between-forward 입력 준비,
attention plan, sampling은 이 시간에 포함하지 않는다.10개의 실측 replay 합/10이
forward1회 평균이다. Tree의 전체 rollout graph/K와 혼동하지 않는다.

가용 계산창은 `forward 합 + signed 여유`다. DS 여유의 기준은 proxy 송신 종료,
PS 여유의 기준은 다음 target 요청 시작−draft 수신 준비 완료다. 계산창은 이미 사용한
forward 시간을 포함하며 현재 나머지 작업을 고정한 값이다. 음수라면 해당 고정 비용 아래
deadline에 맞는 forward 여유가 없다는 뜻이며, 음수 실행 시간이라는 의미가 아니다.
K를 바꾸면 준비·샘플링 비용도 달라진다. Cross-GPU 시각 정렬의 한계도 유지된다.

Warmup·처음10step·terminal/cap/clip·CPU 기록상 컴파일/capture 영향 step을 제외한다.
속도가 느리다는 이유만으로 제외하지 않는다. Timeline에는 실제 측정한 각 forward를
별도 막대로 표시하고1~10번을 붙인다. 다음 target 요청의 deadline과 draft 준비 완료를
구분하며, 미래 PS 지연을 모든 response wait의 원인으로 해석하지 않는다.

## Accepted-length 그래프

각 cache source s에서 다음 확률을 계산한다.

`P(AL >= x | source=s) = 해당 source에서 AL이 x 이상인 step 수 / 해당 source 전체 step 수`

AL은 기존 보고서와 같은 **수락 draft 토큰+correction/bonus 토큰**이다.
따라서 x=1은100%, 가능한 최대 AL은11이다. 각 curve는 P1 hit, P2 hit, miss 조건부이며
P1/P2의 전체 hit 비중을 곱하지 않는다. 원시 `accepted_len`을 사용하고, 상한 때문에
실제로 반환되지 않은 토큰을 포함하는 step은 제외한다.

메인 그림은 latency 표와 같은 중간 구간을 사용한다. 초기10step과 준비 구간도 포함한
비종료·비cap 전체 decode의 민감도 결과를 JSON/CSV에 함께 보존한다.
95% pointwise bootstrap 구간은12개 UID 단위로 재표집하여 두 반복을 함께 묶는다.
그림은 첨부 예시처럼 선과 점으로 표시하고 구간은 CSV에서 확인한다.

## 재현

```bash
cd /home/chokwans99/PSD-MLSys-ver0.1
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/chain40_k10_20261016/run.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/chain40_k10_20261016/analyze.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/chain40_k10_20261016/render.py
```

원시는 `/data/chokwans99/duet-chain40-k10-20261016`에 보존한다. 실행별 COMMAND,
RESOLVED_CONFIG, SOURCE_MANIFEST, chain_shapes, profile, telemetry가 있다.
Benchmark wrapper는 기존 설정 연결과 CPU shape 확인만 추가한다. Sampling/verification과
실행 엔진 runtime은 변경하지 않는다. Python assertion은 직전 실험과 같이 켜져 있다.
`pilot_7b`는 기능 확인용이며 본 그래프와 평균에서는 제외한다.

이번에는 root 수와 K를 함께 변경했으므로 직전 tree W9/K3 결과와의 차이를
chain/tree 구조 하나의 효과로 주장할 수 없다.
