# 동일 조건 비교의 실제 파라미터

| 항목 | 값 |
|---|---|
| Model pair 1 | LayerSkip-Llama2-7B + AMD-Llama-135m, full BF16 |
| Model pair 2 | LayerSkip-Llama3-8B + Qwama-0.5B-Instruct, full BF16 |
| GPU | RTX 4090 두 장; target 1장 + draft 1장 |
| Batch / temperature | B=8 / T=0.7, target·draft·proxy 점수 동일 온도 |
| P1/P2 forward round | K1=4 / K2=2 |
| Early exit | Config 값 21 (동일 모델 두 arm에서 같은 값) |
| P1 생성/검증 node cap | root 하나당 8/8 |
| P2 생성/검증 node cap | root 하나당 4/4 |
| 형제 최대 폭 | C=3 |
| P1 root 후보 | 문맥당 2개. 실제 응답 문맥 폭에 따라 수가 달라짐 |
| P2 root budget | 5개 (`proxy_fan_out=1`, `max(K1,K2)+1=5`) |
| P1 문맥 bucket | 3/5/9 → root canvas 6/10/18개, request별 독립 |
| Proxy 위치별 top-K / wire 수 | 자동 확장된 13 / 25개, 두 arm 동일 |
| P1 start/conf threshold | 0 / 0 |
| P2 start/conf threshold | 0.01 / 0.03 |
| Miss | 기존 JIT chain, K2=2. Mixed-miss AR 적용 안 함 |
| P1/P2 tree selector | 둘 다 on, dynamic. 점수·배분·frontier만 대조 |
| Target attention | 기존 tree attention. Target 전체 packed 변경 없음 |
| Proxy/accept | 기존 ver0.2의 flat 연산 압축, 해당 조건 동일 |
| 별도 stream/replica | 비교 harness에서 둘 다 OFF, 비동기 draft-target 구조는 유지 |
| Prefix reuse | 양 arm OFF (`PAPER_COLD_PREFIX=1`) |
| 문맥 상한 / 측정 입력 cap | 2048 / 512 tokens |
| 출력 cap | 64 tokens |
| Warmup / 평가 | 48문항 warmup; 전체 480문항 첫 턴 × 2 passes |
| Seed | target 15101 (pass마다 재설정), draft process 시작 seed 231 |

CLI의 `--roots 2`는 이 harness의 `duet_draft_fan_out` 명칭이다. **전체 root 예산이 2라는 뜻이 아니다.** 위 실제 P1/P2 budget과 구분한다. 원시 로그의 `[Config]`에는 자동 확장된 top-K, wire, phase cap이 기록돼 있다.

메모리 여유에 따른 KV block 수는 실제 모델별 실행 로그에 있다. 성능 결과는 전체 문항을 대상으로 평가했지만 입력·출력을 위 cap으로 제한했으므로 무제한 길이 serving의 결과로 일반화하지 않는다. 이 파라미터는 알고리즘 효과를 구분하려고 고정한 것이며 모델별 최적값을 새로 탐색한 결과가 아니다.

분리 비교의 실제 root 정책은 plan의 `DUET_LOCAL_REVIEW_ROOT`와 시작 로그 `[research ablation]`에 기록한다. Config의 `[DUET policy]` 문구는 운영 기본 preset 설명이므로 benchmark monkeypatch의 변경값을 표시하는 로그가 아니다. FINAL_AUDIT.json에서 plan과 실제 ablation 시작 로그를 함께 확인한다.

별도 48문항 profile에서 P1/P2 total은 CPU launch gap·준비를 포함한 CUDA event 구간이고 replay는 graph replay 구간이다. 현재 bulk export는 P1/P2의 cache metadata를 모아 P2 끝에서 처리하므로 `P2 total` 안에 두 phase의 공통 export 비용도 포함된다. Target·draft 구간을 서로 더해 직렬 실행 시간으로 쓰지 않는다.

같은 G/M cap은 실제 생성 node 수가 같다는 뜻이 아니다. Frontier가 기존에 남던 예산을 더 쓰고 깊이를 바꿀 수 있다. 또한 full 정규화는 P2 root 점수의 절대 크기도 바꾸므로 고정 start threshold=0.01과 상호작용할 수 있다. 이번 비교에서는 threshold를 재튜닝하지 않았고, 실제 hit tree의 node 수·깊이를 TABLES/RESULTS에 함께 기록한다.
