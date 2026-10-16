# Target verification 사이의 전체 공백 검토

2026-10-17, `MLSys-ver1.2`. GPU를 사용하지 않고 현재 코드와 기존 두 실험의 trace를 검토했다.
Inference runtime·파라미터는 변경하지 않았다. 새 GPU 실험이나 성능 개선량 검증 결과가 아니다.

## 먼저 구간을 바로잡기

사용자가 말한 response wait은 **target 모델 계산이 끝나고 다음 target 모델 계산이 시작될 때까지의 전체 공백**이다.
코드의 `target_recv_response_wait`는 이 공백 안에서 draft 응답을 받는 일부 구간일 뿐이다.
앞서 그 marker만 분석한 것은 사용자가 지칭한 전체 구간보다 범위가 좁았다.

이번 집계 경계는 같은 요청 안의 `final_logits(n-1).end → graph_pre(n).start`다.
Final LM head까지 끝난 뒤 수락 판정, 후처리, scheduler, 다음 handshake, 다음 forward 준비를 모두 포함한다.
각 boundary를 target GPU의 동일 시간축에서 읽고, 여섯 개의 서로 겹치지 않는 연속 구간으로 나눴다.
합이 전체 공백과 일치하는지 모든 행에서 검사했다.

## 실제로 어디서 시간을 쓰는가

아래는 **다음 step이 P1 hit인 경우**의 평균이며 단위는 ms다.
이전 step의 source는 고정하지 않았다. 전체 source·개별 step은 `SUMMARY.json`/`DETAILS.json.gz`에 있다.

| 구간 | Tree K3 7B | Tree K3 13B | Chain K10 7B | Chain K10 13B |
|---|---:|---:|---:|---:|
| 이전 결과의 수락 판정·필요한 target KV 확정 | 1.462 | 1.684 | 1.684 | 1.775 |
| 수락 후 통계·sequence 상태·postprocess | 0.351 | 0.356 | 0.330 | 0.356 |
| scheduler와 다음 요청 입력 준비 | 0.443 | 0.420 | 0.411 | 0.433 |
| 다음 요청 송신 | 0.455 | 0.480 | 0.466 | 0.483 |
| 송신 완료 후 draft 응답 수신 완료까지 | 3.159 | 3.284 | 3.842 | 2.791 |
| 응답 해석·sequence 구성·다음 verification 준비 | 2.178 | 2.933 | 1.476 | 1.465 |
| **Target forward 사이 전체 공백** | **8.048** | **9.158** | **8.209** | **7.303** |
| 표본 수 | 698 | 741 | 750 | 728 |

따라서 전체 공백은 약 7~9ms이며, draft 응답 대기 약 3ms만 줄인다고 나머지가 사라지지 않는다.
표의 각 행은 boundary 사이의 elapsed 구간이다. 예를 들어 마지막 행의 2.933ms가 전부
attention mask kernel 비용이라는 뜻은 아니다. CPU dispatch와 GPU 완료 대기를 포함한다.

실제 실행 순서:

```text
Target forward n 종료
  → accept/reject 및 recovery token 확정
  → 통계·sequence 갱신·scheduler
  → 실제 outcome으로 다음 후보 요청
  → draft가 cache 조회 및 응답 준비
  → token/q/topology 수신
  → CPU sequence 재구성·target 입력/mask 준비
  → Target forward n+1 시작
```

## 코드에서 확인한 원인

### 1. 수락 판정도 forward 사이에 직렬로 놓인다

- `verifier.py: verify()`는 target model 호출이 끝난 뒤 `verify_sample_accept`를 실행한다.
  실제 recovery token과 수락 경로를 알아야 다음 cache key를 보낼 수 있으므로 이 의존성 자체는 필요하다.
- **현재 B1 tree 경로는 `tree_verify_walk_tensor()`의 Python 제어 루프**다.
  연산은 GPU tensor로 수행하지만 `float(D[t])`, `float(R[t])`, `float(R.sum())` 등으로
  GPU 결과를 CPU로 읽어 다음 분기를 결정한다. 함수 이름의 `tensor`가 전체 보행 GPU 실행을 뜻하지 않는다.
- `tree_kv_commit`은 선택한 경로의 target KV를 확정한다. Tree에서는 노드 순서와 수락 경로 순서가 달라 필요한 작업이다.
- P1-hit 표본의 이전 tree walk span 평균은 7B 0.953ms, 13B 1.006ms다.
  이전 step 중 chain/miss도 섞여 있으므로 모든 이전 step에 tree walk가 있었다는 뜻은 아니다.
  Target KV commit span은 같은 모집단에서 0.199/0.393ms다. 이들은 수락 행 내부 구간이며 표에 다시 더하지 않는다.
- Chain K10의 이전 chain accept span도 1.408/1.485ms다. Tree를 chain으로 바꿔도 수락·recovery 처리가 없어지지 않는다.

### 2. Draft가 준비됐어도 cache hit 응답을 즉시 반환하지 않는다

- `draft_runner._service_spec_request()`는 요청을 받으면 **이전 tree 수락 경로의 draft KV 복구를 cache 조회보다 먼저** 수행한다.
- 이후 `hit_cache_and_respond()`에서 cache 조회, phase/root 판별, tree 포장·검증, parent-q gather를 한다.
  여러 GPU scalar 읽기와 D2H/H2D가 요청 처리 중에 남아 있다.
- Tree K3 P1 hit에서 이전 PS가 response-recv 구간과 겹친 시간은 7B 0.004ms, 13B 0.007ms다.
  이 경우 큰 원인은 PS의 초과가 아니다.
- 같은 표본에서 draft KV 복구 span은 0.552/0.590ms, hit tree 포장·CPU 확인은 0.570/0.575ms다.
  Parent-q 수신 span은 0.147/0.155ms다. 약 3ms 전체를 vocabulary payload 전송 비용이라고 해석하면 안 된다.
- Chain K10에서는 이전 PS 미완료가 response-recv 구간과 겹친 시간이 1.941/0.855ms로 커진다.
  이 설정은 PS도 실제로 늦다. 두 실험은 K/root/M/구조가 모두 달라 구조 하나의 A/B 비교는 아니다.

### 3. 응답을 받은 뒤에도 곧바로 Transformer를 실행하지 않는다

- `speculator_async.extend_seqs_for_verify()`는 `valid_k.tolist()`와 후보 token `tolist()`로
  GPU 데이터를 CPU sequence에 반영한다.
- `verifier.verify()`가 `valid_k.max().item()`, tree header 확인, tree wire `tolist()`를 수행한다.
  Speculator와 verifier에서 이미 아는 작은 metadata를 다시 읽는 경로가 남아 있다.
- Tree wire 해석·검증 → proxy용 topology 준비 → parent-q row 선택 → `model_runner.prepare_decode()` →
  `verify_setup` → mask/attention buffer 준비를 거친 후 `graph_pre`를 시작한다.
- 일부 mask 생성과 attention buffer 처리는 이미 GPU kernel/기존 graph 경로로 최적화되어 있다.
  이번 검토에서 그것들을 아직 구현하지 않은 것처럼 분류하지 않았다. 남은 문제는 이 경로 전체의 host 제어·metadata 재전달이다.
- Tree P1 hit에서 `verify_setup` 자체는 약 0.634/0.672ms이며,
  수신 후 forward 전 전체 2.178/2.933ms와 다르다. 이 마지막 구간에는 그 앞의 여러 처리도 포함된다.

## 줄이는 우선순위

| 우선순위 | 변경 방향 | 보존할 조건 / 검증 |
|---|---|---|
| 1 | 수신한 token·valid·phase·topology를 한 번에 CPU snapshot으로 읽고 speculator/verifier/runner에서 재사용 | 같은 proposal/token/valid 길이·RNG 소비 보존. 기존 반복 `.item()`/`.tolist()`만 제거하며 target 입력을 GPU에서 CPU로 다시 만들지 않는 경로까지 검토 |
| 2 | Cache hit 응답을 먼저 보내고, 이전 tree의 draft KV 복구를 응답 뒤·다음 glue 전에 수행 | Miss는 JIT 전에 복구해야 한다. 이전 served topology와 staged KV를 별도 snapshot으로 보존해야 하며 현재 hit의 served metadata로 덮어쓰면 안 된다. 전체 DS/PS 예산도 다시 측정 |
| 3 | G=M의 확정된 root별 tree wire/host metadata를 다음 요청 전에 준비 | 후보 선택·형제 순서·parent q를 바꾸지 않는다. 전체 root precompute가 DS/PS 시간을 더 늘릴 수 있어 중첩 및 준비 비용을 함께 검증. Double buffer 필요 |
| 4 | Tree 수락 루프의 GPU→CPU 분기 왕복을 줄이는 실행기 | 현행 stochastic residual ladder와 RNG/수락 법칙을 보존해야 한다. 실제 AL에 따라 CPU 방식보다 느릴 수도 있어 component A/B 후 채택 |
| 5 | 긴 PS가 다음 요청을 막는 설정의 시간 예산 또는 서비스 구조 조정 | K2 감소는 AL tradeoff다. P1 hit 시 PS를 건너뛰고 먼저 응답하려면 현재의 단일 loop·일괄 graph·cache/KV 수명을 바꾸어야 하므로 단순 poll/isend 변경으로 해결되지 않음 |

이 우선순위는 알고리즘을 바꾸지 않는 실행 경로 개선을 먼저 다룬다.
`dist.recv`를 `irecv`로 이름만 바꿔도 다음 forward가 필요한 token을 받기 전에는 시작할 수 없다.
GPU tensor로 계산한다는 사실만으로 Python 분기에 필요한 sync가 없어지는 것도 아니다.

**현재 확정한 것은 병목 위치와 개선 후보다. 개선량을 측정하거나 production 경로에 채택한 단계는 아니다.**
기존 `tolist()` 생략 시험은 PS 준비의 다른 경로이며 미채택 상태를 그대로 유지한다.
이번 전체 target gap 분석을 그 시험의 재현 성공으로 해석하지 않는다.

## 재현과 한계

```bash
/home/chokwans99/PSD/ssd/.venv/bin/python -B \
  results/mlsys_ver12/response_wait_review_20261017/analyze.py
```

- 현재 로컬 서버의 저장된 profile과 기존 source 분류를 읽는다. 새 모델을 로드하지 않는다.
- Tree K3는 `local_phase_latency_20261015`, chain K10은 `chain40_k10_20261016`의 기존 모집단이다.
  12입력×2회 진단이며 full Spec-Bench가 아니다. 두 모델의 draft는 모두 llama-68m다.
- Profile ON 데이터이므로 정상 성능 측정의 확정 latency/TPS로 인용하지 않는다.
  어제 추가한 2배 초과 제외를 이 진단에 소급 적용하지 않았다.
- 전체 공백 분할은 target의 같은 시간축이다. 이전 PS와의 교차 비교에는 기존 GPU-to-host 정렬 오차가 있다.
- 세부 span은 서로 포함·중첩될 수 있다. 여섯 개 boundary partition 이외를 합쳐 전체 시간으로 만들지 않는다.
- `SUMMARY.json`은 통계·출처 hash, `DETAILS.json.gz`는 개별 step과 코드 hash다.
  위 표와 관련된 모든 구간의 non-negative와 합 일치를 실행 중 검사했다.
