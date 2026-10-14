# 남은 시스템 최적화 검토: ver0.2 이후

최종 판정: **기존 채택 최적화는 유지하고, B>1 tree의 중복 proposal 복사 제거를 작은 공통 최적화로 추가한다.** 25 suite/265개 검사와 원격 full 모델480문항×2회 대조를 마쳤다. 두 모델 모두 출력960/960 및 모든 step의 verification event가 동일하다. 새 알고리즘이나 TPS 향상 기법을 추가한 것은 아니다. 성능 기준은 사용자 지시대로 해당 구성요소의 latency이며, 전체 TPS 감소도 아래 공개한다.

## 배포와 비교 기준

- 2026-10-14 `MLSys-ver0.2@0d5374289b8112db53c8f82ea7eafadc0c30a696`을 `origin/MLSys-ver0.2`에 push했다. 로컬/원격 ref 일치를 확인했다.
- 그 커밋에는 이전 보고서의 261개 검사와 실모델6개 기능 검사, 승인된 모델/B>1 확장, 확률 압축/입력 준비/prewarm, 상세 인수인계가 포함되어 있다.
- 이 폴더는 push 후 추가 검토다. 기존 release의 원시 데이터·manifest·검사 기록을 새 결과로 덮어쓰지 않는다.
- 계보: 논문 `a82f7d2` → 공통 수정 `f12d0f1` → 승인 extension+후속 최적화 `0d53742` → 이번 target 입력 준비 변경. 이번 A/B는 **마지막 한 변경의 추가 효과**다. 원 논문과의 전체 개선률이 아니다.

## 이미 적용한 개선은 의미가 있는가

이 표는 [기존 release 보고서](../REPORT.md)와 [이전 matched 실험](../../mlsys_ver01/tree_followup_20261014/)을 재검토한 결과다. 이번에 다시 측정한 수치와 구별한다.

| 변경 | 직접 효과와 측정 범위 | 판정 |
|---|---|---|
| Draft의 NumPy 입력 준비 | 3090/B8, 약1.69–1.80→1.30–1.40ms/호출. P1/P2 모두 사용. 고정 buffer 값 비교 및 phase별 대조 있음 | 유지. 후보/위치 점수와 독립적인 준비 비용 감소 |
| Proxy·수락 확률 행 압축 | 4090/B8/V128256, N4/C2에서2.561→1.878ms, N8/C3에서5.951→3.442ms. 두 logits gather와 host 준비 포함 | 유지. Target attention shape는 그대로이며, vocabulary 연산할 context 수만 감소 |
| 위 개선의 통합 실행 | L2/B8 matched 전체480×2: verify22.365→21.718ms(−2.89%), 출력960/960 동일 | 해당 조건에서 통합 이득 확인. L3/4090/C2는 verify CI가0을 포함하므로 보편적인 전체 이득으로 확대하지 않음 |
| Graph 사전 준비 | 이전 full 및 release smoke에서 관측된 decode 중 capture step0 | 유지. Startup·메모리 비용은 존재하고 모든 향후 설정의 capture0을 보장하지 않음 |
| 새 모델/B>1 실행기 | 요청별 KV/예산/ordered proposal·T0·요청 종료 경계 검증 | 지원·정확성의 근거. 기능 추가 자체를 특정 latency 개선률로 표시하지 않음 |

## 이번 추가 변경

기존 B>1 tree verification은 받은 `[B,W,V]` draft logits를 새 `[B,N,V]` tensor에 padding 복사하고, proxy의 고정 buffer에 다시 복사했다. 새 경로는 proxy가 이미 갖고 있는 padding/gather 기능에 받은 payload를 직접 넘긴다.

- **변경 위치**: `ssd/ssd/engine/helpers/batch_tree_verify.py`. Target가 draft 결과를 받은 뒤 proxy 계산을 준비하는 부분이다.
- **동작 유지**: target forward shape, 실제 proposal q 값, root/위치/tree 선정 수식, 난수 소비, KV commit, P1/P2는 동일하다. B1 전용 경로는 바꾸지 않는다.
- **Dense**: proxy buffer를0으로 초기화하고 실제 W행만 복사하면, 먼저 N행으로 padding한 것을 복사한 기존 값과 정확히 같다.
- **Compact**: topology가 참조하는 실제 node 행은 W 범위 안이다. 같은 행을 gather하므로 중간 padding이 없어도 값이 같다.
- **경계**: W=0 또는 token/logit 폭 불일치는 기존 padding 경로를 유지한다. Eager dense 수락은 N행인 proxy 고정 buffer를 사용해 짧은 W와 verify 폭N이 다른 경우도 처리한다.
- 별도 옵션을 추가하지 않는다. 유효한 입력에 기본 적용하며, 수락/선정 정책은 바꾸지 않는다.
- BF16 B8/N8/V128256에서는 중간 q tensor만 약15.7MiB다. 이것을 매번 만들지 않게 되지만 allocator가 메모리를 보관할 수 있으므로 실제 상주 VRAM이 그만큼 줄었다고 주장하지 않는다.

### 기능·성능 검사

- 실제 verifier 경로: graph/eager × T0/T0.7, 긴→짧은→빈→불일치→긴 payload, dense/compact/부분 batch, FP32 q/BF16 p. 기존 padding 계약과 출력·proxy wire·RNG·KV commit bookkeeping 동일.
- 최종 전체 회귀: `checks/CHECKS.json` 25 suite/265개, 실패·오류·skip0. `CHECKS_PROVENANCE.json`은 runtime/test/bench SHA가 검사 전후 같은지 확인한다.
- Component: V32000/128256, B2/B8/부분B3, verify 폭4/8, 입력 폭4, dense/compact 12조건. 각 조건 seed10회 buffer/proxy/수락/RNG exact 대조. 준비 구간과 proxy+accept 전체를 AB/BA 교대로 측정한다.
- 첫 측정: 탐색적 component, 별도 연속 telemetry 없음. GPU 사용 스냅샷만 남았으므로 무간섭이 연속 입증됐다고 쓰지 않는다.
- 두 번째 측정: 실행 중 GPU6에 타 사용자 PID2792425가 들어온 것이 telemetry에 기록됐다. 수치와 raw는 보존하지만 **성능 근거에서 제외**한다. 같은 이유로 평균을 첫 실행과 합치지 않는다.
- 세 번째 component는 GPU4에서 외부 process 관측0. 전체12조건의 중앙값은 개선됐지만, 일부 조건의 평균 CI에는0이 포함됐다. 대표 B8/N4/C3/V128256의 준비 시간 중앙값은 **0.3973→0.3594ms**, 준비+proxy+accept는 **1.7227→1.6848ms**다. 이 조건의 후처리 평균 변화95% CI는 **[−0.0416, −0.0334]ms**다. 조건별 raw와 CI를 모두 보존한다.
- Component는 고정 synthetic logits/topology이며, compact 사례는 절반 요청의 실제 proposal 길이가0인 경계 형태다. 이 절대 시간을 실제 serving의 평균 구성비로 간주하지 않는다. 실제 모델/양수 miss proposal은 아래 full 대조로 별도 확인했다.
- 로컬 L2 최초 candidate의 초기화에 외부 GPU5 PID2794734가 들어왔다. 이 run은 완료했지만 성능 근거에서 제외했다. 출력618/960만 일치했고 첫 event 차이는 measured_1/index128/step4의 수락 길이였다. 외부 간섭이 유일한 원인이라고 확정하지 않는다. 원시 기록과 제외 사유를 남기고 새 무간섭 대조를 수행했다.
- 로컬 GPU5–7이 다른 작업에 점유되어 이후 full 검증은 eslab18(nova)의 GPU4–7에서 수행했다. 기존 작업 폴더를 수정하지 않고 별도 source snapshot 두 개를 만들었다. 원격 기준 runtime79개가 실제 `0d53742`와 byte-for-byte 동일하고, candidate는 이번 한 파일만 다르다. 원격 snapshot 자체 commit은 `fa76960`이며 실제 원본 commit으로 위장하지 않는다.
- Snapshot에는 `.gitignore`가 없어 생성된 `__pycache__`가 raw git status에 표시된다. 기준 runtime의 tracked 수정은 없으며, 모든 runtime SHA는 실행 전후 로컬 기준/검사된 수정본과 일치했다. Snapshot git ref를 다른 서버에 merge할 필요는 없고 원래 `MLSys-ver0.2`를 사용한다.

### 실모델 전체 대조 — nova / RTX3090

각 모델480문항의 첫 턴×2 pass, 960출력/arm. 같은 모델 A/B는 같은 GPU pair를 사용했다. 입력 cap512/출력 cap64, max context2048, T=0.7, warm48, prefix cache OFF, target seed10143/draft seed231. Cap/clip·부분B·graph 준비 step을 분자/분모에서 함께 제외한다. 반복 pass는 독립 문항으로 bootstrap하지 않고 동일한8문항 group을 하나의 cluster로 묶었다(60 clusters).

| 모델 | B | AL 전→후 | Cache hit 전→후 | Verify ms 전→후 | TPS 전→후 | 출력·step event |
|---|---:|---|---|---|---|---|
| Llama2-7B + AMD135M | 8 | 1.9741→1.9741 | 72.877%→72.877% | 31.5675→31.5582 | 410.94→410.03 | 960/960 및 모든 event 동일 |
| Llama3-8B + Qwama0.5B | 8 | 2.1543→2.1543 | 65.957%→65.957% | 35.0004→34.9503 | 376.13→375.15 | 960/960 및 모든 event 동일 |

- L2 verify −0.0295%, 95% CI [−0.1669%, +0.1221%]: 전체 verify 이득은 미확정. TPS −0.220%, CI [−0.574%, +0.117%].
- L3 verify −0.1433%(−0.0502ms), CI [−0.2412%, −0.0547%]. TPS −0.262%, CI [−0.523%, −0.0117%]. 해당 paired run에서는 전체 TPS가 소폭 낮아졌다. 이를 향상/중립이라고 바꿔 쓰지 않는다.
- 두 모델의 비-verify step 구간은 각각 약0.0940ms, 0.1704ms 증가했다. 이번 코드가 이 구간을 직접 수정한 것은 아니지만, 원인을 환경 변동 또는 overlap 변경 중 하나로 확정하지 않았다. 작은 component 이득으로 전체 TPS 이득을 보장할 수 없다.
- 이 CI는 해당 A/B 내 문항 group 변동이며, 여러 날짜/프로세스 A/B의 run-to-run 변동까지 포함하지 않는다. 네 run에서 관측된 외부 GPU process0, 최고온도65°C 이하, decode 중 graph 준비 영향 step0이었다.
- **채택 이유는 작은 준비 비용 감소·불필요 임시 tensor 제거·동일 계산 계약이다.** 전체 TPS 우위나 새 핵심 contribution의 근거로 채택한 것이 아니다. 최종 알고리즘/하드웨어의 최종 TPS는 다시 평가해야 한다.

| 모델 | K1/K2 | Exit | P1 생성/검증 | P2 생성/검증 | Root 수 | 형제 폭 |
|---|---|---:|---|---|---:|---:|
| L2 | 3/2 | 19 | 6/6 | 4/4 | 2 | 3 |
| L3 | 2/2 | 17 | 4/4 | 4/4 | 2 | 2 |

이 파라미터는 기존 timing/calibration에서 검토한 고정 설정이다. 새 root/위치/tree 점수에서도 최적이라고 주장하지 않는다. 두 모델 모두 target GPU1장 + draft GPU1장이다. 이번 표는 논문70B/Blackwell 성능 재현 또는 SSD와의 최종 비교가 아니다.

## 보류 항목은 왜 유지하는가

| 항목 | 현재 근거 | 이번 판단 |
|---|---|---|
| Full target packed tree | Query 수 감소 외에도 attention 계획·mask·GEMM shape가 달라짐. 모델/shape별 이득이 다르고 logits 차이도 있었음 | 확률 후처리 압축만 기본 유지. Full target 변경의 효과/수치 검증을 이번 작은 복사 제거 결과로 대신하지 않음 |
| Mixed-miss AR | Full480×2에서 k2=2/threshold1은 TPS 약+1.04%이나 CI가0 포함, k2=4/threshold2는 약−0.15%. AL 손실 및 다른 GPU의 퇴행도 확인 | 기본 적용하지 않음. 새 알고리즘/시간창으로 AL과 miss 비용이 바뀌면 재평가 |
| Replica+별도 stream 자동ON | TP2 고정 입력에서 target+proxy 감소 사례는 있으나, L3/4090/B8 full은 verify CI가0 포함. Source/출력 변화 원인도 미분리 | 명시 기능 유지, 자동ON 보류. 논문 Blackwell/70B 환경에서 기법이 무의미하다는 뜻이 아님 |
| 남은 CPU topology/metadata 처리 | Topology 검증, request별 KV commit, 일부 metadata D2H/CPU 분기 남음 | 성능 근거 없이 제거하지 않음. 특히 검증 제거·tree 정책 변경과 단순 중복 복사를 구별 |
| Miss DS/P1 tree, root/위치/tree AL 점수 | 다른 서버에서 연구 알고리즘 작업 중 | 이번 runtime에서 변경하지 않음 |

Mixed-miss의 판단은 단순 miss 개수만으로 고정할 수 없다. 같은 상태에서 JIT 사용 시 기대 출력 A·시간L, AR 혼합으로 잃는 출력 ΔA·아끼는 시간ΔL라면, 양수 분모에서 혼합이 유리한 조건은 `(A−ΔA)/(L−ΔL)>A/L`, 즉 `ΔL/L>ΔA/A`다. 최종 알고리즘·k2·tree budget이 정해지면 이 두 상대량을 다시 측정해야 한다. 관측되지 않은 counterfactual AL을 현재 raw에서 정확히 아는 것으로 취급하지 않는다.

## 다른 서버 알고리즘과의 결합

1. 이번 payload 변경은 같은 node-local q를 다른 중간 buffer 없이 전달한다. 새 root/위치 점수를 제약하지 않는다.
2. 새 후보식을 합칠 때 `batch_tree_sampling.candidates`와 `flat_tree_sampling.candidates`를 함께 검토한다. Dense만 변경하면 compact 사용 여부에 따라 다른 알고리즘이 실행될 수 있다.
3. Epoch2의 node-local q, G>M 생성 순서 prefix, request별 topology/KV 소유 관계, sampler의 실제 proposal 법칙은 유지한다.
4. Frontier 등으로 node 수·분기 폭·phase 비용이 달라지면 graph 사전 준비 범위, compact 이용 비율, P1/P2 시간창을 재검증한다. 현재 최적 파라미터가 모든 새 알고리즘의 최적값이라는 주장은 하지 않는다.
5. 70B/G>M AL 영향·기존 tree 반복−2.08% 원인·main 서버27개 작업 완료는 이 검토로 완료 처리하지 않는다.

## 반영·재현·자료

- 코드 commit: `99dec94` (`perf(tree): avoid redundant proposal padding copies during batched verification`). Runtime 변경1파일, 회귀4개 추가, 독립 component 측정기 및 전체 suite 목록을 함께 저장했다.
- 측정은 commit 전에 수행했다. 원시 JSON의 당시 commit/dirty 값을 보존하며, 최종 commit의 모든 runtime/test/bench 파일이 검사된 SHA와 일치하는지 별도로 확인했다. 원격 실행은 위 snapshot 식별자를 사용한다.
- 상세 숫자/조건/CI: `RESULTS.json`, `repeat1.json`·`repeat2.json`·`repeat3.json`. `repeat2`와 로컬 최초 online 비교는 간섭 때문에 제외했다.
- 전체 회귀: `checks/`. 원격 실행 전후 source: `REMOTE_SOURCE_HASHES.json`, `REMOTE_AFTER.json`. 모델 config 정보: `MODELS.json`.
- 원시 기록/로그/telemetry/capture audit: `RAW_REVIEW.tar.gz`. 테스트된 소스: `TESTED_SOURCE.tar.gz`. 각 파일 해시: `MANIFEST.json`. 과거 release의 manifest는 수정하지 않았다.
- 두 원격 모델 pair의 full 실험은 모두 종료했다. 다른 사용자의 GPU 프로세스를 종료하지 않았다. 로컬 GPU 점유로 시작하지 못한 retry 계획도 보존하되 완료 실험으로 세지 않는다.

재검증 예시(먼저 선택한 GPU가 비어 있는지 확인):

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
  PYTHONPATH=ssd OMP_NUM_THREADS=2 /path/to/python -B \
  results/mlsys_ver01/extensions/check_suites.py --output-dir /new/checks

CUDA_VISIBLE_DEVICES=<free_gpu> /path/to/python -B \
  ssd/bench/mlsys_tree_payload_review.py --output /new/component.json --repeats 120
```

Full 대조는 `nova_l2_plan.json`/`nova_l3_plan.json`을 복사하고 해당 서버의 모델·checkout·CPU/cache 경로를 수정한 별도 plan으로 `mlsys_tree_campaign.py`를 실행한다. 옛 raw를 덮어쓰지 않는다. `analyze.py`의 raw 경로도 필요에 따라 변경한다. 검증할 새 알고리즘을 합친 뒤에는 이 commit의 검사 통과 상태를 그대로 재사용하지 않는다.
