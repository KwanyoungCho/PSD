# MLSys-ver0.2 반영 내역·검증·서버 간 인수인계

기준: 논문 `feat/duet-p2tree-g0@a82f7d2` → 이미 배포한 `MLSys-ver0.1@f12d0f1` → 이번 `MLSys-ver0.2`. 실제 작업 디렉터리는 기존 `/home/chokwans99/PSD-MLSys-ver0.1`을 유지한다. 브랜치 이름과 디렉터리 이름은 다르다. 원본 논문 브랜치와 `MLSys-ver0.1` ref는 변경하지 않는다.

이 문서는 ver0.2의 배포 판단 기준이다. [ver0.1 통합 기록](../mlsys_ver01/REPORT.md)의 과거 “미커밋/미배포” 상태, 자동 replica 채택 주장, 당시 회귀 개수는 각각 기록 시점의 상태다. 과거 source/raw/manifest는 그 실험의 증거로 보존하고 현재 코드의 실행 결과로 바꿔 표시하지 않는다.

## 1. 이번에 들어가는 것

| 묶음 | 기존 상태·문제 | ver0.2 반영 | 알고리즘/정확성/실행 비용 구분 |
|---|---|---|---|
| 새 모델 정식 지원 | AMD dtype·Llama3 token mapping·넓은 Qwama tree workspace 제약 | Llama2-7B/AMD135M, Llama3-8B/Qwama0.5B native 실행 지원 | 모델 호환성·메모리/실행 경로. 두 조합의 최적 파라미터 탐색 완료를 뜻하지 않음 |
| B>1 tree 실행 | 요청마다 길이·tree·KV·hit source가 다름 | 요청별 cache/예산/KV를 보존한 P1/P2, tree attention, 수락, commit, shrink/refill | 묶음 실행 지원. 요청 간 attention이나 KV를 섞지 않음 |
| Draft 처리 간소화 | 여러 작은 CPU tensor 생성·topology 전송/계산 | NumPy 입력 준비, bulk metadata, mask/fanout/slot 결합 kernel·buffer 재사용 | P1/P2 실행 준비 비용 감소. Tree 선정 점수는 기존 q-path 유지 |
| Proxy·수락 확률 연산 압축 | Padding context에도 vocabulary 전체 계산 | 실제 context의 exit/final logits를 gather하고 proxy·수락 연산만 압축 | Target GEMM/attention shape는 유지. Full-B이고 압축 이득이 있는 shape에 적용 |
| Graph 사전 준비 | 새 shape 준비가 decode 중 노출 | 구성에 필요한 유한 batch/width/page 버킷 사전 준비, 공유 buffer, RNG 보존 | Startup/메모리 비용은 존재. 모든 미래 설정의 capture0을 일반 보장하지 않음 |
| 명시 replica/stream의 B>1 연결 | 논문 B1 overlap을 새 묶음 실행기에 연결할 필요 | 명시한 replica에서는 별도 proxy stream 사용 가능, 실제 선택값/출처 로그 | GPU2장에 target을 분할한다는 뜻이 TP2. 자동ON은 채택하지 않음 |
| 최종 리뷰 오류 수정 | Split이 아닌 forward에서 proxy 설정을 불필요하게 참조; 테스트 중복/미발견 | 분기/속성 guard, package/discover import 정리, function-style 검사를 unittest에 연결 | 동작 경계/회귀 신뢰성 |
| B1 형제 수 검증 | Target와 verifier가 `sibling_capacity=3`으로 고정 | 양쪽 모두 `duet_tree_c_tensor` 적용 | C3 결과는 유지. C4 합법 tree 허용, C2에서 불법 세 번째 형제 거부. Tree 생성 알고리즘 변경 아님 |
| 무디코딩 보고 | 정상 보호 종료와 오류 구분 부족; 빈 모집단 나눗셈 | Benchmark에서 명시 종료 사유·요청별 기록·엄격 분류·`None`/유효 분모·공통 문항 집계 | 후처리/측정 도구 수정. Scheduler 문맥 보호/문맥 상한은 그대로 |

연구 수식 `e(1-q)`, overlap 기반 위치 선정, reach/gain/frontier tree는 이번 runtime에 이식하지 않았다. 다른 서버에서 진행 중인 알고리즘을 임의로 합치지 않는다. B1에서 GPU tree walk로 기본 수락기를 교체하지도 않았다.

## 2. ver0.1에서 이미 상속한 것

- Chain 수락의 불필요 동기화/CPU 처리 제거, B1 tree CPU topology·온도 재사용.
- T0 tree argmax 검증, 온도 전용 sampler, ragged greedy recovery 수정, eager T0 fallback, 온도 판별 sync 제거.
- 후보 점수에 실제 온도 반영(P01), G>M에서 생성 순서 prefix로 proposal 법칙 보존(P02).
- 실제 반환 토큰 및 cap/clip·부분 batch 제외 집계(P03/P04), prefill 포함 resident B 상한(P05), int64 크기 유지 token 범위 확장(P06).
- Cached prefill page/실제 KV 길이, 완전 cached 입력의 마지막 query, preemption 출력 예산·경계, KV block hash 수정.

P01은 사용자 승인 정책 변경이며 자동적인 성능 향상이나 lossless의 필수조건이 아니다. P02는 정확성 수정이므로 옛 rerank의 AL이 더 높아도 성능 때문에 되돌리지 않는다. P01을 재평가할 때의 T1 복원 범위는 ver0.1 보고서의 `proxy_temperature.probabilities` 설명을 따른다. 생성 proposal과 target verification의 온도는 따로 유지한다.

## 3. 리뷰 지적별 처리

| 지적 | 처리·판정 | 근거/후속 |
|---|---|---|
| 234회가 최종 runtime 전체 검사로 오해됨 | 최종 source로 전체 검사 재실행. 검사0건·skip도 성공으로 처리하지 않음 | 이 폴더 `checks/CHECKS.json`, `CHECKS_PROVENANCE.json`. 과거248과 이번 결과를 구별 |
| `duet_exit_replica` 없는 fixture에서 실패 | `split` proxy 호출에서만 확인하고 `getattr(..., False)` 사용 | 기존 regression과 forward 경계 검사 |
| Import로 기본 unittest 중복 발견 | 모듈 import로 바꾸고 package/discover 경로 모두 지원 | `test_tree_extension_review` |
| 후속4개 suite 누락·CHECKS가 원격에만 있음 | 전체 runner inventory에 추가하고 로컬 로그/결과 포함 | Flat review의 기존 function 검사도 실행되게 수정 |
| TP2 replica 자동ON 근거 부족 | 전체480×2 실험 후 자동ON 보류. 기존 명시 경로 유지·시작 로그 추가 | ver0.1 `precommit_review_20261014/TP2_FULL_RESULTS.json` |
| 단독/통합 캠페인 날짜·TPS 비교 | 같은 날이지만 GPU UUID 쌍이 다름을 정정. 캠페인 내부 matched A/B 우선 | 독립 component latency와 전체 TPS를 구별 |
| Commit이 한 덩어리 | 승인 extension → 후속 production → 검토 도구/테스트 → 자료 순서로 구분 | 최종 commit 목록은 아래 배포 기록 |
| B1 `sibling_capacity=3` | Target와 verifier 두 경계 모두 설정값으로 수정 | C2/C3/C4 경계 CPU 대조 및 C4 실모델 검사 |
| Main 서버의 0초 오분류 | 아래 엄격 분류 계약을 로컬 benchmark schema에 구현 | 원격 report.py/manifest/launcher를 덮어쓰지 않음 |
| G>M의 70B AL 영향·P01 성능 | 실험 의미와 rollback 범위를 문서화; main 서버 실험과 구분 | 미완료 실험을 코드 검증 완료로 대체하지 않음 |

논문의 full-head 복제·별도 stream 기법을 무효화한 것은 아니다. 고정 입력에서 구간 지연 감소가 있었지만, 새 B8/Llama3/4090 전체 실행에서는 통계적으로 확실한 개선과 동일 실행 경로를 확보하지 못했다. 자동 기본값 결정과 기존 논문 환경에서의 기법 유효성을 구분한다. 과거 문서의 “원 논문과 같이 replica OFF”는 **환경변수 미지정 시 코드 기본값**이며 논문 실험에서 OFF였다는 뜻이 아니다.

## 4. 무디코딩 종료와 집계 계약

`mlsys_tree_followup.py`는 scheduler가 명시적으로 `is_prefill=None`을 반환한 경우에만 `context_limit_guard`를 기록한다. 빈 출력이나 짧은 시간에서 종료 이유를 추정하지 않는다. 요청 ID·실제 출력·decode 이벤트를 함께 보존한다.

`mlsys_request_reporting.py`는 다음 조건을 모두 확인한다: 명시 보호 종료, 출력 개수0·ID 목록 빈 값, decode token/verification0 또는 없음, decode step 없음, 종료 문맥=입력 문맥≤상한, 유한한 정확한0초. 다른0초·음수·NaN·무한대·서로 모순되는 출력/step/time은 오류다. 양수 시간에서도 step/토큰/검증 수 합이 일치해야 한다.

- 무디코딩 요청은 전체 처리 건수·raw에 남기고 해당 TPS/AL/hit/step latency는 `None`으로 표시한다.
- 전체 eligible step이0이어도 summary 생성과 bootstrap 비교는 예외 없이 `None`/유효 표본0을 반환한다. 실제 step이 있는데 시간이0인 것은 오류다.
- `request_population`에 전체 문항/턴 기록 수, 지표별 유효 문항/턴 수, 보호 종료 UID를 기록한다. 문항 내 유효 턴/반복을 먼저 평균한 뒤 문항 간 평균한다. P1/P2 조건부 AL의 분모도 따로 공개한다.
- 완료 arm의 `request_records`가 모두 있으면 `mlsys_tree_results.py`가 모든 기록된 턴/반복에서 decode한 공통 문항 집합의 표를 추가한다. UID/문항·턴·반복이 다른 arm을 조용히 비교하지 않는다. 옛 raw에 이유 기록이 없으면 `unavailable`로 표시한다.
- **기존 로컬 주 TPS 정의는 유지**한다. Cap/clip·부분 B·관측된 graph 준비 step을 분자·분모에서 함께 제외한 총 토큰/총 시간이다. 새 `request_decode_tps`는 해당 요청이 활성화된 공유 batch step 시간 기준이며 전체 serving TPS와 다르다. B>1 요청 시간을 합해 batch 실행시간으로 쓰지 않는다.
- 요청별 부록은 종료 이유와 생성 여부를 확인하기 위한 raw decode step 기준이다. Cap/clip·부분B도 포함할 수 있으며, 필터링된 주 AL/hit 표의 대체 지표가 아니다.
- 이 부록은 tree/chain/SSD의 비동기 decode 이벤트 형식에 적용한다. AR은 첫 토큰을 prefill에서 출력하고 동기 SD에는 해당 이벤트가 없어 같은 검증을 적용하지 않는다. 이 두 모드의 기존 raw·주 집계는 유지하고 부록은 `unavailable`로 명시한다. AR의 prefill 출력1/시간0을 DUET의 무디코딩 예외로 허용하는 식으로 규칙을 느슨하게 만들지 않는다.
- 로컬 실행기는 여전히 문항별 **첫 턴만** 입력한다. 집계 함수의 MT 두 턴 평균 검증과 실제 560턴 평가 완료는 다르다. 이번 첫 턴 평가를 main 서버의480문항/560턴 결과와 같다고 표시하지 않는다.
- 기존 raw·성능 archive·manifest는 그대로다. 새 기록 형식과 benchmark CPU 관측이 추가된 이번 smoke는 기능 확인이며 과거 TPS와 직접 성능 향상률을 계산하지 않는다.

문맥 보호와 임시 KV 예약 계산은 `a82f7d2`/`f12d0f1`에서 변경되지 않았다. 전달된 `1734+325>2048`은 원래 보호 조건과 일치하지만, 원격 plan의325칸을 여기서 재산출했다고 주장하지 않는다. 긴 입력에서도 생성하도록 tree 예산 축소/안전한 fallback을 구현한 것은 아니다.

## 5. 성능 근거와 주장 범위

아래는 이번 코드 검증과 분리한 **이전 matched 실험**의 수치다. 최종 release에서 모든 성능 실험을 다시 수행한 것으로 표시하지 않는다. 이전 최종 runtime 대비 이번 inference 변경은 B1의 형제 수 validation 두 곳이며, C3의 기존 동작은 동일하다.

| 변경·조건 | 직접 측정한 결과 | 해석 |
|---|---|---|
| Draft 입력 준비, 3090 B8 | 약1.69–1.80 →1.30–1.40ms/호출 | 준비 비용 약0.4ms 감소, P1/P2 양쪽 사용 |
| Proxy+수락 압축, 4090 B8 V128256 N4/C2 | 2.561→1.878ms | Exit/final gather 포함 후처리 비용 |
| 동일 압축, N8/C3 | 5.951→3.442ms | 같은 입력 대조, target attention 비용은 그대로 |
| 통합 Llama2 B8/4090 | Verify22.365→21.718ms, 약−2.89%, 출력960/960 동일 | 해당 matched A/B의 근거. TPS595.01→610.21은 보조 |
| 통합 Llama3 B8/4090/C2 | Verify24.297→24.210ms, CI에0 포함 | 전체 verify 이득 미확정, TPS 약−0.68%도 보존 |
| Full-head+stream 고정 입력 | Llama3/3090 TP2 B8 약42.62→39.99ms | 기법이 무의미하다는 결론이 아님 |
| 자동 replica/stream 전체 L3/4090 TP2 B8 | Verify19.5009→19.4664ms, CI에0 포함 | 이 환경의 일괄 자동ON은 보류. 출력/source 차이 원인도 미분리 |

원시 자료: [후속 component/full 기록](../mlsys_ver01/tree_followup_20261014/), [자동 replica 전체 대조](../mlsys_ver01/precommit_review_20261014/). Prefix OFF·출력 cap64·입력 cap512 등의 평가 조건은 각 plan을 따른다. 논문70B/Blackwell의 절대 TPS 재현값이 아니다.

## 6. 기본 채택에서 제외한 것

- Full target packed tree: 모델/shape별 지연·logits 차이. 실험 helper는 재현용으로 분리하고 production default에서 호출하지 않는다.
- Mixed-miss AR: 전체 평가에서 일관된 TPS 이득 미확인. 기본 miss JIT chain 유지.
- TP1 stream 일괄 활성화 및 TP2 replica 자동ON: 조건별 근거 부족. 명시 기능은 유지.
- B1 GPU tree walk로 수락기 교체, 실제 sibling 폭마다 graph family 추가: 추가 이득 미확정.
- Miss DS/P1 tree, 새 root/위치/tree 연구 알고리즘: 별도 검토/다른 서버 작업.

## 7. 최종 검증과 배포 기록

**최종24개 suite/261개 검사 통과, 실패0·오류0·skip0.** GPU7의 CUDA 회귀를 포함한다. 실행 전후 runtime/test/bench SHA256이 동일했고 실모델6개 실행도 같은 최종 source였다. 검사 후 commit한 코드도 파일별 hash와 일치함을 확인했다. 앞선260개 검사와5개 실모델 결과는 `initial_validation/`에 원래 source/raw/manifest hash와 함께 보존했다. 과거234/248/25개 결과를 재사용한 숫자가 아니다. 최종 로그는 `checks/`와 `CHECK_LOGS.tar.gz`, 요약은 [VALIDATION.json](VALIDATION.json), source/artifact hash는 [MANIFEST.json](MANIFEST.json)에 있다.

논문 원본 clean checkout과의 CPU 대조도 다시 수행했다: **chain384, tree288, sampler12, q_probs12** 모두 출력·recovery 분포·RNG 상태 일치. 이는 고정 tensor의 계약 검사이며 모든 모델 token열이 AR과 bitwise 동일하다는 보장은 아니다. `PAPER_PARITY.json`과 `paper_parity.log`에 별도 기록한다.

| 최종 실모델 기능 검사 | 결과 | 범위 |
|---|---|---|
| Llama2-7B/AMD135M, B1/C4/T0.7 | 16문항 생성 완료 | 기존 형제 수3 고정이 있던 B1 경로 |
| 같은 모델, B1/C4/T0 | 16문항 생성 완료 | Greedy tree |
| Llama3-8B/Qwama0.5B, B8/C2/T0.7 | 16문항 생성 완료 | 새 모델·묶음 tree·요청별 집계 |
| 같은 모델, B8/C2/T0 | 16문항, 출력326토큰 | 첫 B8 step들이 EOS clipping, 이후 부분 batch라 필터링된 성능 표본0. 주 TPS/AL/hit `None`, 요청별 생성 지표 유효16문항. 무디코딩 종료와 구별 |
| Llama2 긴 입력2040, max context2048 | 명시 context guard, 출력0·decode time0·verify0 | 요청 기록1개 보존, 주/요청별 지표 `None`. 실제 모델로 보호 종료 후처리 확인 |
| Llama2 AR 기준선 | 16문항, 출력256토큰=decode240+prefill16 | 기존 집계 완료. 요청별 DUET 부록은 `unavailable`로 표시해 잘못된 검증을 피함 |

모든 실행 종료 코드0, 선택 GPU4/5에 외부 process 관측0, decode 중 새 graph 준비 영향 step0. GPU0/1 등 다른 사용자의 작업은 종료하지 않았다. 이 소규모 실행들은 성능 우위/최적 파라미터의 증거가 아니다. 구체적인 모델/온도/seed/K1/K2/exit/node 예산은 [SMOKE_PLAN.json](SMOKE_PLAN.json), 모든 raw/로그/telemetry는 `RAW_SMOKES.tar.gz`와 `RAW_MANIFEST.json`에 있다. Runtime·bench·tests는 `SOURCE.tar.gz`에 함께 보존했다.

전체 회귀 재현(현재 서버의 경로/CPU affinity 예시):

```bash
CUDA_VISIBLE_DEVICES=7 SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
  PYTHONPATH=ssd OMP_NUM_THREADS=2 taskset -c 24-27,56-59 \
  /home/chokwans99/PSD/ssd/.venv/bin/python -B \
  results/mlsys_ver01/extensions/check_suites.py \
  --output-dir /tmp/mlsys-ver02-recheck
```

실모델 재현은 비어 있는 GPU pair를 확인한 후 `ssd/bench/mlsys_tree_campaign.py --plan results/mlsys_ver02/SMOKE_PLAN.json --directory <새 결과 경로> --gpus <target,draft> --port <빈 포트>`를 사용한다. 먼저 plan의 모델·checkout·CPU 경로를 해당 서버에 맞춘 별도 사본으로 바꾼다. 옛 raw/manifest는 덮어쓰지 않는다. Source를 변경한 경우 이 문서의 검사 통과 상태를 새 코드에 그대로 적용하지 않는다.

Commit은 아래 순서로 분리했다. 파일별 목록은 [COMMITS.json](COMMITS.json)에 있다. 문서/자료 commit은 `git log -1 MLSys-ver0.2`로 확인한다. 원본 논문 및 `MLSys-ver0.1` ref는 유지하며 push하지 않았다.

| 순서 | Commit | 검토할 범위 |
|---|---|---|
| 1 | `9a3e596` | 승인 extension, 새 모델과 B>1 tree runtime22개 파일 |
| 2 | `703b3a7` | 후속 production 최적화·리뷰 수정10개 파일 |
| 3 | `cc79f65` | 검토 adapter·집계·회귀·전체 runner38개 파일 |
| 4 | 이 문서를 포함하는 commit | 과거 전체 실험 증거 및 ver0.2의 최종 검사·인수인계 |

## 8. 다른 서버에서 검토·병합하는 순서

1. 현재 작업을 보존하고 기준 커밋이 `f12d0f1`의 수정들을 이미 포함하는지 확인한다. 이 브랜치는 원본 논문에서 직접 새로 시작한 것이 아니라 ver0.1의 후속이다.
   Sparse checkout을 쓰는 서버는 결과 문서 경로가 숨겨질 수 있다. 현재 worktree는 `ssd/bench`, `ssd/ssd`, `ssd/tests`만 sparse 대상으로 두고 있어 결과 자료를 `git add --sparse`로 명시적으로 커밋한다. 다른 서버에서는 sparse 설정을 확인하거나 `git show MLSys-ver0.2:results/mlsys_ver02/REPORT.md`로 문서를 바로 읽을 수 있다.
2. 승인 extension과 후속 production 커밋을 순서대로 검토한다. Runtime만 필요한 서버는 실험 adapter를 production 실행에 연결하지 않는다.
3. 다른 서버의 root/위치 연구 점수를 합칠 때 `batch_tree_sampling.candidates`와 `flat_tree_sampling.candidates` 둘 다 갱신한다. G>M prefix와 실제 proposal q-reference 보존은 유지한다.
4. 다른 서버의 `results/blackwell_validation/report.py`는 자체 schema/manifest가 있다. 이번 공통 분류·분모 규칙만 검토해서 이식하고, 서버 경로가 들어간 launcher/manifest를 덮어쓰지 않는다. 실행 중 source hash를 감시하는 대기열은 먼저 안전하게 정지한 상태에서 분석 파일만 변경한다.
5. 최종 알고리즘·장비에서 phase 시간창과 파라미터를 재선정하고 SSD도 별도로 튜닝해 비교한다. 70B/G>M14→12 AL 영향, 오래된 tree 반복의−2.08% 원인, main 서버27개 계획 완료는 이 release의 코드 회귀로 완료 처리하지 않는다.

재현 명령과 파일 목록·commit 식별자는 최종 검증 절 및 `MANIFEST.json`에 기록한다. 이번 요청은 새 브랜치 생성과 **commit**이며, remote push는 수행하지 않는다.
