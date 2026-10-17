from pathlib import Path
import json
H=Path(__file__).resolve().parent
C=json.loads((H/'DIRECT_COMPARISON.json').read_text())
A=json.loads((H/'RELEASE_AUDIT.json').read_text())
R=C['tree_T07'];assert R['adopted_candidate']
lines=['# MLSys ver1.4 채택 및 최종 검증', '',
'실험 브랜치를 통째로 merge하지 않고 `MLSys-ver1.3@5642de04`에서 새 브랜치를 만들었다. 성능·출력 동일성을 확인한 topology 사전 준비, B1 tree 응답 통합, glue/요청 입력 준비 및 작은 중복 작업 제거만 이식했다. Runtime 기준 커밋은 `fa9963e56e45f73bba3aa4eb39990ebca3ec32c1`다.', '',
'## 실제 ver1.3 대비 직접 비교', '',
'로컬 RTX4090 4/5, 6/7에서 각 pair별 같은 모델·파라미터·입력·seed를 사용했다. Pair0은 ver1.4→ver1.3, pair1은 ver1.3→ver1.4 순서다. 별도 clean detached checkout에서 **커밋된 최종 runtime**을 실행했다. Profile OFF, 모델 graph 준비는 초기화에서 끝내고, step 내 사전 packet 준비와 후속 draft 비용은 전부 포함한다.', '',
'- LayerSkip-Llama2-7B + JackFram/llama-68m, full BF16, B1, T=0.7, target GPU1 + draft GPU1.',
'- K1/K2=8/5, exit=22(0-based), DS 위치당 roots=4, DS continuation width=32, PS roots/width=15/15, C=3, G=M=15/15.',
'- 12문항, 입력 최대512, 출력 최대256, 2 passes. 반복을 독립 문항으로 세지 않았다. 이 설정을 새 전역 default나 최적 preset으로 바꾸지 않았다.',
'- 양 arm prefix cold 동일. 전체 Spec-Bench/1024토큰·70B·B>1 성능 평가로 해석하지 않는다.', '',
'| Step source | 포함 step 수 | ver1.3(ms) | ver1.4(ms) | 변화(ms) |', '|---|---:|---:|---:|---:|']
for src,label in [('1','P1 hit'),('2','P2 hit'),('0','Miss'),('all','전체')]:
 d=R['sources'][src];a,b=d['step_ms'];lines.append(f'|{label}|{d["n"]}|{a:.3f}|{b:.3f}|{b-a:+.3f}|')
lines += ['',f'전체 step 감소 **{R["reduction_pct"]:.2f}%**, 문항 cluster bootstrap CI95 **{R["ci95_pct"][0]:.2f}~{R["ci95_pct"][1]:.2f}%**. 독립 wall observer도 같은 방향이다. Process 전체의 환경 변동을 이 CI가 제거하지는 않는다.', '',
'| Pair | before → after | 감소 |', '|---|---|---:|']
for p in R['process_pairs']:lines.append(f'|{p["before"]}|ver1.3 → ver1.4|{p["reduction_pct"]:.2f}%|')
lines += ['',
'모든 warm/measured request의 token ID 전체, source, hit, valid-k, accepted length가 각 arm에서 일치했다. 따라서 AL·cache hit 및 검증 입력 수 변화로 얻은 TPS 이득이 아니다. Miss만 따로 보면 반드시 빨라진다는 결과는 아니며 표의 관측값을 유지한다. 채택 근거는 반복 비교의 전체 tree step 감소와 hit 응답의 직접적인 비용 감소다.', '',
'과거 사전 연구의 48문항/56턴 관측값은 `evidence/local_fused_response_20261017/REPORT.md`에 있다. 그 비교의 base는 prepack `1dcb10ba`였으며 ver1.3 자체가 아니다. 이전 1.43%, 작은 0.04ms, 이번 비율을 더하지 않는다. 초기 통합의 동일 엔진 mode 전환에서 약0.35ms 감소가 확인된 근거도 해당 증거에 보존했다.', '',
'## 채택 파일과 기존 동작의 차이', '',
'| Runtime 파일 (`ssd/ssd/engine/` 아래) | 채택 내용 |', '|---|---|',
'|draft_runner.py|DS/PS 완료 때 G=M topology/응답 packet 준비. Hit는 사전 packet 선택 후 parent-q를 gather하여 전송. Miss classification 재사용. Glue는 CPU ancestry 배열을 일괄 복사.|',
'|helpers/response_topology.py|Device/CPU/Python integer topology snapshot. Hit root를 검증 후 사용.|',
'|helpers/fused_tree_response.py|B1 tree hit readback 통합·사전 응답 선택·CPU glue/request 배열 준비. Dummy uniform RNG 소비는 유지하고 사용하지 않는 argmax 생략.|',
'|speculator_async.py|B1 non-EAGLE request packet을 CPU에서 채워 한 번의 H2D. Wire 형식·전송 순서 유지.|',
'|helpers/p2_tree.py|정확히 전체 폭을 overwrite하는 proxy 입력의 zero 생략. Partial width는 기존 zero 유지.|',
'|model_runner.py|바로 검증한 동일 immutable tuple만 validation 재사용. TP follower/다른 객체/가변 packet은 기존 검증.|',
'|verifier.py|검증된 tuple을 한 call 동안만 공유하고 finally에서 해제. Graph에서 쓰지 않는 eager parent/sibling list는 만들지 않음.|', '',
'후보/위치 점수, draft q, target 수락·recovery 법칙, tree 알고리즘과 예산, K1/K2, exit, G=M 계약, temperature, scheduler, KV 주소/복원 의미와 통신 형식은 그대로다. 새 사용자 옵션은 없다. EAGLE·B>1·사전 tree 응답이 없는 chain/G>M cache는 기존 serving 경로를 사용한다. Request packet CPU 준비는 B1 non-EAGLE에서 적용되며 packet bytes가 같다.', '',
'최종 점검에서 tree packet이 없는 cache를 추가 GPU 조회 전에 제외하도록 eligibility guard를 보강했다. 최소 chain fixture의 누락 속성을 읽던 초기 오류2건은 `checks/batch_initial.log`에 보존했다. Fixture를 덧대지 않고 해당 경로가 helper에 들어갈 필요가 없도록 고친 뒤 batch 8개를 다시 통과했다. 실험 snapshot 대비 차이는 이 guard·설명 docstring·새 guard regression·새로 추가한 두 빈 줄의 공백 정리뿐이다.', '',
'## 채택하지 않은 것', '',
'- 모델 graph의 valid-node bucket 축소, parent-q 수신 overlap, 응답 isend/token-first prototype, host-list 실패 후보.',
'- Packed target, mixed-miss 정책, 새로운 GPU verifier 선택, 새 알고리즘/수식/모델 preset/parameter search 변경.',
'- 과거 실험 도구는 evidence 아래 기록으로만 포함한다. 이들의 runtime 변경이나 default는 이식하지 않았다.', '',
'## 기능과 데이터 검증', '',
'- 최종 코드 CPU/CUDA 회귀: selected 28 + batch 8 + compatibility 70 = 총106회, failure/error/skip0. Suite별 결과는 `checks/CHECKS.json`과 로그에 있다.',
'- Chain T=0.7 및 tree T=0 smoke도 ver1.3/1.4 각각 같은 2입력·64출력 상한·warmup 조건으로 실행했다. 전체 출력·event 일치. 이 짧은 실행의 TPS를 성능 근거로 사용하지 않는다.',
'- 최종 runtime 95개 Python 파일은 clean benchmark checkout과 동일 hash다. Baseline commit과 candidate commit, benchmark source 일치, raw token/time 재집계, 독립 wall, GPU 외부 process 부재를 `RELEASE_AUDIT.json`에서 확인했다.',
'- 기본2× long-step filter, cap/clip·부분 batch를 시간과 token에서 함께 제외하는 정책을 유지한다. Raw는 보존하며 아래 제외 건수를 공개한다.', '',
'| Run | measured 요청 | long step 제외 | cap/clip 제외 |', '|---|---:|---:|---:|']
for r in A['runs']:lines.append(f'|{r["tag"]}|{r["measured_requests"]}|{r["excluded_long"]}|{r["cap_clip"]}|')
lines+=['',
'`release_raw.tar.xz`는 이번8실행의 원본 run/telemetry/provenance/wall 기록이다. `evidence/`는 과거 보고서의 timestamp·commit·수치를 보존한다. 과거 문서의 “push하지 않음”은 당시 상태이며 현재 인수인계는 루트 `MLSYS_VER14_HANDOVER.md`가 정본이다.', '',
'다른 서버에서는 루트 handover 순서대로 새 worktree를 받아 코드·runtime hash를 먼저 확인한 다음 그 서버의 원래 고정 설정으로 검증한다. 기존 실험 브랜치와 연구 결과를 통째로 덮어쓰지 않는다.']
(H/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('Release report written')
