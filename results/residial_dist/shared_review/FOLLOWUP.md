# Residual 오차 비교와 lossless 검증의 추가 점검

2026-09-12 추가 실험은 [학습 없는 후보·위치 배분 개선](../training_free/REPORT.md)에
정리했다. 실제 temperature별 생성, 2,640개 조합, 독립 프롬프트와 wire 동점 검증을 포함한다.

2026-09-11. 기존 8-run의 전체 확률 snapshot을 재분석하고 현재 chain 검증 코드를
점검했다. 모델 생성 실험을 새로 실행하거나 엔진 코드를 수정하지 않았다.
현재 범위는 AWQ LayerSkip-70B + TinyLlama, exit 56, T=1, B=1,
only-proxy chain, tree/sampler_x off다.

**1. 작은 residual 예시가 실제 데이터에서 차지하는 비중**

설명 예시 p=(.30,.30,.40), q=(.28,.29,.43), e=(.27,.33,.40)에서는
Z=TV(p,q)=.03이다. 같은 prefix에서 draft 제안을 평균한 수락률은 97%다.
따라서 draft가 target에 매우 가까운 예시라는 지적은 맞다. 작은 Z는 residual
불안정성이 생길 수 있는 조건이며, 큰 오차의 충분조건은 아니다.

실제 11,980개 non-bonus 위치에서 Z의 중앙값은 .29313, 위치 동일 가중 평균은
.31446이다. 아래의 event 비중은 저장된 실제 draft 경로를 조건으로 한
first-rejection 확률 h_i로 가중했다. 관측한 실제 거절 횟수의 단순 집계가 아니다.

| 조건 | 위치 수 | 전체 위치 비중 | 전체 reject-event 질량 비중 |
|---|---:|---:|---:|
| Z < .03 | 1,824 | 15.23% | 0.67% |
| Z < .05 | 2,174 | 18.15% | 1.08% |
| Z < .10 | 2,891 | 24.13% | 2.65% |
| Z < .20 | 4,303 | 35.92% | 9.08% |

TV(p,e)<.05이면서 TV(R,Rhat)>.5인 위치는 1,160개(9.68%)지만, 그 위치들의
reject-event 질량은 전체의 .43%다. 예시 같은 현상은 존재하지만, 거절이 드문
위치들이라 전체 correction 성능 차이를 이 현상 하나로 설명할 수 없다.
실제 proxy의 평균 TV(p,e)는 위치 동일 가중 .25878, reject-event 가중 .33717로,
설명 예시의 .03보다 훨씬 크다.

**2. 정답을 R로 통일한 비교**

TV(p,e)와 TV(R,Rhat)는 다른 추정 문제의 오차다. 이 두 숫자의 비교만으로
proxy residual과 e-only 후보 정책의 우열을 판정할 수 없다.
설명 예시에서도 TV(R,e)=.40이고 TV(R,Rhat)=2/3이다. 따라서 e를 R의 추정으로
사용할 때의 오차도 별도로 계산해야 한다.

이번에 저장 snapshot 전체에서 다시 계산한 값은 다음과 같다. 모두 같은
11,980개 위치, 같은 true first-rejection h_i 가중이고, 전체 어휘를 사용했다.

| 정답은 모두 R | Proxy residual | Early-exit 확률 |
|---|---:|---:|
| 토큰 제외 전 전체 분포 TV | .495774 | .517927 |
| 양쪽 모두 실제 draft token y를 제외하고 전체 어휘 재정규화한 TV | .492276 | .475678 |

두 번째 줄은 actual candidate policy의 sampled-token exclusion을 양쪽에 동일하게
적용한 비교다. 잔여 질량이 0인 residual은 0개였다. TV 계산에는 top-M 절단을
적용하지 않았다. 거절된 y의 true R(y)는 이상적 산술에서 0이므로, 이 제외는
특히 raw e의 불필요한 질량을 줄일 수 있다. 실제 correction sampler가 cache
후보 집합으로 제한된다는 뜻은 아니다.

첫 번째 줄에서는 residual이 조금 더 가깝고, 두 번째 줄에서는 e-only가 더
가깝다. 이것만으로 top-k 성능이 결정되는 것은 아니다. 실제 후보 정책에는
토큰 제외, top-M 재정규화, 위치 확률 h, 제한된 전역 후보 예산이 들어간다.
동일 위치에 3개씩 고르는 실제 exclusion 적용 정책의 reject-event 가중 coverage는
residual .489398, e-only .539757이다. 이 값은 본 보고서의 prompt-balanced
held-out global top-15 coverage .605563/.642619와 평균 방식과 목적이 다르다.

**3. Cache 준비 분포와 실제 correction sampling 분포**

같은 prefix에서 다음 항등식이 표준 speculative sampling의 분포 보존을 설명한다.

\[
P(\text{최종 토큰}=v)
=q(v)\min(1,p(v)/q(v))+Z R(v)
=\min(p(v),q(v))+[p(v)-q(v)]_+
=p(v).
\]

첫 항은 draft 토큰이 수락되어 나오는 확률 질량이고, 둘째 항은 거절 후 보정으로
나오는 확률 질량이다. Z는 draft 제안 y까지 평균한 거절 확률이다. 이미 관측한
경로의 h_i와 구분해야 한다. [표준 알고리즘 §2.3 및 Appendix A.1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

현재 구현의 순서는 다음과 같다.

1. Proxy 점수로 correction 후보 집합 S를 정하고, 각 후보 c에 대해 prefix+c
   이후의 draft continuation을 미리 계산한다. Root 선정은 top-k이며 실제
   correction을 Rhat에서 확정 샘플링하는 동작이 아니다.
2. Target 최종 분포가 준비되면 원래 draft 토큰을 p/q로 검증한다.
3. 실제 거절이면 전체 어휘의 R=normalize((p-q)+)에서 c를 샘플링한다.
   전부 수락했다면 bonus 위치의 p에서 샘플링한다.
4. c를 cache key에 넣어 조회한다. Hit면 해당 c 이후의 continuation을 재사용하고,
   miss면 c를 그대로 두고 JIT draft를 실행한다. Cache에 들어 있는 토큰만으로
   c를 다시 뽑거나, miss를 이유로 c를 바꾸지 않는다.
5. 다음 검증에서 c는 이미 확정된 root다. c 이후에 q에서 생성한 토큰들만 p/q로
   검증하며, 그때의 p와 q는 prefix+c 및 앞선 수락 토큰들로 조건화된 분포다.

따라서 현재 설계에서 root c에 q(c)나 Rhat(c)를 사용한 추가 ratio acceptance는
필요하지 않다. 그 뒤 draft proposal의 분모는 실제 proposal q다. Cache 준비에
사용한 proxy 점수를 분모로 대체하면 안 된다.

만약 설계를 바꾸어 correction proposal 자체를 s=Rhat에서 뽑고 별도로 보정한다면,
거절 이후의 목표는 R이므로 acceptance는 min(1,R(c)/s(c)), 실패 시
normalize((R-s)+)가 된다. 보통의 next-token proposal을 s로 바꾸는 경우의 목표는
p이므로 min(1,p/s)다. 두 경우 모두 실제 생성 분포를 분모로 사용해야 한다.

**4. 코드 경로와 실험 설정 확인**

- [후보 선택 함수](/home/chokwans99/PSD/ssd/ssd/engine/helpers/p2_tree.py:1200):
  chain_proxy_candidates_fixed의 source 분기는 후보 ID/score만 바꾼다.
- [검증 호출](/home/chokwans99/PSD/ssd/ssd/engine/verifier.py:514):
  최종 logits_p와 speculate_result.logits_q를 verify에 전달한다.
- [실제 검증](/home/chokwans99/PSD/ssd/ssd/utils/verify.py:119):
  p/q acceptance. 174–180에서 실제 (p-q)+로 correction을 샘플링한다.
  193–198에서 speculations[:,0]은 확정 root로 유지한다.
- [Draft 생성 및 cache 응답](/home/chokwans99/PSD/ssd/ssd/engine/draft_runner.py:986):
  jit_speculate는 sample한 토큰과 logits를 함께 기록한다.
  _decode_tree_step도 logits를 기록한 뒤 같은 logits의 sampler를 호출한다.
  _merge_and_populate_cache는 root key와 continuation tokens/logits를 같은 행으로
  저장하고, hit_cache_and_respond는 같은 행의 tokens/logits를 함께 반환한다.
- [Cache 요청](/home/chokwans99/PSD/ssd/ssd/engine/speculator_async.py:287):
  recovery_token_id가 key의 세 번째 항목이다.
- [벤치 설정](/home/chokwans99/PSD/ssd/bench/bench.py:379):
  DUET는 jit_speculate=True를 강제한다. 따라서 cache miss에서도 유효한 q로
  생성한 토큰을 p/q로 검증한다. 일반 verify 함수의 non-JIT miss 경로와 구분된다.
- verifier._run_exit_probe는 실제로 전달된 q와 최종 p를 저장하고, context probe에
  tree/sampler_x가 켜지면 오류를 발생시킨다. 캠페인은 T=1, tree off, sampler_x off다.
- Replay는 고정 snapshot을 읽고 true h_i R_i로 후보 coverage를 평가한다. 각
  후보 정책으로 생성 분포를 바꾸거나, 보정 토큰을 proxy로 대체하지 않는다.

**5. 이번에 추가 실행한 검사와 한계**

[followup_verify_checks.py](followup_verify_checks.py)는 production verify 함수와
Sampler 클래스를 AST로 그대로 로드해 CPU에서 실행했다. GPU model runner 초기화는
생략했다. 세 가지 분포 각각 20만 개, 총 60만 개의 draft proposal에 대해 proposal
분포 q, 첫 최종 출력 p, 거절 시 correction R, 전부 수락 시 bonus p를 검사했다.
고정한 6-standard-error 기반 기준을 모두 통과했다.

예: p=(.5,.3,.2), q=(.7,.2,.1)에서 최종 출력 빈도는
(.500470,.300095,.199435), 거절 시 보정 빈도는 (0,.503337,.496663)이었다.
검증 입력과 난수를 고정하면 jit_speculate=True에서 hit/miss 표시에 관계없이
결과가 완전히 같았다. 어휘 밖 sentinel root를 넣어도 root를 재검증하지 않고
그대로 전달하는 것을 검사했다.

Production hit_cache_and_respond의 B=1 조회도 정상 hit, 다른 토큰, 다른 위치,
다른 sequence, empty cache의 5가지 경우에서 검사했다. JIT model 실행만 stub으로
대체했고, 실제 key lookup과 token/logit 복사, valid_k, glue 조립을 실행했다.
모두 통과했으며, miss 시 cache 후보로 root를 바꾸지 않았다.

잘못된 분모를 넣는 음성 대조 계산에서는 q=(.7,.2,.1)에서 생성했는데
e=(.2,.3,.5)를 proposal로 취급하면, 목표 p=(.5,.3,.2) 대신
(.76,.20,.04)가 되어 TV=.26의 편향이 발생했다.

원래 캠페인이 기록한 verifier/probe/model-runner 등 Python 파일 hash는 모두
현재와 같았다. 당시 따로 기록하지 않았던 verify.py/draft_runner.py 등은 이번에
hash를 기록했다. 과거의 모든 실행 상태를 복원했다는 의미는 아니다.

이 점검은 해당 chain 경로의 확률 역할과 CPU 샘플링/조회 동작을 뒷받침한다.
분산 GPU kernel, KV 상태, 비동기 race, 모든 옵션의 end-to-end 정확성 증명은 아니다.
또한 코드에 float32 연산 및 1e-10 안정화 항이 있으므로 실수의 정확 산술 수준에서
완전히 동일한 분포라는 주장은 별개의 문제다. Lossless의 기준 target은 이번 실험의
AWQ target이며, 양자화 전 원본 checkpoint와의 동일성을 뜻하지 않는다.

재현 명령:

```bash
ssd/.venv/bin/python results/residial_dist/shared_review/followup_stats.py
ssd/.venv/bin/python results/residial_dist/shared_review/followup_verify_checks.py
```

결과: [분포 재분석 JSON](followup_stats.json), [검증 결과 JSON](followup_verify_checks.json).
