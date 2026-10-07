**DUET MLSys 확장 — 통합 변경 기록, 질문 답변, 다른 서버 merge 검토표**

**최신 상태 (2026-10-08): Round5 실험·추가 최적화·검증 완료.** 이번 실제 온도 root 수식 통합·독립 SSD 튜닝·B1/B8 공동 파라미터 탐색과 결론은 **20절**, 최종 수치와 다른 서버의 재현 절차는 **20.4–20.7절**을 따른다. 완료487실행/691cell, 그중 full480실행70개이며 실패4시도도 보존했다. **19절은 완료된 Round4의94개 실행 기록**이다. 0–18절은 round3까지의 역사적 기록이다. 부록의 commit/file 목록은 Round5 runtime 기준으로 갱신했다.

작성일: 2026-10-07, 최종 실험 갱신: 2026-10-08. 이 문서는 이번 systems branch의 작업을 한곳에서 검토하기 위한 기준 문서다. 과거 보고서의 시점별 미완료 설명은 역사적 기록이며, 아래0–18절은 당시 source/history와 저장된 실험을 재검토한 내용이다. 당시 문서 감사에는 새 GPU 실험/runtime 변경이 없었으며, 이후 추가 작업은19절, 최신 상태는20절을 따른다.

**0. 정확한 기준과 먼저 바로잡을 설명**

| 구분 | 고정 revision / 상태 |
|---|---|
| 논문 기준 | `a82f7d24fb36827a9a81a3567f344dccb71f193e` |
| 이번 systems 작업 | `feat/duet-mlsys-coverage`, 코드·실험 기준 `68d0a26d17303a416ad3f27009454ba3ced814be` |
| 작업 checkout | `/home/chokwans99/PSD-mlsys-coverage` |
| 기존 root/tree 연구 | `feat/duet-proxy-source-ablation@cc4a3ba11475aeaf6c20069c9a65315f78b0cf1e`, `/home/chokwans99/PSD` |
| 원격 논문 branch 확인 | 10/07 `git ls-remote origin refs/heads/feat/duet-p2tree-g0 refs/heads/main` 결과 두 ref 모두 `a82f7d2` |
| 공개 SSD 대조 | `https://github.com/tanishqkumar/ssd`, 조회 시 HEAD `d7eb8fa0edb77a6d0876af1903367b9bb82f54e7` |

다른 서버의 미push 변경은 이번 조회로 확인할 수 없다. Merge할 때 그 서버의 HEAD와 working diff를 별도로 남긴다. 공개 SSD의 현재 snapshot과 이 저장소 최초 snapshot(`f46aecd`)을 모두 대조했지만, 최초 import의 정확한 upstream commit을 복원한 것은 아니다.

- **논문 기준 코드에는 이미 B1 tree attention, ordered-sibling residual 검증, accepted-path KV commit이 있었다.** 이번에 처음 발명하거나 누락을 새로 채운 것이 아니다. 이번 핵심은 B>1의 독립 요청들을 같은 실행 경로에서 처리하도록 확장한 것이다.
- 논문 기준에는 B>1 **chain**도 있었다. `valid_k`, split K1/K2, batch proxy wire 등의 기존 구현 위에 수정·최적화를 더했다.
- TPS에서 버린 토큰까지 세면 **과대평가**한다. 과소평가가 아니다.
- Tree에도 이미 CUDA graph를 적용했다. 구조가 바뀔 때마다 graph를 다시 만드는 구현이 아니다.
- 앞서 “완료”는 두 dense pair의 지원·검증 범위를 뜻한다. TPS 전역 최적화, 논문 70B 장비 재현, 이전 reach/gain 연구의 통합, 모든 부동소수점 경로의 bitwise 동일성까지 완료한 것은 아니다.
- **Round3 시점**에는 논문 기존 root/continuation 점수를 유지했다. 당시 성능 수치에는 `e(1-q)`, 새로운 위치 점수, calibrated reach/gain selector가 없었다. 이후 통합·검증은19–20절에 별도로 기록했다.

**1. 오류 여섯 가지의 의미·발생 조건·출처 — 사용자 질문 1**

“오류가 과거 source에 있었다”와 “과거 논문 run이 그 조건을 실제로 밟았다”는 다르다. 아래 영향 판정은 이 구분을 유지한다.

| 항목 | 이전 동작 → 수정 | 출처/발생 조건 | 기존 논문 영향 판정 | SSD 비교 시 처리 |
|---|---|---|---|---|
| Ragged greedy recovery | 짧은 행의 accept 길이는 `valid_k`로 잘라도 recovery를 자르기 전 위치에서 가져옴 → 실제 accept 끝의 target logit 사용 | DUET가 기존에 추가한 서로 다른 K1/K2 길이와 T0의 상호작용. `valid_k` clamp는 `62dc29c`에서 추가됨. 공개 SSD에는 이 ragged 확장 자체가 없음 | 동일 길이 B1/T>0 결과에 이 증상을 그대로 적용할 수 없음. T0+short row+padding 일치 조건의 run을 점검 | 공통 verifier에는 수정 포함. 고정 K SSD에서는 이 조건이 보통 생기지 않음 |
| Paged prefix prefill | page cache를 연속 K/V처럼 펼침 → 각 요청의 page table·실제 KV 길이로 attention | 최초 local snapshot과 공개 SSD에 같은 패턴. 이번 수정 대상은 SGL prefill 경로 | Prefix 재사용 또는 re-prefill을 하고 해당 backend를 쓸 때 문제. 논문 Blackwell용 기존 FlashInfer 경로는 이미 `run_paged` 분기가 있어 이 수정의 직접 대상과 다름 | AR/SD/SSD/DUET 모두 해당 backend에 공통 적용 |
| Fully cached prompt | block 경계에 맞는 prompt가 모두 hit하면 query 0개 → 마지막 query 1개를 재계산 | 공통 block manager의 기존 edge case. KV만 저장하고 마지막 vocabulary logits는 저장하지 않음 | 완전 일치하는 cached prompt가 해당 조건에서 재사용될 때만 발생. 모든 일반 prompt에 영향이 있었다는 뜻 아님 | 공통 적용 |
| Prefill admission | decode batch만 B로 제한하고 prefill은 대기 요청을 더 많이 resident로 수용 → resident 요청 수도 B 이내로 제한 | 최초 local/public SSD의 기존 scheduler 의미. 출력 수학 오류라기보다 **B의 계약을 명시적으로 변경** | 한 요청씩 넣는 B1 runner면 차이 없음. corpus를 한 번에 넣으면 KV pressure·prefill 스케줄·캐시 재사용이 달라질 수 있음 | 모든 방법에서 resident B와 active decode B 정의를 통일. 이전 throughput과 무조건 직접 비교하지 않음 |
| Preemption output budget | 재-prefill 때 생성 prefix를 새 prompt로 취급하면서 출력 시작점도 이동 → 최초 prompt 경계를 별도로 보존 | 최초 local/public SSD의 공통 sequence/scheduler 상태 관리 | 생성 이후 preemption이 실제 발생할 때 영향. 그런 이벤트가 없으면 이 원인으로 결과가 달라지지 않음 | AR/SD/SSD/DUET 공통 적용 |
| Completed-block hash | 완성된 block `j`의 내용/hash를 마지막 page에 기록 → `block_table[j]`와 그 직전 block의 hash 사용 | 최초 local/public SSD의 공통 speculative postprocess 경로 | 완성 block이 마지막 page가 아닌 경우 잘못된 메타데이터가 생길 수 있음. 출력 오염에는 잘못된 entry를 나중에 재사용하는 경로까지 필요 | SD/SSD/DUET에 공통 적용. 영향 노출은 workload별 검사 |

예시:

- Proposal 길이가 A=4, B=2이면 B의 실제 두 토큰을 모두 수락한 뒤 `p_B(next | prefix+2 tokens)`를 써야 한다. Padding까지 우연히 맞았다고 `prefix+4` 위치의 logit을 쓰면 안 된다.
- Physical pages가 요청 A에 `[7,2]`, B에 `[5,9]`로 배치돼 있다면, cache 전체를 0,1,2… 순으로 펼쳐 읽어서는 각 요청의 문맥을 복원할 수 없다.
- Prompt 100개+이미 생성한 30개를 재-prefill하더라도, 원래 출력 예산이 64개면 남은 예산은 34개다. 130개를 새 prompt로 간주해 출력 카운트를 0으로 만들면 이미 출력한 30개와 남은 예산이 틀어진다.

이번 1차 주요 결과 30개 run은 기존 [block hash 노출 감사](block_hash_exposure_audit.json)에서 re-prefill/잘못된 generated-block 재사용 가능성이 없음을 확인했다. **이 감사는 그 30개 run에 한정되며 옛 논문 전체 run의 무영향 증명이 아니다.** 당시 paper driver의 일부는 repository 외부에 있어, 원 논문 TPS counter와 이벤트까지 이번에 모두 재검증하지 못했다.

Source 근거: `ssd/ssd/utils/verify.py`, `layers/attention.py`, `engine/{block_manager,scheduler,sequence,step}.py`. 공개 snapshot의 [scheduler](https://github.com/tanishqkumar/ssd/blob/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7/ssd/engine/scheduler.py), [attention](https://github.com/tanishqkumar/ssd/blob/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7/ssd/layers/attention.py), [step](https://github.com/tanishqkumar/ssd/blob/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7/ssd/engine/step.py)와 대조했다. Source 패턴의 일치 확인이지 공개 SSD에서 이번 전체 GPU matrix를 다시 돌린 것은 아니다.

**2. 모델 호환성 수정 세 가지 — 사용자 질문 2**

1. AMD dtype: `dataclasses.replace()`가 draft config를 다시 읽으면서 AMD checkpoint의 float32 설정을 가져왔다. 해당 FlashAttention 실행과 맞지 않아, draft layer·graph·통신 버퍼를 만들기 **전에** target runtime dtype으로 정렬했다. Llama2/AMD 실험은 FP16이었다. 모델 크기 축소나 양자화가 아니다. 기존 TinyLlama 실험에서 같은 문제가 있었다는 근거는 없다. 공통 `DraftRunner` 수정이므로 SSD/DUET 모두 동일 pair에 적용한다.
2. Token wire: 기존 DUET tree의 `P_iv` pack은 token 15bit+score 16bit+marker였다. Llama3의 128,256 vocabulary를 담을 수 없어 명시적으로 거부하던 범위를, int64 한 개 안에서 token bits 0–31, score bits 32–47, marker bit48로 바꿨다. Payload 개수는 그대로다. **양쪽 프로세스가 같은 형식이어야 한다.** Llama2/TinyLlama의 32k vocabulary에 이 제한이 걸렸다는 뜻은 아니다. DUET 고유 wire 수정이며 SSD에 이 형식을 새로 요구하지 않는다.
3. Qwama P1 workspace: TinyLlama 기준 48MiB 제한이 Qwama의 가장 넓은 P1 FA2 shape에서 필요한 약56MiB보다 작았다. 기본64MiB로 수정하고 override를 유지했다. 이후 B>1 경로에서는 요청별 workspace 중복을 없애 기본128MiB float scratch를 공유했다. Attention integer plan은 구조별로 유지한다. 기존 TinyLlama용48MiB가 잘못됐다는 결론이 아니라 모델/shape별 필요량이 다르다는 뜻이다.

별도 공통 수정으로 token vocabulary **크기뿐 아니라 token→ID mapping**을 검사했다. 공개 SSD의 같은-family 제약만으로 Qwen 구조+Llama vocabulary인 Qwama 조합을 허용/검증할 수 없으므로, 같은 구현 기반의 SSD baseline에도 이 호환성 수정을 적용해야 한다.

**3. TPS 집계 방향과 과거 수치 — 사용자 질문 3**

Verifier가 5개를 수락했지만 남은 출력 예산이2라면 실제 반환량은2다. 이전 counter가5를 세면 같은 시간에 출력량을3개 부풀린다.

\[
TPS_{old}=\frac{N_{emitted}+N_{discarded}}{t},\qquad
TPS_{correct}=\frac{N_{emitted}}{t}.
\]

“실제로 더 출력할 수 있었는데 누락했다”가 아니다. 요청이 끝난 뒤의 가상 작업량은 사용자가 받은 출력 TPS에 넣지 않는다. Acceptance 길이는 별도 진단 지표로 보존할 수 있다. 오류 비율은 `N_discarded/N_emitted`이며, 원시 counts가 없으면 임의로 보정하지 않는다.

이번 1차 결과도 저장된 실제 출력 token으로 재집계했다. 기존 paper runner가 이미 출력 길이를 직접 셌다면 그 수치는 이 counter 오류에 영향이 없다. 내부 `decode_total_tokens`를 썼다면 재집계가 필요하다. SSD/DUET 모두 같은 문제를 가질 수 있어도 AL/EOS 패턴이 다르므로 상대 이득에서 자동 상쇄된다고 가정하지 않는다.

**4. 요청별 query 수가 다른 이유와 packed batch — 사용자 질문 4**

서로 다른 batch가 아니라 **같은 batch 안의 요청별** query 수가 다르다. P1 hit는 K1, P2 hit/JIT-short는 K2개의 proposal을 가질 수 있다. Target 입력에는 현재 recovery token 한 개도 필요하므로 `k_i+1` query다.

예를 들어 proposal 길이 `[4,2,2]`이면 query는 `[5,3,3]`이다.

| 방식 | 입력 표현 | 총 query |
|---|---|---:|
| Rectangular padding | 3행×최대5열 | 15 |
| Packed varlen | 11개 token을 이어 붙이고 경계 `[0,5,8,11]` 전달 | 11 (+총량 bucket 정렬 padding) |

Transformer의 linear/GEMM은 `[전체 token 수, hidden]`로 처리할 수 있다. Attention에는 요청 경계·각자의 KV page·문맥 길이를 전달하여 서로 섞이지 않게 한다. 따라서 batch 연산이 반드시 요청마다 같은 query 길이를 요구하지는 않는다.

현재 chain 구현은 총 query 수를4개 단위 bucket으로 정렬하고, 남는 정렬 query는 **별도 dummy sequence**에 둔다. KV slot=-1이라 실제 요청의 KV를 쓰지 않는다. 출력은 verifier가 기대하는 논리적 dense row로 다시 매핑한다. 실제 shape가 줄어도 kernel 효율 때문에 latency가 같은 비율로 줄지는 않는다.

Packed chain은 현재 SGL backend의 DUET split 경로에 연결돼 있다. 공개 SSD의 고정 K에서는 원래 길이 차이가 적어 이득이 다를 수 있다. 모든 SSD 경로에 이 옵션이 자동 적용된다고 말하지 않는다.

**5. Mixed miss의 동작·threshold·수학적 선택 기준 — 사용자 질문 5**

현재 batch barrier에서는 miss 행의 JIT가 있으면 hit 행도 응답을 기다린다. 다만 all-hit도 draft의 기존 P1/P2 작업 완료·cache 준비·통신을 기다릴 수 있으므로, **all-hit는 JIT 제거 조건이지 draft가 항상 완전히 숨겨진다는 충분조건은 아니다.**

`SSD_MIXED_MISS_AR=1`의 실제 규칙:

- DUET chain, B>1, hit가 하나 이상 있는 mixed batch에서만 동작한다.
- Hit 행은 cache proposal을 사용한다.
- Miss 행은 `valid_k=0`; 현재 recovery만 target에 입력하고 다음 토큰은 해당 prefix의 **target p에서 직접** 뽑는다. T0이면 argmax다. 그 행은 이 step에서 AR처럼 한 토큰씩 진행한다.
- All-miss/startup은 기존 JIT를 유지한다. 다음 cache 준비를 위해 recovery의 draft KV 처리는 필요하다.
- Miss 행의 proxy 후보도 draft residual이 아니라 해당 zero-proposal 상황에 맞는 proxy 분포를 사용한다.
- 현재 B>1 tree 경로에는 이 scheduling 옵션을 적용하지 않았다.

**새 연속 threshold는 추가하지 않았다.** 현재는 ON/OFF 옵션과 위 조건뿐이다. 다만 불변 알고리즘의 단순 코드 최적화가 아니라 AL/latency를 바꾸는 scheduling 정책이므로 ablation을 따로 둔다.

향후 숫자 threshold를 임의 sweep하기보다 calibration으로 판단할 수 있다. 지금 바로 검증할 때 기대 출력/시간을 `(N0,t0)`, JIT를 기다리면 추가되는 값을 `(ΔN,Δt)`라고 두면, Δt>0에서 한-step 처리율 기준은

\[
\frac{N_0+\Delta N}{t_0+\Delta t}>\frac{N_0}{t_0}
\iff \Delta N>\frac{N_0}{t_0}\Delta t.
\]

추가 토큰의 가치가 추가 대기·검증 시간 비용보다 크면 기다리는 쪽이 유리하다. Δt에는 JIT뿐 아니라 query shape 변화와 후처리 차이도 포함한다. 이 식은 상태별 추정이 정확하다는 가정의 국소 기준이며 이후 cache 상태·대기열·latency SLO까지 포함한 전역 최적 정책은 아니다. **이 자동 선택기는 아직 구현하지 않았다.** AL 자체가 목적이면 이 TPS 정책을 AL 개선으로 채택하지 않는다.

**6. 70B 때와 target/draft latency 비율이 비슷했는가 — 사용자 질문 6**

**비슷하게 맞췄다고 검증하지 않았다.** 모델 parameter 비율만으로 이를 대체할 수 없다. 실제 draft는 여러 root/branch를 배치하고 forward round는 순차이므로 단일-token AR latency와 같지 않다. Target TP 수·dtype/양자화·GPU·batch·검증 폭·문맥 길이도 달랐다.

저장된 1차 chain profile을 이번에 다시 집계했다. Nominal B8, K1/K2=4/2, exit21. Step ID≥10만 포함하며 batch가 줄어드는 tail도 들어 있다. 아래는 profiler ON 진단값으로 최종 round3 성능이나 동일 문맥 인과 비교가 아니다.

| CUDA event 구간의 중앙값(ms) | Llama2/AMD | Llama3/Qwama |
|---|---:|---:|
| Target pre+post 합, 같은 step끼리 합산 후 중앙값 | 17.695 | 18.059 |
| P1 draft round replay 한 번 | 0.861 | 2.658 |
| P2 draft round replay 한 번 | 0.737 | 2.488 |
| P1 build 구간 | 5.656 | 13.842 |
| Draft가 proxy를 기다리는 구간 | 6.642 | 0.002 |
| Target speculation 대기 구간 | 4.028 | 6.713 |

Target 모델 구간은 비슷해도 Qwama의 draft round는 더 오래 걸렸다. Qwama profile의 proxy wait가 거의0인 것은 P1 이후 proxy가 이미 준비된 상황과 일관되지만, P1이 모든 시간 손실의 원인이라는 증명은 아니다. 각 구간의 중앙값을 더해서 한 가상 step의 timeline을 만들면 안 된다.

이전 70B/4090 AWQ TP4 calibration은 다른 조건이다. 같은 validation에서 K1/K2=4/2는 cycle29.285ms·U2.892, 8/4는52.858ms·U3.313이었다. **70B여도 긴 draft가 항상 숨겨지지 않았음**을 보여준다. 원 논문의 dense70B/Blackwell/TP2와 이 AWQ 진단을 혼동하지 않는다.

재실험에서는 같은 step의 clock으로 다음을 측정해야 한다.

\[
C=\max(x+D_1,P)+D_2,\qquad
slack=F-C,\qquad exposed=\max(0,C-F).
\]

`x`: target verify 시작에 대한 P1 시작, `D1`: P1 준비·forward 완료까지, `P`: proxy 도착, `D2`: P2 준비·forward·merge, `F`: target의 다음 요청 준비 시각. 실제 구현에 맞춰 glue 등 경계를 고정한다. 대략

\[
K_2d_2+H_2\le F-\max(x+D_1,P)
\]

이면 P2가 deadline 안에 들어갈 수 있다. 우변이 음수면 그 step에는 숨길 P2 시간이 없다. `d2,H2`는 실제 batch/폭에서 측정한다. 기존 cache-wait 정책에서는 이 노출 시간과 다음 lookup/JIT 비용이 함께 cycle을 정한다.

두 phase가 실제로 쓰인 증거는 있다. 초기 round3 sampling cell의 source별 P1/P2 비중은 Llama2 tree31.42%/45.82%, Llama3 tree43.18%/29.13%였다. 이는 P2가 쓸모없었다는 뜻은 아니지만, P2 ON/OFF의 인과적 순이득이나 deadline 준수율은 아니다.

**미완료:** 70B와 7B/8B의 같은 조건 latency ratio·phase slack 비교, 이번 B>1 tree의 step-ID별 완전한 timeline, P1/P2 ON/OFF 대응 실험. `Avg draft step time`에는 대기/통신이 섞일 수 있어 pure draft forward로 나누어 비율을 만들지 않는다.

**7. T0에서 확률은 예산에 쓰고 token은 argmax로 뽑는가 — 사용자 질문 7**

그 구분은 타당하다. 두 값을 명시적으로 분리해야 한다.

\[
q_{score}=softmax(z_D)\quad (T_{score}=1\text{ 고정}),\qquad
x_D=argmax_v z_D(v).
\]

`q_score`는 후보 우선순위/예산 점수이고, T0의 실제 deterministic proposal 법칙은 argmax의 point mass다. Greedy verifier는 target argmax를 따라가므로 soft score를 acceptance 분모로 쓰지 않는다. 별도 score temperature sweep을 추가할 필요 없이 우선1로 고정할 수 있다.

구현의 실제 구분:

- Chain continuation: 한 문맥에서 argmax 한 개씩 생성하는 것이 자연스럽다.
- Tree continuation: C>1의 형제는 같은 argmax를 반복해서는 서로 다른 후보가 생기지 않는다. 현재는 **draft score 상위 C개의 서로 다른 token**을 둔다. 첫 후보가 최고점이고 후속 후보는 결정론적 대안이다. 모든 부모에서 자식 하나만 원하면 C=1이지만, 그 경우 root 아래 continuation은 chain으로 바뀐다.
- `q_probs_from_logits`/`selected_q_probs_from_logits`는 T0 점수 계산에서1로 나누어 soft score를 유지한다. P1 tree root 준비도 이를 사용한다. Target T0 tie는 argmax one-hot으로 수정했다.
- Root correction 후보까지 모두 argmax 하나로 바꾼다는 뜻은 아니다. Root cache는 target correction의 여러 가능성을 준비하는 별도 작업이다.
- `topk`의 exact tie 순서는 `argmax`의 첫-index 규칙과 동일하다고 일반화하지 않는다. 논문에 “첫 draft 후보가 항상 torch.argmax와 동일”을 명시하려면 동점 규칙도 고정해 검사해야 한다. Target 출력의 greedy 규칙과 후보 준비의 tie 순위는 별개다.

T>0용 frozen acceptance/reach calibration을 T0의 실제 acceptance 확률로 그대로 해석할 수 없다. T0에서는 “target argmax가 후보에 포함되고 그 경로로 이어질 확률”을 별도 평가해야 한다. T>0 target+T0 branching draft는 현재 명시적으로 지원하지 않는다.

**8. 이전 tree 개선 연구는 어디까지 했고 이번에 포함됐는가 — 사용자 질문 8·14**

포함되지 않았다. 다음 세 연구 단계를 구분한다.

| 연구 | 실제 결과 | 해석/현재 상태 |
|---|---|---|
| q-path → calibrated reach, full online | 480문항/560turn×2정책×3seed, hit descendant AL2.065133→2.100308(+1.70%), 차이95% CI[-0.002489,+0.072056] | 실제 생성 비교 완료. 전체 AL 우위 미확정 |
| Posthoc 점수·두-node fanout | 37,869 hit trees·262,145 nodes; reach MSE 약15.03% 감소. 국소 fanout gain .51485→.53006(+2.95%) | 관측 pool의 국소 분석. 전체 inference AL+2.95%가 아님 |
| C=3 후속 gain/frontier | q+breadth .681903, reach+breadth .685109, q+gain .698233, reach+gain .696515 | 관측602라운드/369문항의 기대 추가 AL. Reach+gain 차이CI[-.002021,+.024562], 전체 online 결과 아님 |

후속 여섯 정책은 `q_path`, `reach`, `q_gain`, `reach_gain`, `reach_frontier`, `reach_gain_frontier`다. 목적은

\[
\max_{\{c_u\}}\sum_u\widehat\rho(u)\gamma_\phi(c_u),\qquad \sum_uc_u\le B_{nodes}.
\]

여기서 `B_nodes`는 node 예산이며 batch B와 다르다. 부모별 새 자식 수를 **그 자식 token을 샘플하기 전에** 정한다. 이미 뽑힌 자식의 confidence로 사후 잘라내는 것과 다르다.

후속 real-model campaign은 완료되지 않았다. 최신 기록 `results/duet_tree_followup/STATUS_20260927.md`에 따르면 첫 smoke의 AWQ 로딩 중 다른 GPU process와 메모리가 겹쳐 OOM, queue 중단·STOP 보존 상태다. 6정책×3seed full480과 두 정책 frontier trace는 미실행이다. 이 실패로 알고리즘 성능을 판정할 수 없다. 오래된 REPORT의 “대기/실행 중” 문구를 현재 상태로 인용하지 않는다.

이번 systems 결과는 **기존 tree와 chain 비교**다. 위 연구는 **같은 tree budget에서 selector 비교**다. 모델·정밀도·batch·AL 정의도 다르다. 특히 기존 연구 AL은 root/recovery를 제외한 hit-conditional descendants이며, 이번 round3 AL은 verification event별 recovery 포함 길이다. 상대 개선율을 서로 직접 비교하거나 합치지 않는다.

이전 연구 자료는 research branch `cc4a3ba`에 있다:

- `results/duet_tree_al_full/REPORT.md`
- `results/duet_tree_posthoc/{FINDINGS,REPORT,THEORY}.md`
- `results/duet_tree_followup/{REPORT,STATUS_20260927}.md`
- `results/duet_tree_analysis/score_hook.py`, `results/duet_tree_followup/policy_hook.py`

**9. 이번 실행 최적화를 어떤 tree 알고리즘에도 쓸 수 있는가 — 사용자 질문 8의 추가 답변**

요청별 arena·cache·KV 분리, round별 batched forward, ancestor attention, workspace 공유, 고정 용량 graph는 여러 selector에 재사용할 수 있다. 그러나 임의 알고리즘에 수정 없이 자동 적용되는 것은 아니다.

계약은 다음과 같다.

- 같은 모델/dtype, 같은 physical round width·round 수·page 형상의 arena를 묶는다.
- Root와 node ID는 요청별 독립 namespace다. 요청별 예산을 유지한다.
- Parent가 child보다 앞에 있고 ancestor closure와 sibling 순서를 보존한다.
- T>0에서는 실제 proposal/temperature/sampler 변환에 맞는 parent-q를 전송한다.
- CUDA capture 전에 selector/hook/calibration을 설치한다. Capture 후 Python 함수를 바꿔도 이미 capture한 graph가 바뀌지 않는다.
- 새로운 알고리즘이 round 폭·forward 수·attention 구조·후보 생성 법칙을 바꾸면 graph/통신/verifier까지 재검증한다.

기존 reach/gain 연구는 process-local monkeypatch다. Runtime에 자동 연결된 production CLI가 아니다. 새 batched executor가 기존 `iter_rounds`를 재사용하므로 이식 경로는 있지만, 실제 hook 적용·각 요청의 selector output·eager/graph parity를 검증해야 한다. 새 dense pair에 70B AWQ calibration 값을 그대로 일반화하지 않는다. **B>1 reach/gain 통합은 남은 작업이다.**

**10. “전체 tree 실행 경로 연결”의 구체적 의미 — 사용자 질문 9·10·11**

기존 B1에는 아래 한 요청의 경로가 있었다. 새 경로는 여러 요청을 동시에 처리하면서 종료/재충전에도 각 상태를 유지한다.

```text
Target: 요청 A의 실제 recovery → Draft cache(A, 직전 terminal, recovery)
                                      ├─ hit: 저장된 tree
                                      └─ miss: JIT chain
Target: 요청 B의 실제 recovery → Draft cache(B, 직전 terminal, recovery)
                                      ├─ hit: 저장된 tree
                                      └─ miss: JIT chain
          ↓ 요청별 parent/q/KV를 보존한 batched target attention
          ↓ A와 B를 각각 검증, 수락 경로 KV를 각자의 prefix에 반영
          ↓ 다음 요청에 필요한 draft KV 복원 + 요청별 P1/P2 forest 준비
```

이전 primitive는 “A/B의 이번 draft forward를 묶기”만 했다. Cache 조회, response wire, target verification, KV commit까지 연결되었다는 보장은 없었다. Round3는 그 나머지를 연결했다.

논문 기준 `a82f7d2`의 tree 기능은 B1/T>0 조건이었다. B>1에서는 tree executor gate가 fallback했고, target tree의 B1 계약도 있었다. 따라서 최신 논문 코드를 놓치고 tree attention을 다시 발명한 것이 아니다. **기존 B1 기능을 새 기능처럼 설명한 이전 답변의 표현을 정정한다.**

Miss는 논문 기준과 새 batch-tree 기본 경로 모두 JIT **단일 경로 chain**이다. “토큰 하나”라는 뜻은 아니다. `SSD_DUET_JIT_SHORT=1`이면 K2개, 아니면 통상 max(K1,K2)개를 순차 생성한다. 그 chain을 tree wire에서는 각 node가 자식 하나를 갖는 형태로 표현할 수 있다. Miss에서 즉석 branching tree를 만드는 구현은 아니다. Mixed-miss AR는 5절의 별도 chain-only 옵션이다.

Target ancestor attention은 B1에서 이미 존재했다. 새 구현은 서로 다른 요청의 prefix·page table·ancestor mask·depth-based RoPE를 한 batched attention에 넣고, padding의 KV slot을-1로 막는다. 수락 경로만 gather/scatter하는 의미도 기존과 같다.

**11. “실제 proposal 법칙 보존”에서 바꾼 것 — 사용자 질문 12**

기존 B1의 `tree_verify_walk_tensor`도 이미 올바른 형태의 ordered-sibling residual ladder를 사용했다. 이번에 chain 방식 `p/q`만 쓰던 옛 tree를 전면 수정한 것처럼 설명하면 부정확하다.

새 batch 경로에서 그 법칙을 유지하도록 연결·병렬화하고 추가 edge case를 수정했다.

\[
R_1=p,\quad Q_1=q,\quad
a_j=\min(1,R_j(x_j)/Q_j(x_j)),
\]
\[
R_{j+1}=norm([R_j-Q_j]_+),\qquad
Q_{j+1}=norm(Q_j\,1_{v\ne x_j})\quad\text{(기각 시)}.
\]

앞 형제의 기각 뒤 둘째 형제를 검사할 때는 원래 p/q 그대로가 아니다. 같은 부모의 후보들은 원래 parent-q 참조를 공유해야 하고, 그 원래 분포로부터 위 조건부분포를 순서대로 만든다. 첫 smoke에서 새 wire의 sibling reference 연결 오류를 잡아 수정했다. 기존 B1이 같은 참조 오류를 갖고 있었다는 근거는 아니다.

독립 요청·문맥은 병렬 계산하고, 각 문맥 안의 sibling 갱신은 순차 법칙을 유지했다. 각 node의 independent coin을 먼저 만들어도 방문하지 않은 coin은 무시한다. Scalar reference와 고정 coin 검사를 했다.

추가 수정:

- WOR support 소진 뒤 확률0의 padding node 수락 방지: 새 batched verifier edge case.
- Mixed T0 tie의 deterministic argmax: 새 mixed-temperature 검증 edge case.
- G>M인 새 배치 경로: sampled confidence rerank 대신 generation-order prefix 보존. 원래 B1 confidence rerank까지 이 수정이 적용됐다고 말하지 않는다.

G>M 사후 selection의 분포 문제는 이전 tree 연구에서도 이미 확인했다. 단순 ancestor/sibling closure만 지켜서는 실제 proposal 선택 법칙이 보존되지 않을 수 있다. **Merge 후 legacy B1 G>M을 사용하려면 별도 검토가 필요하다. 이번 성능은 G=M이다.** 또한 임의 adaptive selector의 losslessness를 이 검사 하나로 보장하지 않으며, 정책이 바뀌면 후보 생성/사용 조건까지 다시 검사한다.

**12. Tree의 CUDA graph 재사용과 추가 최적화 — 사용자 질문 13**

두 혼동을 정정한다.

- 앞서 +43.86%/+5.62% 표는 chain의 graph 개선이 아니라 **B8 tree 보조 연산 graph OFF/ON 비교**였다.
- Tree topology가 달라져도 같은 용량이면 tokens/parents/mask/positions를 고정 buffer에 덮어쓰고 graph를 재사용한다. Graph는 topology 자체로 key를 만들지 않는다.

현재 주요 graph key:

| 부분 | key / 재사용 조건 |
|---|---|
| Target tree 및 draft glue | `(batch_capacity, query_width, page_bucket)` |
| P1/P2 forest | phase·context bucket·batch별 executor, 내부 page bucket |
| Root/input/accept | 해당 batch·node·vocab·executor shape |
| 선택적 packed tree | `(batch_capacity, total_query_capacity, page_bucket)`; 요청별 query 경계 변경 시 plan 갱신 |

실제 동일-GPU optimized 로그의 **draft forest capture**는 Llama2 32회에 phase replay 총7,382회, Llama3 30회에 총5,430회였다. Warmup+두 pass 누계다. Target/glue/root/input/accept capture까지 센 지표는 아니므로 전체 engine capture 비율로 사용하지 않는다. 그래도 “tree마다 거의 새로 capture”한다는 설명은 이 실행기에 맞지 않는다.

첫 pass가 느려지는 것은 처음 만난 shape에 대한 lazy capture가 있기 때문이다. 페이지 수, 활성 batch, context bucket이 늘면 추가 shape가 필요하다. 두 번째 pass에도 처음 보는 shape가 생길 수 있다. Chain도 일반적으로 단일 graph 하나가 아니라 B/K/packed-total에 따른 여러 graph를 사용한다.

**이미 완료한 최적화:** request별 forward batching, float workspace 공유, 짧은 query bucket, root/input/accept capture, tree forward pre/post capture, packed target query의 실험적 구현.

**남은 구체적 최적화 후보:**

1. Calibration에서 자주 쓰는 graph key를 수집해 서비스 전에 bounded prewarm. 모든 조합 무제한 capture는 메모리·시작 비용을 늘린다. Target/glue/forest/helper별 capture 횟수·시간·cache hit 계측이 먼저 필요하다.
2. `BatchedTreeForward.run`의 CPU 입력/ancestor mask 조립과 H2D, `_save_entries`의 `.cpu().tolist()` 등 host 왕복 감소. 현재 arena-input graph가 target/glue의 모든 CPU 준비까지 제거한 것은 아니다.
3. Target의 exit/final norm·LM head 일부는 `pre/post` model graph 밖이다. TP collective 순서와 proxy overlap을 보존하는 범위에서 capture/fusion 후보를 평가한다. “전체 target가 이미 한 graph”라고 표현하지 않는다.
4. Packed tree의 요청별 partition 변화에 따른 host re-plan 비용 분리. 지원 API를 확인한 device metadata 갱신, 제한된 length-layout bucket, 덜 촘촘한 total-query bucket을 대조한다. 라이브러리 private plan 구조를 무조건 재사용하면 correctness가 깨질 수 있다.
5. Parent-q 전송은 같은 부모를 참조하는 여러 node에 중복된다. 정확한 원래 q를 유지하면서 unique-parent 단위로 압축할 여지가 있다. Top-k approximation으로 q를 바꾸는 작업과 구분한다.
6. Hit-first 분리 batch / miss 비동기 합류 scheduler. 작은 batch·추가 launch 비용과 공정성까지 측정해야 하며 아직 구현하지 않았다.

현재 packed tree는 query 사용률 약65%→93%였지만 TPS는 Llama2 423.43→277.57, Llama3 233.81→224.40으로 악화되어 OFF다. “Tree에 graph가 없어서 packed가 느렸다”는 결론이 아니다. 위 후보들의 효과는 아직 입증되지 않았다.

**13. 이번 branch 전체 작업·실험 결과 요약**

| 단계 | 구현/실험 | 결과와 한계 |
|---|---|---|
| Round1 공통 지원 | Ragged greedy, paged cache, admission, token wire/dtype, fast verifier, B>1 proxy graph, greedy sampler, preemption, counter, block hash, IPC namespace, colocated norm capture | 같은 설정 전체480×두 모델×3seed 최적화 전후2,880출력 동일. 코드만 개선 Llama2+2.3%/Llama3+3.1%. 184검사 당시 통과 |
| Round1 튜닝 | P1 폭, K1/K2, exit, replica, AR/SSD/SD, 긴 입력/배치 stress | Llama2 P1폭4/greedyK1=6 근거. Llama3 P1폭1 소표본 이득은 전체 입력에서 실패. T0 sampler 자체 이득<0.5% |
| Round2 chain | Packed chain, mixed-miss AR | Base→packed: Llama2 634.15→651.70 TPS, Llama3 646.76→647.58. Llama3 mixed672.26. 480×독립process3회 |
| Round2 SSD 대조 | 같은 pair 교대 실행, SSD K3/F2와 각 최종 설정 | Llama2 DUET652.45±3.79/SSD649.08±1.70: 차이 미확정. Llama3 673.06±5.54/652.33±4.16: 관측+3.18%, AL 감소 정책 포함 |
| Round2 tree | B1 greedy tree, Qwama workspace, batch forward primitive, host topology 재사용 | B1 tree AL 상승/TPS 하락. Host topology TPS 개선 없음→OFF. Primitive는 serving 전체가 아니었음. 263검사 당시 통과 |
| Round3 tree | B>1 cache/wire/attention/verification/KV 연결, workspace·bucket·graphs | 두 full pair B1/2/3/4/8/16, T0/.7. B8 full480 tree/chain, P2-only B8/16 full480, TP2 작은 통합, preemption 검사 |
| Round3 same-GPU | Tree helpers eager/graph, chain, packed tree, GPU 배치 대조 | Graph 두 번째 pass L2 294.33→423.43 vs chain538.72; L3 221.36→233.81 vs chain362.86. 같은 engine 두 pass이지 독립 반복CI 아님 |
| Round3 AL | 기존 tree vs chain, T별3개 target-seed cell | L2 T0 +5.94%, T.7 +4.99%; L3 T0 +4.36%, T.7 +1.57%. Hit는 낮아짐. 새 reach/gain 결과 아님 |
| Round3 negative 결과 | Packed, PIX/NODE 배치 가설 | Packed 두 모델 TPS 악화. GPU pair 변경만으로 이전 높은 L3 수치 재현 안 됨. 원인 미분리 |
| Round3 정확성 | Dense attention, scalar ladder, fixed coins, Monte Carlo, RNG, TP2, full HF first-divergence, 생성 중 preempt | 최종280검사 통과. AR bitwise 동일 보장 아님. 34 campaign jobs 중30성공·초기4실패 기록, 완료480-input cell64개 |

AL/TPS 실험 환경: RTX4090 24GiB, dense LayerSkip7B+AMD FP16 / LayerSkip8B+Qwama BF16, 주요 targetTP1+draft1GPU. Corpus480first-turn, input512/output64, naturalEOS, chat template 없음. 논문70B/Blackwell/TP2나 원본 full multi-turn 프로토콜 재현이 아니다. AR1GPU와 SSD/DUET2GPU 비교는 총 GPU 예산이 같지 않다.

Round3의 T별 seed cell은 target RNG만 reset되고 draft RNG는 계속 진행한다. Round1/2의 새 process 반복과 구분한다. 첫 pass의 추가 geometry capture를 숨기지 않는다. 이전 revision과 다른 GPU pair에서 나온 절대 TPS를 비율 계산에 섞지 않는다.

Greedy 검증은 전체 token 완전 일치와 분포/규칙 검증을 분리했다. 예를 들어 round3 B8 AR48문항 완전 일치는 Llama2 46/48, Llama3 38/48. 첫 차이는 HF의 근접 logit 후보에서 관찰됐지만 모든 차이가 무해하다는 증명은 아니다. Packed tree의 전체480 비교에서 차이가 난 모든 첫 prefix도 HF로 검사했다.

생성 중 preemption은 Llama2 196회/Llama3 70회를 별도 합성 workload로 유도했다. 독립 HF continuation 검사12/10지점에서 기존 prefix 보존과 다음 argmax를 확인했다. 진단 workload TPS를 주요 성능표에 포함하지 않았다.

**14. 공정한 SSD 비교를 위한 실험 설계**

공개 SSD에도 batching, greedy, CUDA graph, prefix cache의 기본 기반이 있다. “기존 SSD는 B>1/T0을 못해서 DUET가 새로 지원했다”로 서술하지 않는다. 새로운 두 모델 조합/장비/실행 범위를 위해 공통 엔진을 이식·수정한 부분과 DUET 고유 확장을 나눠야 한다. [공개 SSD 설명](https://github.com/tanishqkumar/ssd/tree/d7eb8fa0edb77a6d0876af1903367b9bb82f54e7)

| 비교군 | 포함할 것 | 의미 |
|---|---|---|
| 원본 SSD 재현 참고 | 고정 upstream commit, 원래 지원 모델/장비 또는 명시적 최소 이식 | 원본 재현 범위. 실행 실패를 algorithm 성능0으로 세지 않음 |
| SSD + 공통 기반 수정 | Cache/preemption/accounting/model dtype/vocab/backend/동일 scheduler 계약, 적용 가능한 fast verifier·greedy sampler | DUET와의 주 비교 기준 |
| SSD + 공통 scheduling 후보 | Mixed-miss AR/hit-first 등 SSD에도 의미가 있으면 별도 이식·동일 예산 튜닝 | DUET만 일반적인 엔진 이득을 독점하지 않게 하는 보조 비교. 현재 SSD mixed-miss 이식은 미완료 |
| DUET 기존 알고리즘 | 위 공통 기반+논문 P1/P2, 기존 root/tree 정책 | DUET 본체의 효과 |
| DUET + 각 개선 | Packed/mixed-miss, root score, position score, tree selector를 하나씩 변경 | 알고리즘 기여와 시스템 최적화의 ablation |

현재 실험의 SSD는 **수정된 같은 저장소에서 DUET를 끈 baseline**이지 순수 upstream 재현은 아니다. Common fixes는 포함됐지만 packed/mixed-miss는 DUET-gated 경로다. 고정 K SSD에 packed의 동일 문제가 없을 수 있으나 mixed-hit scheduling은 SSD에도 검토할 가치가 있다. 기존 +3.18%를 이 가능성까지 최적화한 SSD에 대한 최종 우위라고 쓰지 않는다.

실험 계약:

1. 같은 target/draft weights·token mapping·dtype·GPU budget/배치·TP·backend·KV memory policy. Shared fixes와 runtime options를 baseline manifest에 명시한다.
2. 같은 corpus/tokenization/history/EOS/output cap. Paper runner와 이번 harness를 별도 이름으로 기록한다.
3. 같은 tuning 비용과 독립 calibration/validation/test 분리. SSD의 K/fanout도 튜닝하고 DUET만 test set에서 선택하지 않는다.
4. 먼저 같은 physical budget에서 AL/hit를 비교하고, 별도로 각 방법의 최선 throughput 설정을 비교한다. Root→hit, tree→AL의 연구 목적을 유지한다.
5. Root score×position score×tree selector를 조합하되, 먼저 하나씩 분리한다. 이전 `e(1-q)`와 reach/gain이 자동 적용됐다고 가정하지 않는다.
6. Cold capture/startup, warm steady decode, prefill 포함 generation TPS, 실제 출력 count를 분리한다. Batch arrival와 residency 계약도 맞춘다.
7. 같은 GPU pair에서 순서를 교차하고 독립 process seeds를 반복한다. 정해진 전체 문항 기준 paired uncertainty를 보고하며, 한 engine 반복을 독립 process 반복으로 부르지 않는다.
8. 원 논문 target/draft timing과 phase slack을 먼저 확인한다. 7B/8B 결과의 순위를 그대로70B에 일반화하지 않는다.
9. AR 비교는 target1개 latency 외에 같은 총 GPU 예산의 AR replica throughput도 포함한다.
10. 정밀도/shape별 greedy 차이, stochastic proposal 법칙, G>M 정책을 validation과 분리하지 않는다. Invalid 설정의 빠른 TPS는 결과에서 제외한다.

**15. 남은 작업과 우선순위 — round3 당시 상태 (최신은20절)**

| 우선순위 | 작업 | 완료 조건 |
|---|---|---|
| P0 merge | 다른 서버 source와 겹친 공통 파일을 의미 단위로 통합 | 아래16절 checklist, branch별 기존 회귀 모두 통과 |
| P0 논문 영향 감사 | 당시 runner의 실제 count, preemption/prefix backend, B residency, G/M 조건 확인 | 원본 outputs/logs로 영향을 run별 판정; 없으면 미확인으로 남김 |
| P0 correctness | Legacy B1 G>M 재정렬 정책 결정 | G=M 고정 또는 proposal 법칙을 보존하는 선택기로 교체·검증 |
| P1 timing | 논문70B와 두 dense pair의 phase timeline/latency ratio 비교 | 실제 P1/P2 deadline slack 분포·노출 비용·source별 AL/hit 측정 |
| P1 selector | 기존6정책을 새 B>1 executor에 이식 | B1/B>1 eager/graph·q-reference·요청격리 검사, full online 별도 calibration/여러seed |
| P1 baseline | SSD 공통 fixes/greedy/fast verifier 확인, mixed-miss의 대칭 적용 가능성 | 같은 튜닝 예산과 paired full 데이터 대조 |
| P2 graph | 모든 graph 종류의 capture 계측·bounded prewarm·bucket 선택 | Cold 비용/메모리/warmTPS 모두 개선; topology와 RNG 보존 |
| P2 hot path | Host metadata/KV 준비, LM head, parent-q 중복 전송 | Profile로 비용 분리 후 필요한 경로만 변경·정확성 검사 |
| P2 packed/scheduler | Plan 갱신 비용 절감, hit-first/miss join | 전체 모델 동일 조건 A/B에서 확인. 현재 packed tree OFF 유지 |
| P2 coverage | B16 chain 대조, B별 full 반복, 긴 context/output, 실제 targetTP2 throughput | B16tree vs B8chain처럼 조건이 다른 수치를 섞지 않음 |
| 연구 설명 TODO | Root 개선식·위치 점수·reach/gain 전체 설명 | 사용자 이해 확인까지 완료 처리하지 않음 |

“남은 최적화가 없다”가 아니다. 위 항목은 구현만 남은 것, 실험만 남은 것, 효과가 아직 가설인 것을 구분한 목록이다. 이번 질문에 대한 문서 작성 중 이를 GPU 대규모 실험까지 새로 수행했다고 주장하지 않는다.

**16. 다른 서버 merge 검토 절차**

현재 research `cc4a3ba`와 systems `68d0a26`이 공통 base 이후 **동시에 수정한 runtime 파일6개**:

| 파일 | 충돌 때 보존할 양쪽 의미 |
|---|---|
| `ssd/ssd/config.py` | Root source/only-proxy gate + greedy/tree/batch 계약, vocab wire 범위. Unsupported 조합을 조용히 fallback하지 않음 |
| `ssd/ssd/engine/helpers/p2_tree.py` | Research source 선택·score hook + T0 score/proposal 구분·32bit wire·기존 stochastic law |
| `ssd/ssd/engine/helpers/cudagraph_helpers.py` | Exit tap/probe self-check + batched proxy/packed/greedy captures. 새 graph의 source 정책 누락 방지 |
| `ssd/ssd/engine/verifier.py` | Research probe/only-proxy source + fast/greedy/mixed miss 및 batch-tree dispatch. Root score를 proposal denominator에 넣지 않음 |
| `ssd/ssd/engine/model_runner.py` | Hidden tap/replica + 새 tree/packed captures·prefill 처리·dtype. TP collective 순서 보존 |
| `ssd/ssd/engine/llm_engine.py` | Collector flush + vocab 검증·IPC namespace·greedy 계약·계측 |

다른 서버 HEAD가 `cc4a3ba`와 다르면 교집합을 다시 계산한다. 특히 `draft_runner.py`, `scheduler.py`, `sequence.py`, `attention.py`, wire parser, 모델 kernel이 그 서버에서 추가 변경됐을 수 있다. Text conflict가 없더라도 semantics 검토가 필요하다.

- [ ] 양쪽 HEAD·dirty diff·실행 환경·model revision을 기록한다. 기존 연구 checkout의 미추적 `results/residial_dist/root_progress_20261001/` 자료는 git merge에 자동 포함되지 않으므로 별도 보존한다.
- [ ] 통합 전 공통 base를 확인하고 통합용 branch/worktree에서 작업한다. 결과 JSON·실패 로그를 덮어쓰지 않는다.
- [ ] Common fixes를 AR/SD/SSD/DUET에 같은 의미로 적용한다. Scheduler의 resident-B 변경은 별도 정책으로 기록한다.
- [ ] Target/draft 양쪽의 P_iv wire v2와 새 batch tree response epoch를 함께 맞춘다. 한쪽만 이식하지 않는다.
- [ ] `--duet_proxy_source`, only-proxy/P1 off, split K1/K2, tree ON/OFF, greedy, mixed temperatures의 dispatch를 확인한다.
- [ ] Root 정책 변경은 eager/B1 graph/B>1 graph/tree proxy 각각에 전달되는지 검사한다. 새 `BatchedTreeProxy.candidates`는 기존 residual ladder를 직접 구현하므로 chain source hook merge만으로 e(1-q)가 적용되지 않는다.
- [ ] Reach/gain hook은 모든 spawned worker에서 arena 생성·capture 전에 설치한다. CLI 값만 달라지고 graph가 같은 정책으로 남는 문제를 막는다.
- [ ] G>M와 sibling/parent-q 계약을 확인한다. Algorithm 변경에 맞는 finite-vocab distribution negative control도 유지한다.
- [ ] B1/2/3/4/8/16, T0/.7, mixed hit/miss, shrink/refill, prefix hit, fully cached query, 생성 중 preemption을 검사한다.
- [ ] 독립 dense attention/scalar verifier, 고정 coin, graph RNG, TP2 통신 순서를 검사한다. 모든280개 검사는 이 서버 코드 기준이므로 merge 후 다시 실행한다.
- [ ] 새로운 selector 검사/원래 research 회귀도 별도로 실행한다. Systems280개로 research 정책 전체를 검증했다고 하지 않는다.
- [ ] Blackwell/backend/library가 바뀌면 packed attention private plan 및 mask offset을 재검증한다. 현재 packed chain은 SGL 전용, packed tree는 설치 FlashInfer plan 내부에 의존한다.
- [ ] Environment/path/arch는 새 서버 값으로 옮긴다. `/home/...`나 RTX4090 arch8.9를 그대로 복사하지 않는다. 공유 venv를 덮어쓰지 않는다.
- [ ] 논문 원 driver/chat template/length 프로토콜과 repo harness를 구분하여 새 결과 디렉터리를 만든다.
- [ ] 마지막으로 correctness를 유지한 공통 baseline → 동일 설정 최적화 A/B → root/tree 정책 ablation → 각 방법 best throughput 순서로 평가한다.

**17. 기본값·지원 범위·재현 진입점 — round3 기준**

| 옵션 | 현재 선택/의미 |
|---|---|
| `SSD_BATCH_TREE_ROOT_GRAPH`, `INPUT_GRAPH`, `ACCEPT_GRAPH` | 각각1; helper graph ON |
| `SSD_PACKED_TREE_VERIFY` | 0; 성능 악화로 opt-in만 유지 |
| `SSD_BATCH_TREE_FIXED_VERIFY` | 0; 짧은 view bucket 허용 |
| `SSD_BATCH_TREE_WORKSPACE_MB` | 128; 동일 stream float scratch 공유 |
| `SSD_BATCHED_TREE` | 기본0이나 B>1 tree config는 자동 새 경로. 1은 B1에서도 새 경로 강제 검사 |
| `SSD_PACKED_VERIFY` | Chain 전용 별도 옵션. Tree와 혼동 금지 |
| `SSD_MIXED_MISS_AR` | 기본0, DUET chain 실험용. AL 감소 가능 |
| `SSD_TREE_HOST_TOPOLOGY` | 기본0; B1 실험에서 TPS 이득 없음 |

새 batched tree는 dense full vocabulary 두 pair 검증 범위다. EAGLE, raw proxy-on-draft, exit top-m gather는 지원하지 않는다. TP2는 작은 통합 검사를 마쳤지만 실제 논문 TP2 throughput을 완료한 것은 아니다.

Source 시작점은 `ssd/ssd/engine/helpers/batch_tree_{draft,common,forward,inputs,roots,sampling,verify}.py`, `batched_tree_executor.py`, `packed_tree_forward.py`다. Scheduler/sequence/attention/verifier의 공통 변경도 함께 필요하다. 파일 하나 복사로 전체 기능이 이식되지 않는다.

```bash
# Systems 회귀: GPU가 할당된 환경에서 실행. 이번 문서 감사에서는 재실행하지 않음.
CUDA_VISIBLE_DEVICES=0 MLSYS_PYTHON=/새환경/bin/python \
  bash results/mlsys_coverage/run_regressions.sh

# 새 서버의 model/prompt 경로를 반영한 새 plan 사용. 기존 결과를 덮어쓰지 않음.
/새환경/bin/python ssd/bench/mlsys_campaign.py \
  --plan /새서버/plan.json --directory /새서버/새결과 \
  --gpus 0,1 --port 33000 --timeout 2100

# 본 문서의 source/history/profile 감사 재생성. Upstream checkout은 위 SHA 고정.
python results/mlsys_coverage/review_audit.py --upstream /path/to/pinned-ssd
```

**18. 근거 파일과 실험 재생성**

| 내용 | 파일 |
|---|---|
| Source 패턴/변경 교집합/기존 profile 재집계 | [review_audit.py](review_audit.py), [review_audit.json](review_audit.json) |
| Round1 전체 실험·TPS 수정 집계 | [REPORT.md](REPORT.md), `FINAL_NUMBERS.json`, `GREEDY_NUMBERS.json`, `analyze_results.py` |
| Round2 독립 process chain/SSD 비교 | [round2/REPORT.md](round2/REPORT.md), `round2/NUMBERS.json`, `round2/*_plan.json` |
| Round3 전체 결과 | [round3/REPORT.md](round3/REPORT.md), [round3/SUMMARY.md](round3/SUMMARY.md), `round3/NUMBERS.json`, `RUN_INVENTORY.csv` |
| Round3 재현·파일별 역할 | [round3/HANDOVER.md](round3/HANDOVER.md) |
| Same GPU graph/chain, packed, placement | `round3/comparison_plan.json`, `packed_comparison_plan.json`, `placement_plan.json`와 각각의 결과 폴더 |
| Greedy/HF/preemption | `round3/*_audit.json`, `*_boundary_plan.json`, `ssd/bench/mlsys_{greedy,preemption}_audit.py` |
| Full480/P2-only/TP2/edge | `round3/llama{2,3}_{full,p2,edge}_plan.json`, `tp2_plan.json` |
| 최종280검사 | `round3/regressions_final.log`, `run_regressions.sh` |
| 결과 그래프 | [round3/throughput_comparison.png](round3/throughput_comparison.png) |

이 문서 아래 부록에는 이번 branch의 commit 전체와 code/bench/test/dependency 변경 파일 전체를 적는다. 다른 서버에서 누락 여부를 파일 단위로 확인할 수 있다. Runtime source 기준은 위 `68d0a26`이며, 문서·감사 script를 추가하는 후속 commit은 해당 runtime 실험 revision을 바꾸지 않는다.



**19. Round4: 이 서버의 추가 구현·실험 (2026-10-07~08)**

이 절이 0–18절의 시점별 미완료 항목을 갱신한다. 시작 commit은 `696e92c`, 최종 runtime/bench/test commit은 `c0600ea`, 논문 기준은 `a82f7d2`다. 다른 서버의 branch를 가져오지 않았다. 모든 변경은 `/home/chokwans99/PSD-mlsys-coverage`, `feat/duet-mlsys-coverage`에 있다. Round4 상세 근거는 [REPORT](round4/REPORT.md), [수치표](round4/TABLES.md), [수식](round4/THEORY.md), [작업 로그](round4/NOTES.md)에 보존한다. 이 절에도 merge 판단에 필요한 구현·결과·제약을 함께 남긴다.

**19.1 사용자 요청 다섯 가지와 처리 범위**

| 요청 | 적용/검증 |
|---|---|
| Cap/EOS 경계 집계 제외 | 마지막 cap/clip sequence event를 AL에서 제외. TPS는 해당 batch-step의 token과 시간을 함께 제외. 전체 반환 TPS/원시 기록도 보존 |
| 논문 breakdown으로 P1/P2 숨김 확인 | 원본 Fig5 확인. 실제 proxy NCCL 완료·P1/P2 완료·target final model/accept-ready 시점을 계측. B1/B8, K4/2·8/4·2/1 비교 |
| 이전 tree 개선 적용 | Reach, gain allocation, frontier를 실제 공통 실행기에 연결. 기존 table transfer와 dense 모델8질문 calibration을 분리하여 full480 및 held-out472 평가 |
| 짧은/분기형 miss | 기본 chain2를 확인하고 chain1/chain4/star3/tree2x2를 독립 구현. B1/B8 full480, node 수를 맞춘 tree2x2–chain4 대조 |
| B1 G>M 직접 검토 | 사후 confidence pruning을 고정 generation prefix로 교체. P1 precompute/on-demand와 P2 모두 수정. 유한 분포 전수 감사·unit test·dense model smoke |

경계 제외 단위에 대한 선택 질문에는 응답이 없었으므로, 전체 요청 삭제 대신 마지막 event/영향 batch-step 제외를 가정한다고 안내한 뒤 진행했다. 이전 논문에서 긴 **입력**을 사전 제외한 것과, 상한에 도달한 **출력 결과**의 요청 전체를 지우는 것은 다르다. 원시 결과는 삭제하지 않는다.

**19.2 실험 범위와 GPU 배치**

- Full dense Llama2-7B/AMD135m(FP16), Llama3-8B/Qwama0.5B(BF16), RTX4090 8개. 첫 점검 시 모든 GPU가 비어 있었다.
- B8 Llama2는2,3 PIX, Llama3는4,5 PIX. B1 Llama2는0,1 NODE(같은 NUMA), Llama3는6,7 PIX. 같은 모델/조건의 상대 비교는 동일 pair를 유지했다. 다른 topology의 두 모델 절대 TPS 비교는 통제 실험이 아니다.
- 저장된 dataset의 전체480 question **첫 turn**. 여섯 task group각80개. 입력 cap512(잘린 입력 Llama2 141개/Llama3 136개), 출력 cap64, EOS 적용. 전체 question 목록과 전체 원문 token 사용을 구분한다. B8 주요 policy/miss는 T=.7, target seed3개. B1은 T=.7,1seed. 작은8/32/48질문 실행은 smoke/profile로 표시한다.
- 기본 exit21, K1/K2=4/2, P1/P2 G=M=8/4. Root의 `e(1-q)` 개선이나 새 위치 점수는 포함하지 않았다. 독립 tree continuation policy 비교다.
- Pass 사이 target RNG는 재설정하지만 draft RNG는 process 시작 seed0에서 이어진다. 동일 engine의3pass는 독립 process replicate가 아니다. 질문 단위 paired bootstrap2,000회, nominal95% CI를 사용하며 다중비교 보정은 하지 않았다.
- Detailed profile OFF인 full 성능 실행과 profile ON 진단을 구분했다. Full batch timing은 activeB==configuredB만 집계하고 capture/초기20step을 제외했다.

**19.3 실제 변경 파일과 merge 시 보존할 계약**

| 파일 | Round4 변경 | Merge 시 확인 |
|---|---|---|
| `engine/draft_runner.py` | miss JIT의 독립 깊이/폭; legacy B1 G>M prefix selection | 이미 sampling한 자식의 점수를 보고 그 자식을 버리는 cache를 되살리지 않기. 모든 sibling의 원 parent q 유지 |
| `helpers/batch_tree_draft.py` | source0 shallow tree wire/KV restore, config 기반 기본 miss, bulk metadata export, phase timing | Miss도 tree일 수 있음. Source0=chain이라고 가정하지 않기. P1/P2 arena는 별도이며 duplicate root는 P1 우선. 전체 build 후 응답하는 현재 barrier 계약 필요 |
| `helpers/p2_tree_executor.py` | `iter_rounds`에 정책 연결, mask/fanout fusion, parallel insertion 옵션 | 이전 `run_once` monkeypatch를 중복 적용하지 않기. Graph 내부 입력 버퍼가 replay마다 바뀌는 계약 유지 |
| `helpers/p2_tree.py` | all-frontier 선택 지원; diagnostic rerank 설명 정정 | q/proposal 및 sibling-order 보존. old rerank utility를 production에 다시 연결하지 않기 |
| `helpers/tree_expansion_policy.py` (신규) | q/reach, gain, frontier dispatch 및 frozen table 로딩 | G=M, global expansion 지원 범위 검사. q 자체를 추정 alpha로 치환하지 않기 |
| `helpers/tree_gain_allocation.py` (신규) | root별 작은 정수 partition 최적화 | 요청/root 간 budget을 섞지 않기. 기존 future-round reserve 유지 |
| `helpers/tree_fused_math.py` (신규) | exact mask packing, round-robin closed form | stable priority tie, 63bit ancestor word, invalid lane, mutable capture input 검사 |
| `helpers/tree_rerank_gpu.py` | 기존 utility의 lossless-safe 주장 제거 | GPU helper 자체를 새 lossless production selector로 오해하지 않기 |
| `helpers/batch_tree_forward.py` | target pre/post/final logits CUDA timing | Profile OFF에서 기존 실행 경로 유지 |
| `engine/step.py`, `llm_engine.py` | emitted/cap/clip/event ID, batch step별 time/token | Verifier의 진단 AL과 실제 반환 TPS를 구분. 토큰만 제외하고 시간을 남기는 불공정 비교 방지 |
| `bench/mlsys_coverage.py` | 경계 지표, G>M smoke 인자, frozen calibration 관측, GPU preflight, resolved env/source hash | Shared venv여도 이 checkout import. Preflight는 RNG 복구/메모리 해제 후 engine 초기화. 초기 compile cache와 후속 pass 구분 |
| `bench/mlsys_tree_observer.py` (신규) | Legacy B1의 full p/q passive snapshots | Diagnostic only. T>0, 평가 시만 켜고 throughput 결과에 사용하지 않기 |
| `tests/test_tree_round4.py`, `test_tree_fused_math.py` (신규), `test_output_accounting.py` | 수학/serving/graph/경계 계약 검사 | 새 서버 CUDA/Triton/backend에서 재실행 |

Tree reach는 `rho_parent × Π(앞 sibling 거절률) × alpha_hat`다. Gain은 `Σ rho_i gamma(c_i)`의 **현재 round 추정 이득**을 root별 예산에서 최대화한다. 전체 실제 AL의 전역 최적성은 주장하지 않는다. Calibration은 신경망 학습은 없지만 경험적 보정이다.

**19.4 핵심 correctness 판정**

- Legacy G>M의 옛 confidence pruning은 실제 proposal 법칙을 바꿀 수 있다.18개 유한 sampling 결과와 모든 accept/reject coin branch를 전수 계산한 예에서 target `(0.4,0.3,0.3)`가 `(0.4,0.31875,0.28125)`로 바뀌었고 TV=.01875였다. 고정 generation prefix는 정확히 target과 일치했다. 실제 LLM bias 빈도/크기 추정은 아니다.
- P1 precompute cache를 비활성화/무효화했고 on-demand P1/P2 모두 같은 prefix 규칙이다. G=M zero-copy는 유지한다. 양 phase/stale cache/qref remapping test와 두 dense model G>M smoke를 수행했다. 옛 논문 전체 G/M 설정은 원 로그로 확인해야 한다.
- 기본 miss는 **chain2**였다. `duet_jit_short=True`가 config 생성 중 env에 반영된다. Baseline source0의 실제 `valid_k=2`를 전수 확인했다. 명시적 chain2는 재현 반복이고 개선 실험으로 세지 않는다. 독립 chain4 대조군을 추가했고 초기 잘못된 추론/수정 기록도 남겼다.
- 새 shallow miss는 모든 ordered-WOR sibling을 유지하고 첫 sibling만 다음 깊이로 확장한다. 원 parent logits와 target tree mask, 수락 경로 KV를 함께 전달한다. SpecInfer-inspired 설계 검토이며 전체 SpecInfer 재현이 아니다.
- Fused mask/fanout과 bulk export는 tree 점수/예산을 바꾸지 않는다. Byte/정수 수준 parity, capture replay, 실제 full model output 대조를 수행한다. T0 miss smoke 최초 차이 하나는 독립 HF FP16 logit tie였으며 모든 shape의 bitwise 동일성을 보장한 것은 아니다.

**19.5 논문 timing과 이번 측정이 말하는 것**

원 Fig5는 dense70B/TinyLlama, Blackwell targetTP2+draft1GPU, B1, K8/4, exit56이다. 정규화된 hit timeline에서 P1 42%, proxy wait22%, P2 27%, target pre/post61%/24%, sync9%, sampling6%다. 현재7/8B + RTX4090의 비율이 같지 않다.

측정 마감은 `C1=a+D1`, `C2=max(C1,P)+D2`, `C1<=P`, `C2<=F_ready`다. `P`는 P1 뒤 wait 호출 시각이 아닌 독립 stream의 실제 recv 완료다. Model 완료 `F_model`도 별도 기록한다. All-hit만으로 P1/P2의 완료를 보장할 수 없다.

| 측정 조건 | P1→proxy 여유 중앙값 | P2→target ready 여유 중앙값 | 의미 |
|---|---:|---:|---|
| Llama2 B1 K4/2 | +6.13ms | +2.82ms | 두 phase 모두 충분히 들어옴 |
| Llama2 B1 K8/4 | +1.70ms | +0.65ms | 대체로 들어오지만 여유가 작아짐 |
| Llama3 B1 K4/2 | -0.30ms | +0.93ms | P1은 자주 proxy보다 늦음, P2 ready는 대부분 충족 |
| Llama3 B1 K8/4 | -11.09ms | -15.14ms | 이 pair에는 forward 예산이 과함 |
| Llama2 B8 K4/2 baseline | -5.43ms | -7.53ms | Draft가 숨지 않음 |
| Llama2 B8 K4/2 fused | -0.81ms | -0.71ms | 같은 AL 정책의 kernel overhead 감소 |
| Llama3 B8 K4/2 baseline | -11.67ms | -14.49ms | 더 작은 target/draft 시간 간격 + tree overhead |
| Llama3 B8 K4/2 fused | -7.16ms | -7.92ms | 개선됐지만 여전히 마감 초과 |
| Llama2/Llama3 B8 K2/1+bulk | +4.87/+0.95ms | +1.66/+1.79ms | Profile에서 ready 정시율100%; AL tradeoff는 full 데이터로 별도 평가 |

이 timing의 B1도 `SSD_BATCHED_TREE=1`인 unified 경로이며 legacy B1과 구분한다. B1은16질문/cap32, B8 주요 profile은48질문/cap64다. 표의 margin은 각 step에서 먼저 차이를 계산한 후 중앙값을 취했다. 서로 다른 step latency 중앙값을 합쳐 가상의 timeline을 만든 것이 아니다. [실제 step 그림](round4/figs/04_actual_step_timeline.png)과 [전체 timing 통계](round4/timeline.json)를 보존한다. 작은 시계 anchor 오차가 있어0근처 margin의 엄격한 보장을 주장하지 않는다.

**19.6 결과·채택 설정**

<!-- ROUND4_FINAL_RESULTS_BEGIN -->
**검증 완료 범위:** full dense 두 모델 pair에서 GPU job94개를 모두 완료했다. 그중62개 job은 전체480질문을 사용했고, target-seed pass는132회다. 전체178개 cell의 실제 반환3,614,764token을 event/step/counter와 대조한1,958개 집계 검사에서 불일치가 없었다. 할당 GPU에 외부 process가 겹친 기록도 없었다. 기본 경로와 fused+bulk+parallel 경로 각각288/288 회귀 검사가 통과했다.

**AL을 우선한 최종 검증 조합:** K1/K2=4/2, 기존 reach+gain+frontier table, miss chain4, fused mask/fanout + bulk export. Root 후보 수식은 그대로다. 아래 비교는 두 arm 모두 같은480질문/target seed2026·2027/draft startup seed0다. AL*는 cap/clip terminal event 제외, TPS*는 경계 batch-step의 token과 시간을 함께 제외했다. TPS는 후속 pass 값을 사용하며 괄호에는 제외 전 실제 반환 TPS를 함께 썼다.

| 모델/B8 | 기존 AL* → 조합 AL* | AL* 개선 | ΔAL* 95% CI | 기존 TPS* → 조합 TPS* (전체 TPS) |
|---|---:|---:|---|---|
| llama2 | 1.9808 → 2.0476 | +3.37% | [+0.0367, +0.0989] | 480.1 → 498.4 (484.5 → 513.3) |
| llama3 | 2.3736 → 2.5376 | +6.91% | [+0.1233, +0.2031] | 381.8 → 422.0 (382.2 → 411.3) |

이 조합의 AL 개선에는 **miss node2→4 확대 효과가 포함**된다. 이를 모두 tree 점수 개선으로 주장하지 않는다. 같은 corpus에서 구성 요소를 본 뒤 확인한 조합이며, 새로운 미관측 test set의 확증도 아니다. 검증한 후보 중 AL 우선 선택지이지 모든 parameter 조합의 최적성 증명은 아니다.

**구성 요소별 결론:**

1. **기존 tree 점수 개선은 Llama3에서 유효했다.** Miss2/G=M8·4를 고정한3pass 비교에서 historical reach+gain+frontier의 AL*는 Llama3 +1.91%(ΔCI [+.0172,+.0746]), Llama2 +0.63%(ΔCI [-.0096,+.0350])다. Llama2의 우위는 확정하지 않는다. Legacy B1에서도 Llama3 +2.72%, Llama2 +1.49%이며 후자는 CI가0을 포함한다.
2. **8질문 dense calibration은 필수가 아니었다.** Dense reach+gain+frontier는 q-path 대비 Llama3 +2.10%, held-out472에서 +2.12%였으나, historical table 자체와 비교한 추가 ΔAL* CI는 Llama3 [-.0277,+.0335], Llama2 [-.0128,+.0305]다. 보정 표본이 작고 수집B1/평가B8의 context 이동도 있어 새 보정이 더 좋다고 확정하지 않는다.
3. **Miss를 무조건 짧게 하거나 넓히는 것은 AL 목표에 맞지 않았다.** 기본 miss는 이미 chain2다. B8에서 chain4의 AL*는 Llama2 1.991→2.030, Llama3 2.382→2.469로 증가했다. 같은4-node의 tree2x2는 각각2.010/2.409로 chain4보다 낮았다. 이는 첫 sibling만 다음 깊이로 확장하는 이번 얕은 설계의 결과이며, 모든 token tree/SpecInfer의 열등성을 뜻하지 않는다. Star는 구조상 AL<=2라 Llama3 chain4 miss 조건부 AL≈2.31을 따라갈 수 없다.
4. **Llama3에서는 miss4에서도 tree 점수의 추가 AL 이득이 남았다.** Target seeds2026·2027로 맞춘 AL-only 분석에서 chain4 조건의 score 개선 ΔAL*는 Llama3 +.0888(CI [+.0499,+.1272]), Llama2 +.0120(CI [-.0182,+.0430])다. 조합 arm의 kernel 구현도 달라 TPS의 완전 factorial 비교는 아니며, 별도로 확인한 구현 동등성 범위에서 AL을 해석한다. 두 요소의 interaction CI는 양 모델 모두0을 포함하므로 양의 synergy를 확정하지 않는다. Llama2의2seed 결과만 골라3seed 주 분석보다 강한 결론을 내리지 않는다.

**알고리즘을 바꾸지 않은 실행 최적화:** K4/2·q-path·miss2의 전체480개 출력이 매 pass 모두 동일했다. Fused/bulk/parallel의 main 비교3개×2모델×2pass, 그리고 독립 draft seed1의 보조 pair 비교에서도 출력 동등성을 확인했다. B1 두 모델의 fused+bulk 비교도 전체480출력과 AL이 동일하다.

| 모델/B8 | 같은 AL을 유지한 관측 최고 설정 | 후속 pass 전체 TPS | 경계 제외 TPS* |
|---|---|---:|---:|
| llama2 | fused | 484.5 → 556.2 (+14.8%) | 480.1 → 551.2 |
| llama3 | fused_bulk_parallel | 382.2 → 456.0 (+19.3%) | 381.8 → 455.0 |

별도 GPU pair·draft seed1/target3030·3031의 fresh-engine 확인에서 fused+bulk 후속 pass TPS는 Llama2 470.0→532.3(+13.3%), Llama3 344.9→398.5(+15.6%)였다. Pair별 절대 TPS는 합치지 않는다. Llama2에서는 bulk/parallel을 더 켜도 fusion 단독보다 빠르지 않았고, Llama3의 parallel 추가 약2% 이득은 별도 process 반복으로 더 확인할 여지가 있다. B1 Llama2에서 같은 최적화의 TPS 이득이 거의 없었던 것은 이미 draft가 숨는 조건과 일관된다. Tree CUDA graph를 매번 다시 만들던 문제를 고친 것이 아니라 graph 안의 작은 kernel/metadata 비용을 줄였다.

**Phase 예산 축소의 원인 분리:** K2/1을 그대로 쓰면 default miss도1로 줄어든다. 따라서 miss2를 고정한 추가 full480 대조를 완료했다. 아래는 같은 fused+bulk, 같은 miss2이며 phase/node 예산만 K4/2·G8/4에서 K2/1·G4/2로 바뀐다.

| 모델/B8 | K4/2 AL* → K2/1 AL* | AL* 변화 | 후속 pass 전체 TPS |
|---|---:|---:|---:|
| llama2 | 1.9808 → 1.7501 | -11.64% | 549.9 → 545.3 |
| llama3 | 2.3736 → 2.0497 | -13.65% | 446.0 → 508.0 |

Llama3에서는 짧은 phase가 TPS를 더 높일 수 있지만 AL을 희생한다. **사용자가 정한 AL 우선 목적에는 K4/2 조합을 유지하는 쪽이 맞다.** Profile에서 두 phase가 시간 안에 들어온다는 조건만으로 최적 parameter가 정해지지 않는다. K를 바꾸면 depth/node/query shape와 target latency도 바뀌므로 이들을 포함한 AL/시간 비교가 필요하다. 여기서는 node budget도 함께 바뀌는 실제 설정을 비교했으며, 모든 K/N 조합을 sweep한 것은 아니다.

**적용 지침:** G>M proposal-law 수정과 실제 반환/경계 집계는 공통으로 채택한다. AL 우선 실행은 `*_budget_plan.json`의 `*_combined_chain4` job을 사용한다. 동일 AL에서 TPS를 우선하는 K4/2 실행은 Llama2의 `*_fused`, Llama3의 `*_fused_bulk_parallel` job이 이번 관측 최고다. 새 성능 옵션의 기본값은0으로 남겨 원 설정 재현/새 하드웨어 대조가 가능하게 했고, 검증한 plan은 필요한 옵션을 명시한다. 이 옵션들은 새 확률 threshold가 아니라 구현 ablation용이다. 다른 서버에서는 위 preset으로 시작하고 backend/hardware 변경 후 parity와 performance를 재확인한다.

**남은 범위:** 이 서버의 위94개 실행과 correctness/집계 검증은 완료했다. Dense70B·Blackwell targetTP2, 긴 출력1024/장문 전체 입력, 새 root 수식과의 결합, 다른 workload·독립 seed에서 작은 AL 이득의 재현은 후속이다. 이번 결과는 신규 SSD/Mirror-SD baseline 비교가 아니므로 그 대비 우위를 이 수치만으로 주장하지 않는다.
<!-- ROUND4_FINAL_RESULTS_END -->

**19.7 다른 서버 재현과 누락 방지**

1. 이 문서0절의 논문 기준, 다른 서버 HEAD/working diff, 이 branch 최종 commit을 먼저 기록한다. 다른 서버의 미push 수정은 이번 검증에 포함되지 않았다.
2. Round3의 공통 scheduler/attention/verifier/wire 수정과 Round4의 실행기·tree policy·sampling/KV 계약을 함께 merge한다. 특히 `draft_runner.py`, `p2_tree{,_executor}.py`, batch tree helpers가 충돌 가능성이 높다.
3. `G>M`에서 옛 confidence precompute/CPU/GPU rerank가 serving에 다시 연결되지 않게 한다. 다른 tree selector를 추가해도 이미 sampled token의 사후 생존 규칙은 별도 losslessness 검토가 필요하다. 다른 서버에서 P1 cache를 P2 완료 전에 독립 제공하도록 바꿨다면 bulk export의 지연이 availability를 바꿀 수 있으므로 그대로 합치지 않는다.
4. K1/K2, 실제 miss 깊이/폭, root fanout, G/M, calibration file hash, dtype, tokenizer-ID mapping, targetTP, GPU topology를 기록한다. 초기 env만 보지 않고 config 완료 후 값을 본다. 후반 report는 새 파일까지 포함한 runtime Python SHA를 기록한다. 초기 report에는 이 필드가 없고 당시 commit/working-diff SHA만 있으므로, 모두 동일한 완전 source snapshot을 저장했다고 주장하지 않는다. 최종 재현 기준은 `c0600ea`와 `round4/SOURCE_MANIFEST.json`이다. `round4/MODEL_MANIFEST.json`에는4개 full checkpoint의 config/tokenizer 및 모든 safetensors shard SHA256을 남겼다. AMD checkpoint는float32지만 실제 실행은FP16으로 정렬했다.
5. `archive_results.py --restore`로 gzip을 풀면 원시 JSON과 profile을 복구할 수 있다. SHA256 manifest로 대조한다. NPZ full-vocab calibration snapshot은 `round4/local_calibration_snapshots.tar`(약238MiB)로 별도 보관했다. 이 tar는 Git에 포함되지 않으므로 정확한 재보정이 필요하면 따로 복사한다. Compact audit/table/각 NPZ 및 tar SHA는 Git에 남긴다.
6. Plan의 옛 absolute model path를 그대로 실행하지 않는다. 아래 script는 새 경로를 넣고 historical optimization 기본값을 명시한다. RTX4090용 `SSD_CUDA_ARCH=8.9`도 새 GPU에 맞춰 변경한다.
7. GPU가 비었을 때 correctness gate→T0/.7 smoke→full corpus AL→latency/throughput 순서로 검사한다. Dataset/full output/seed/repetition을 바꾼 새 결과를 기존 숫자와 혼합하지 않는다.
8. 새로운 SSD/Mirror-SD 비교에는 공통 정확성 수정·실제 emission 집계·같은 resident batch 의미를 공통 적용한다. DUET tree 구현 최적화와 SSD 자체의 최적 파라미터는 각각 적절히 튜닝한다. 이번 tree ablation만으로 Mirror-SD 대비 새 end-to-end 우위를 주장하지 않는다.

```bash
# 이 checkout의 venv 사용. 아래 model/GPU 경로는 새 서버에 맞춘다.
python results/mlsys_coverage/round4/archive_results.py --restore
python results/mlsys_coverage/round4/analyze_results.py
python results/mlsys_coverage/round4/analyze_timeline.py
python results/mlsys_coverage/round4/make_tables.py
python results/mlsys_coverage/round4/make_figs.py

python results/mlsys_coverage/round4/relocate_plan.py \
  results/mlsys_coverage/round4/llama2_full_plan.json \
  --target /models/layerskip-llama2-7b --draft /models/AMD-Llama-135m \
  --output /new-results/llama2-plan.json
SSD_CUDA_ARCH=8.9 python ssd/bench/mlsys_campaign.py \
  --plan /new-results/llama2-plan.json --directory /new-results/llama2-full \
  --gpus 0,1 --port 33000

# AL 우선 조합만 재현하려면 이미 검증한 job 하나를 선택한다.
python results/mlsys_coverage/round4/relocate_plan.py \
  results/mlsys_coverage/round4/llama2_budget_plan.json \
  --job llama2_combined_chain4 \
  --target /models/layerskip-llama2-7b --draft /models/AMD-Llama-135m \
  --output /new-results/llama2-al-plan.json
# Llama3는 llama3_budget_plan.json / llama3_combined_chain4 사용.

CUDA_VISIBLE_DEVICES=0 MLSYS_PYTHON=/새환경/bin/python \
  bash results/mlsys_coverage/run_regressions.sh
# 추가 round4 tests는 ssd/에서 checkout을 import하도록 PYTHONPATH=. 사용
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=ssd python -m unittest \
  tests.test_tree_round4 tests.test_tree_fused_math
python results/mlsys_coverage/round4/audit_proposal.py
```

Local shared venv의 editable install이 `/home/chokwans99/PSD`를 가리키므로 PYTHONPATH/bench의 checkout 우선 import를 제거하지 않는다.

**19.8 이번 서버에서 끝난 범위와 후속 연구의 경계**

- 이번 목표는 correctness 수정과 두 dense pair의1차 full-corpus 판단이다. 모든 모델/정밀도/길이/하드웨어에서 TPS 전역 최적점을 찾은 것은 아니다.
- Dense70B+Blackwell targetTP2의 논문 환경 재현, 긴 출력1024, T0 전체 확장 matrix, 독립 process repetitions, exit/K/새 root 수식/새 tree 점수의 공동 최적화는 별도 후속 축이다.
- Full480 밖의 일반화,8질문 calibration의 안정성, nominal CI의 다중비교 한계, BF16/FP16 tie에 의한 greedy output 차이를 남긴다. 미실행 항목을 성능 검증 완료라고 표시하지 않는다.


**20. Round5 — 실제 온도와 개선 root 수식을 적용한 DUET/SSD 비교 (완료)**

최종 비교·추가 최적화 대조·안정성 재검증·회귀 검사·원본 복원 검증을 완료했다. 종합 결과는 [round5/REPORT.md](round5/REPORT.md), 실행 계획은 [round5/PLAN.md](round5/PLAN.md), 수식은 [round5/THEORY.md](round5/THEORY.md), 예외와 결정 근거는 [round5/NOTES.md](round5/NOTES.md)에 있다. 이 문서20.4–20.7절만 읽어도 결론·선택값·merge 및 재실행 범위를 확인할 수 있다.

- **구현 완료:** 실제 request의 target/draft T로 early-exit root 점수를 계산한다. 기존 T=1 고정은 cache 순위의 heuristic 설정이었으며, 최종 target verification의 확률은 기존에도 실제 T였다. 이번 변경을 과거 lossless 검증 오류의 수정이라고 해석하지 않는다. T=0은 생성/검증 argmax와 soft budget ranking을 분리한다.
- **이전 개선 통합:** 내부 context에서 `e(1-q)`, leaf에서 `e`, 제안된 sibling 제외 후 전체 허용 vocabulary 정규화, observed/overlap terminal mass의 `.75/.25` 혼합을 적용한다. Chain 혼합을 branching tree의 ordered-sibling ladder로 확장했다. 이는 proxy 추정 규칙이며 실제 target 확률이라는 주장이 아니다.
- **Tree selector:** Round4의 `reach_gain_frontier`와 고정 historical table을 사용한다. C=1/2에서도 기존 C=3 gain table의 prefix를 사용하도록 수정했다. 새 평가 결과로 table을 다시 fit하지 않았다.
- **실험 조건:** full32layer FP16 LayerSkip-Llama2-7B+AMD135m, full32layer BF16 LayerSkip-Llama3-8B+Qwama0.5B; T=.7, B1/B8. GPU2,3/4,5/6,7은PIX,0,1은NODE다. 비교쌍은 같은GPU에서 실행하며 topology별 절대TPS를 무조건 합치지 않는다.
- **선택/평가 분리:** 기존480first-turn corpus의6group별8개=48개로 선택한다. Full480과 이번 선택에서 제외한432개를 모두 보고한다. 432개는 과거 모든 연구에서 미관측이었다는 뜻이 아니다. Full은 보유480첫turn 전부이며 원 데이터셋 전체·560turn·무제한context가 아니다.
- **진행 순서:** SSD K/F 독립 탐색 → DUET exit anchor → phase depth → tree root/forward/node budget → 결합/인접 설정과 phase 재배분 → warm 비계측48 최종 후보 → 파라미터 동결 → 새 draft process 반복의full480 → 고정 알고리즘 구현 최적화 대조. AL 우선점과 처리량 우선점을 별도로 보존한다.
- **최종 정확성 근거:** trim0/stream0, trim1/stream0, trim1/stream1에서 각각300검사 통과, skip0. Actual T, ragged bonus, C1 chain 환원, sibling mass, live CUDA graph T 변경, BF16 기본 dtype에서도 T buffer float32를 검사했다. Full480 고정 알고리즘 대조의 두 pass 출력/수락 기록도 동일했고, 안정성 재검증의 stream ON/OFF4쌍도 각각6pass 동일했다.
- **예외 보존:** 자체 회귀 launcher의 empty CUDA_VISIBLE_DEVICES 처리 때문에 GPU0에서 SSD 한 조건과 겹친 결과를 timing 제외하고 깨끗한 반복으로 대체했다. Dtype 수정 중 duplicate keyword가 들어간 네 신규 worker는 실패 기록을 유지하고 수정 후 재시작했다. 실패는 알고리즘 성능 결과로 사용하지 않는다.
- **Merge 주의:** root 연구 branch의 `duet_proxy_source`와 여기의 `duet_root_source`는 수동으로 대응시켜야 한다. `root_policy.py`, `batch_tree_sampling.py`, `p2_tree.py`, `batched_proxy.py`, `batch_tree_verify.py`, `model_runner.py`, `verifier.py`, Config와 bench CLI를 함께 검토한다. 온도 buffer dtype, 실제 bonus 위치, ordered-WOR q 보존, terminal recursion depth>=miss depth를 떨어뜨리지 않는다.
- **최적성 범위:** 유한한 측정 영역과 이웃 탐색에서의 선택이다. 모든 수식/예산/하드웨어/요청분포에 대한 전역 최적성을 주장하지 않는다. Profiling TPS는 최종 성능 수치가 아니다.

**20.1 구현·merge 계약**

Round5 시작점은 `c4f05a6`, 현재 runtime 기준은 `641f5ce`다. 19절의 correctness 수정과 실행 최적화를 포함하는 branch 위에 작업했으며 다른 서버의 branch를 merge하지 않았다.

| 파일/영역 | 변경과 merge 시 확인할 내용 |
|---|---|
| `root_policy.py`, Config, benchmark CLI | `duet_root_source=complement`, `duet_root_normalization=full`, `duet_root_overlap_mix=.25`. 연구 branch의 `duet_proxy_source`와 이름을 수동 대응한다. P1은 draft 기반 root 점수, 새 `e(1-q)`는 proxy phase에 적용한다. |
| `verifier.py`, `batched_proxy.py`, `p2_tree.py` | 실제 target/draft 온도로 점수를 계산한다. T0 생성·검증은 greedy, 예산 순위만 soft score다. FP16/BF16 기본 dtype에서 graph를 만들어도 live temperature buffer는 FP32를 유지한다. |
| `batch_tree_sampling.py` | 이미 생성된 ordered-WOR sibling들을 반영한 두 개의 terminal-mass recursion 후 `.75/.25`로 혼합한다. Target acceptance에 후보 점수를 넣지 않는다. 이미 뽑힌 token의 proxy 비율을 사용한다는 뜻이며 실제 target 수락을 미리 관측한 값은 아니다. |
| `batch_tree_verify.py`, `root_policy.py` | Terminal recursion은 최소 `max(K1,K2,miss_depth)`를 포함한다. Ragged chain의 실제 bonus context에는 leaf 점수를 쓴다. Padding token/q를 실제 proposal처럼 빼거나 제외하지 않는다. |
| `tree_expansion_policy.py` | C1/2에서 C3 calibration의 prefix를 재사용한다. `reach_gain_frontier`와 historical table SHA를 고정한다. 이 정책의 `tree_beta`는 비활성임을 beta0/1 출력 동일성으로도 확인했다. |
| Config, `draft_runner.py` | Independent capacity를 가진 unified tree 경로만 K2>K1을 허용한다. Legacy split chain/B1 tree의 K2≤K1 guard를 무조건 제거하지 않는다. |
| `batch_tree_sampling.py` | `SSD_TREE_LADDER_TRIM=1`: 중복 clone, 쓰지 않는 마지막 draft 정규화와 proxy/complement의 마지막 residual 갱신을 제거한다. Exact verifier/residual-root의 correction 분포는 보존한다. 기본0. |
| `tree_proxy_stream.py`, `batch_tree_forward.py` | `SSD_BATCH_TREE_PROXY_STREAM=1`: TP1 proxy LM head·점수 계산과 target post를 겹친다. Default stream에서 fresh normalized hidden을 먼저 만든다. Graph-post가 바꿀 수 있는 원본 residual을 side stream에 넘기지 않는다. 기본0. |
| Proxy stream 완료/메모리 | Persistent completion event, `record_stream`, 분리된 graph pool, final logits 이후 acceptance/graph-input 재사용 이전의 완료 대기를 함께 보존한다. TP>1 collective head는 기존 stream에 남긴다. 이번 dense 성능 실험은 TP1이다. |
| `batch_tree_inputs.py`, `batch_tree_verify.py`, forward helpers | 선택적 prepare/accept/commit/tree shape/capture 진단. `SSD_PROFILE_DUET_DETAIL=1`은 진단용이며 비계측 최종 결과와 구분한다. 초기 screening에는 나중에 추가한 target/glue capture label이 없어 모든 capture가 제거됐다고 주장하지 않는다. |
| `tests/test_root_policy.py`, `tests/test_phase_budget.py`, `run_regressions.sh` | 수식 reference, chain 환원, sibling mass, 온도 buffer/graph 갱신, trim/stream 동등성, independent phase capacity 검사. 명시적으로 빈 `CUDA_VISIBLE_DEVICES`를 CPU-only로 보존한다. |

**20.2 선택·평가 절차와 파라미터의 실제 의미**

- 기본 온도는 `.7`, full32layer 모델 2쌍, B1/B8이다. 입력512/출력96에서 선택을 시작하고 warm48 후보 비교로 최초 설정을 동결했다. 전체480 확인은 입력512/출력128이다. B2/B4와 입력1024/출력256은 B8 처리량 설정의 **이전 실험**이며 각각 별도 최적점을 찾았다는 뜻이 아니다.
- SSD는 K={2,4,6,8}, fanout={1,3,5}와 경계·인접 설정을 독립 탐색했다. 공유 correctness/fast verifier/CUDA graph는 유지한다. DUET tree를 SSD에 강제하거나 mixed-miss AR 정책을 SSD에 몰래 추가하지 않았다. 이번 baseline은 benchmark의 `mode=ssd`; Mirror-SD proxy-only ablation을 새로 수행한 결과가 아니다.
- K는 직렬 forward 횟수, NV는 root별 continuation node 상한, C는 sibling 폭이다. P1 초기 root 수 U와 이후 forward 폭은 별개다. W의 기본값은 `proxy_fan_out*(max(K1,K2)+1)`이고 이후 draft 폭은 `draft_fan_out*(K1+1)`에 연결되므로 K를 바꾸면 파생 예산도 변할 수 있다. 이를 단일 forward 비용의 인과효과로 부르지 않는다.
- CLI exit는 0-based다. Exit21은 22개 layer 실행 후를 뜻한다. Common step origin에서 `C1=a+D1`, `C2=max(C1,P)+J+D2`. J는 두 준비 조건이 끝난 뒤 P2의 측정 시작까지의 전환·dispatch gap이다. J를 생략한 간단한 식은 D2가 이 비용까지 포함한다는 의미로 쓴다. P1-before-proxy와 P2-before-final-logits/ready를 모두 측정한다. P1이 proxy보다 늦다는 이유만으로 전체 예산을 실패 설정이라고 단정하지 않는다.
- Dynamic/eagle expansion에서는 beta allocator를 쓰지 않는다. Beta0/1 차이를 최적화 성과로 해석하지 않는다. Floor, root 수, forward 폭, 노드 상한, P2 예산, phase depth/exit는 실제 출력·비용 변화와 함께 기록한다.
- 구현 최적화의 동일 설정 대조 후, 바뀐 시간 창을 확인하기 위해 **튜닝48에서만** K1−1/K2−1/exit±2를 진단한다. 상위 두 후보와 stream/trim 기본점을 warm 비교하고, stream 기본점보다 최소2% 빠를 때만 변경하는 사전 규칙을 쓴다. 이2%는 유의성 검정이나 새 serving threshold가 아니다. 새 설정은 다시 full480 두 반복, 유지된 설정은 이미 측정한 대응 반복을 사용한다.
- 두 최종 process 반복은 target6100/6101·draft41 및 target6200/6201·draft42다. 각 process에서 첫 전체 pass는 warmup, 마지막 pass가 측정값이다. 반복마다 GPU쌍이 달라도 각 DUET/SSD 쌍은 같은 GPU·입력·온도·상한·seed를 사용한다. GPU0,1은NODE, 나머지 실험 쌍은PIX다. CPU affinity는 고정하지 않았으므로 작은 시간 차이를 강한 유의성 주장으로 확대하지 않는다.
- AL 우선점과 TPS 우선점을 별도로 보존한다. AL*은 cap/clip terminal event 제외, TPS*은 그런 event가 포함된 **batch step 전체**의 반환 token과 시간을 함께 제외한다. 질문 전체를 삭제한 통계는 아니다. 실제 반환 TPS도 보존한다. Token/step 비율과 step-time 비율로 TPS를 분해할 때는 동일한 timing 포함 step 집합을 사용한다.
- **B8 선택 안정성 재검증:** 짧은 warm48 실행 중 드문 장시간 step이 순위를 좌우한 사례를 발견했다. 기존 SSD/B1 후보에서는 같은 tail이 없었으나 L3B8의 최초 K3/1 후보까지 영향을 받았다. 기존 결정은 보존하고, B8만3회 예열+3회 측정의 pass별 전체TPS* 중앙값으로 재선택한다. L2는4개, L3는8개 DUET 후보와 각각 기존 SSDfast 대조를 사용한다. Slow step을 제거한 TPS를 최종값으로 쓰지 않는다. 후보·규칙·시드를 실행 전에 [PLAN.md](round5/PLAN.md)에 기록했다. `*_STABILITY_FROZEN.json`이 B8 잠정 postopt 선택을 대체하며, B1/AL 우선점은 유지한다. 새 설정만 full480 두 반복을 추가한다. 이 수정은 tuning 기록의 문제를 해결하기 위한 것이며 full480 성적을 보고 설정을 고르는 절차가 아니다.

**20.3 정확성·재현·실패 기록의 확인 방법**

- 최적화 full parity는 두 pass의 모든 출력 token과 step별 `(seq_id, step_id, accepted_len, emitted_len, valid_k, source, cache_hit, cap, clipped)` 기록을 대조한다. 매 full run의 모든 후보 확률 tensor를 저장한 검사는 아니며 kernel 단위 tensor 동등성 검사는 별도다.
- 전체 event, decode step, 실제 반환 token, cap, AL/TPS 요약값을 재집계해 불일치를 검사한다. AL 구간은 질문 단위 paired bootstrap4000회, 같은 질문의 process 반복은 같은 cluster다. TPS 두 반복 범위는 confidence interval이 아니다.
- 유효한 최종 비교에 외부 GPU process가 섞인 경우 자동 중단한다. 초기 GPU 회귀가 SSD 한 tuning 조건과 겹친 기록은 제외하고 재측정했다. Dtype 편집 도중의4개 import 실패, standalone proxy microbench의 scalar-temperature capture 실패도 삭제하지 않는다. 생산 경로는 persistent GPU FP32 temperature buffer를 사용한다.
- 원시 benchmark/profile JSON은 `round5/archives`에 gzip으로 보존하고 SHA256과 원래 modification time을 manifest에 기록한다. Restore 시 mtime도 복원하므로 retry profile을 최신 mtime으로 선택하는 분석이 달라지지 않는다. 완료 전 active 결과를 final archive로 취급하지 않는다.
- 다른 서버의 checkout 경로가 달라도 offline 분석은 복원된 **현재 checkout의 profiles**를 우선한다. Raw report의 원래 절대 경로는 provenance로 보존하며, 실행 계획을 옮길 때만 `relocate_plan.py`로 새 파일을 생성한다.
- 재현할 때 `make_plans.py`를 기존 결과 위에 다시 실행하지 않는다. 최종 `FINAL_PRESETS.json`과 `*_RECOMMENDED_PLAN.json`을 복사하고 `relocate_plan.py`로 model/GPU/저장 경로를 바꾼다. 원본 plan·실패 기록·동결 결정 파일을 덮어쓰지 않는다.
- 연구 서버 merge 시 19절의 proposal-law/KV/scheduler/wire 수정과 이 절의 root 온도·분포·stream 계약을 함께 검토한다. `SSD_TREE_LADDER_TRIM`, `SSD_BATCH_TREE_PROXY_STREAM`의 기본값은0이며 최종 검증 preset이 필요한 값을 명시한다. 사용되지 않는 legacy fallback까지 새 수식이 적용됐다고 추정하지 않는다. 측정 경로는 JIT/unified tree다.

**20.4 최종 처리량 설정과 SSD 비교**

**결론:** Llama2 B1의 AL/TPS 개선을 확인했다. Llama3 B1의 AL 차이는 명확하지 않고 SSD가 더 빨랐다. B8은 두 모델 모두 추가 최적화 후에도 SSD가 우세했다. 이 결과로 DUET가 모든 B/모델에서 SSD보다 빠르다고 주장할 수 없다. 새 수식·tree·예산·구현이 결합된 시스템 비교이므로 특정 root 수식 하나의 인과효과로 해석하지 않는다.

| 모델 | B | DUET K1/K2·exit index | SSD K/F | DUET/SSD AL* | ΔAL*95% 구간 | DUET/SSD TPS* 두 반복 |
|---|---:|---|---|---|---|---|
| Llama2+AMD | 1 | 11/2·21 | 4/7 | 2.3460 / 2.1425 | [+0.1629,+0.2465] | 1.088x / 1.133x |
| Llama2+AMD | 8 | 3/2·21 | 4/7 | 2.1073 / 2.1508 | [−0.0794,−0.0094] | 0.885x / 0.876x |
| Llama3+Qwama | 1 | 4/2·16 | 4/7 | 2.6900 / 2.6611 | [−0.0119,+0.0714] | 0.939x / 0.966x |
| Llama3+Qwama | 8 | 3/1·26 | 4/1 | 2.2843 / 2.6873 | [−0.4414,−0.3647] | 0.786x / 0.781x |

Exit는0-based이며 exit21은22개 layer 실행 후다. 모든 선택값은 tuning48에서 정했다. 최종480에 이48개가 포함되므로, 선택 제외432개 결과도 [POSTOPT_TABLES.md](round5/POSTOPT_TABLES.md)에 따로 제시했다. 432개에서도 위 판단이 유지된다. TPS 범위는 두 process/GPU쌍 반복 결과이며 confidence interval이 아니다.

| 모델/B | C | P1/P2 node 상한 | P1 roots | draft fanout | 실제 P2 W | trim/stream |
|---|---:|---|---:|---:|---:|---|
| L2/B1 | 3 | 16/4 | 2 | 2 | 12 | 1/1 |
| L2/B8 | 3 | 6/4 | 1 | 2 | 4 | 1/0 |
| L3/B1 | 3 | 8/4 | 2 | 2 | 5 | 1/1 |
| L3/B8 | 1 | 3/1 | 2 | 2 | 4 | 1/0 |

공통은 e(1−q), full-vocabulary 정규화, terminal-mass overlap mix .25, reach_gain_frontier, miss chain4, fused/bulk ON, parallel-insert OFF다. P2 proxy/confidence floor .01/.03, P1 start/confidence floor0/0이다. Beta .5는 이 selector에서 비활성이다. C1은 각 cache root의 continuation이 chain 형태라는 뜻이다. 최종 전체 CLI/env/engine 설정은 [FINAL_PRESETS.json](round5/FINAL_PRESETS.json), 즉시 재실행할 두 방법은 `round5/*_RECOMMENDED_PLAN.json`에 저장했다.

**20.5 AL 우선 비교와 추가 workload**

AL이 목표라면 위 TPS 우선 설정을 유일한 추천으로 읽지 않는다. 다음은 튜닝에서 별도로 선택하고 두 process full480으로 확인한 AL 우선점이다. 추가 구현 옵션 이전의 동일 목적 비교이며, 그중 B8 trim 대조도 별도 보존했다.

| 모델/B | DUET K1/K2·exit·C | SSD K/F | DUET/SSD AL* | ΔAL*95% 구간 | TPS* 비율 |
|---|---|---|---|---|---|
| L2/B1 | 12/2·21·3 | 6/5 | 2.3651 / 2.2594 | [+0.0583,+0.1519] | 1.088x / 1.088x |
| L2/B8 | 6/2·21·3 | 8/5 | 2.3191 / 2.3102 | [−0.0361,+0.0524] | 0.758x / 0.769x |
| L3/B1 | 6/4·16·3 | 8/1 | 2.9768 / 3.0114 | [−0.1012,+0.0318] | 0.985x / 0.963x |
| L3/B8 | 6/3·30·3 | 8/3 | 2.9003 / 3.0640 | [−0.2280,−0.0995] | 0.832x / 0.852x |

최초 B8 처리량 설정을 이전한 full480 B2/B4 비교도 했다. L2의 TPS* 비율은1.012x/0.954x, L3는0.941x/0.849x다. 입력1024/출력256의 B8 이전 실험은 L2 0.880x, L3 0.696x다. **B2/B4·긴 길이마다 별도 최적점을 찾았거나 마지막 B8 재선택을 다시 적용한 결과가 아니다.** [FOLLOWUP_TABLES.md](round5/FOLLOWUP_TABLES.md)에 정확한 AL/TPS·계획 경로가 있다.

**20.6 Breakdown·추가 최적화로 무엇을 확인했는가**

1. **70B의 시간 비율을 그대로 쓸 수 없었다.** K4/2 anchor의 B1 P1은 L2약4.1ms, L3약11.2ms였다. B8 proxy 경로도 약1.0ms와5.2ms로 달랐다. L2B1은 남는 P1 창에 K1을 늘릴 수 있었다. L3B8은 exit만 옮겨도 P1/P2가 숨겨지지 않아 phase/width/node 예산을 함께 줄여야 했다. 이는 모델·vocabulary·kernel 조건이 함께 바뀐 관측이다.
2. **P1이 proxy보다 늦어도 전체 build는 숨겨질 수 있다.** 최종 L2B1의 P1 slack은−0.51ms지만 P2-before-final은+1.15ms다. L3B1도−1.58ms/+0.82ms다. B8의 최종 P2-before-next-request는 L2약+0.42ms, L3약−0.50ms다. 별도 median을 합쳐 하나의 timeline으로 해석하지 않으며, 작은 cross-process slack은 anchor 오차까지 감안해야 한다.
3. **높은 hit만으로 AL/처리량을 설명할 수 없다.** L3B8은 DUET hit82.7% 대 SSD35.5%지만 DUET P2의 clean-event 비중30.7%, 해당AL1.611이다. 전체AL*=source 비중×조건부AL의 합이다. 조건부 차이는 도달 문맥과 tree 깊이도 달라 단일 원인의 인과효과가 아니다. 완전한 batch hit는 DUET조차21–22% 정도다.
4. **B8 열세를 출력량과 시간으로 분해했다.** L2B8은 timing 포함 step당 반환량이1.6–2.8% 적고 step 시간이약11% 길다. L3B8은 반환량14–15% 감소와 step 시간약9% 증가가 겹쳤다. Query utilization은 각각약79%/74%지만 이 비율을 같은 비율의 TPS 손실로 바꿔 해석하지 않는다.
5. **Ladder trim은 작은 개선이었다.** 54개 reference CPU/GPU 사례와8개 full-vocabulary graph shape에서 tensor 동일성을 확인했다. Kernel-only1.128–1.697x와 달리 full480 TPS 효과는약0.3–1.3%였다. 최종 preset에서ON이며 기능 기본값은OFF다.
6. **Proxy stream은 B1/B8 결과가 달랐다.** 동일 출력·설정 대조에서 B1은0.3–1.5% 개선, B8은2.6–7.3% 악화했다. B8에는 proxy 도착/P2 완료 지연이 동반됐고 자원 경쟁과 일치하지만 hardware counter로 분리한 원인 증명은 아니다. 최종 B1만ON, B8은OFF다.
7. **재튜닝의 이득을 과장하지 않았다.** L2B1 K11은 warm48에서 선택됐지만 이미 trim+stream을 켠 K12 대비 full480 TPS가+0.30%/−0.43%여서 추가 우위를 확정하지 못했다. AL 우선점은K12다. L3B8은 K3/2·nodes3/2→K3/1·nodes3/1로 TPS7.4–9.1% 증가, AL5.1% 감소였다. SSD를 넘지는 못했다. Exit24/26 warm median 차이도약1.1%여서 유일한 최적 layer를 증명한 것은 아니다.
8. **짧은 튜닝의 timing tail 문제를 보정했다.** B8 최초/잠정 선택에 드문 긴 step이 영향을 주어3warm+3measured pass의 중앙값으로 재검증했다. 원래결정/모든느린step/잠정full 결과도 보존했다. 별도 계측에서는5×median 초과682step 중675개가 동일/직전 capture와 겹쳤다. 비계측의 개별 tail까지 모두 capture라고 단정하지 않는다. 실행 전 protocol은 commit `474eabb`와 PLAN.md에 남겼다.

상세 진단은 [BREAKDOWN_TABLES.md](round5/BREAKDOWN_TABLES.md), SSD 독립 탐색70조건은 [SSD_BREAKDOWN.md](round5/SSD_BREAKDOWN.md), 최종 그림은 [07_final_selected_comparison.png](round5/figs/07_final_selected_comparison.png), stream 비교 실제 timeline은 [06_proxy_overlap_timeline.png](round5/figs/06_proxy_overlap_timeline.png)다. 그림별 PDF도 같이 보존했다.

**20.7 검증 완료 범위·재현·merge 이후의 판단**

- 완료487실행, 총691cell, full480실행70개를 원본 counter와 대조했다. 실패한4개 import 시도는 성공으로 세지 않았다. 초기 GPU overlap1건은 별도 timing 제외 후 재실행했다. [INVENTORY.json](round5/INVENTORY.json), [EXCLUSIONS.json](round5/EXCLUSIONS.json), [ANALYSIS_COMPLETE.json](round5/ANALYSIS_COMPLETE.json)에서 확인한다.
- 최종 GPU 회귀는3구성×300개 전부 통과/skip0. 고정 알고리즘의 fused/bulk 대조2쌍, trim10쌍, stream8쌍에서 full480의 두 pass 출력/수락 기록을 대조했다. 추가 안정성 stream4쌍은 각각6pass 동일하다. [STABILITY_PARITY.json](round5/STABILITY_PARITY.json), [FOLLOWUP_RESULTS.json](round5/FOLLOWUP_RESULTS.json)에 검사 범위를 저장했다.
- Benchmark/profile 원본1169개, 원본합계6440.9MiB를613.8MiB gzip으로 보존했다. **빈 임시 checkout에 실제 복원**하여 전체 SHA256·크기·mtime 검사를 통과했다. 다른 경로에서 retry/최종 선택 profile의 분석값도 동일했다. [ARCHIVE_RESTORE_AUDIT.json](round5/ARCHIVE_RESTORE_AUDIT.json). 실패/prototype 기록도 보존하며 gzip만 Git에 넣고 재생성 가능한 큰 raw JSON은 ignore한다.
- Runtime 최종 변경은 `641f5ce`, 새 B8 안정성 protocol commit은 `474eabb`다. Source hash와 패키지 버전은 [SOURCE_MANIFEST.json](round5/SOURCE_MANIFEST.json), [ANALYSIS_ENV.json](round5/ANALYSIS_ENV.json)에 있다. 10/08 재조회한 원격 논문 branch 및 merge-base는 여전히 `a82f7d2`다. 결과 commit은 이 문서를 포함한 branch HEAD에서 확인한다.

다른 서버에서 **수치만 재생성**하려면 checkout 후 archive를 복원하고 분석 coordinator를 실행한다. 완료 marker도 보존되어 있어 새 GPU 실험을 시작하지 않는다.

```bash
python results/mlsys_coverage/round5/archive_results.py --restore
python results/mlsys_coverage/round5/finish_analysis.py
```

**새 서버에서 측정**할 때는 원본 plan을 덮어쓰지 않고 실제 model 경로를 넣어 새로운 plan·결과 폴더를 만든다. 아래는 L3/B8 SSD+DUET 재실행 예다. GPU쌍과 port는 해당 서버의 idle topology에 맞춘다.

```bash
python results/mlsys_coverage/round5/relocate_plan.py \
  results/mlsys_coverage/round5/llama3_b8_RECOMMENDED_PLAN.json \
  --root "$PWD" --target /path/to/layerskip-llama3-8B \
  --draft /path/to/Qwama-0.5B-Instruct \
  --output /tmp/duet-llama3-b8-replay.json
python ssd/bench/mlsys_campaign.py \
  --plan /tmp/duet-llama3-b8-replay.json \
  --directory results/mlsys_coverage/new_server/llama3_b8 \
  --gpus 0,1 --port 35000
```

반대 process seed/GPU 반복까지 재현하려면 `POSTOPT_RESULTS.json`의각replicate path에 대응하는 원본 `plan.json`을 `--job`으로 골라 같은 방식으로 복사한다. Recommended plan은 각조건rep0의 두 방법을 담는다. 새로운 서버·길이·모델에서는 이 preset을 시작점으로 두고 breakdown을 다시 측정해야 한다. 현재 탐색 script의 exit 상한31은32layer 새모델용이므로70B에 그대로 적용하지 않는다.

**이번 계획의 미실행 필수 job은 없다.** 다음은 이번 결과 밖의 연구/최적화 후보이며 성능이 입증됐다고 쓰면 안 된다: target query bucket/padding 개선, C1 continuation을 위한 chain 전용 실행 경로, hit/miss 분리 스케줄, AL을 보존하면서 P2 노출을 줄이는 구성, 모델별 새 tree calibration, dense70B/Blackwell TP2 재현, 출력1024·더 넓은 독립 workload/온도, B2/B4 개별 재튜닝. 기존 packed-tree 실험은 더 느렸으므로 무조건ON하지 않는다. 다른 서버 merge에서는20.1의 q-law·실제 온도·wire·KV·stream memory/event 계약과19절의 correctness 변경을 함께 보존한다.

**부록 A. Systems branch commit 전체 — 논문 기준 이후**

`a82f7d2..641f5ce`의 전체 commit이다. 결과 문서의 후속 commit은 branch log에서 확인한다. `94676ae`의 일시적 import 오류는 `a329823`에서 수정했으므로 앞 commit만 따로 가져오지 않는다.

| Commit | 작업 |
|---|---|
| `503fa96` | Fix ragged greedy recovery and add MLSys coverage checks |
| `317741d` | Fix paged prefix prefill and bound batch admission; reduce verify synchronization |
| `78d28d8` | Keep a prefill query for fully cached prompts and record batch padding |
| `f50e74a` | Extend captured proxy ranking and exit-replica overlap to batched DUET |
| `aabc246` | Add branch-free stochastic verification and isolated throughput campaigns |
| `13f42d8` | Keep colocated model norm constants capturable and audit batch numeric differences |
| `7260f28` | Count emitted decode tokens and preserve output budgets across preemption |
| `7d3e52b` | Specialize greedy samplers in captured engines and remove temperature readbacks |
| `e2dd89f` | Fix completed KV block hashes and audit preemption on dense models |
| `a3572e6` | Record dense-model batching and greedy throughput evidence and handover |
| `721df75` | Add packed chain verification and mixed-miss direct-target experiment |
| `0974b14` | Enable greedy dynamic tree verification and fix Qwama P1 workspace |
| `14b4b81` | Add experimental batched tree round executor and dense draft diagnostics |
| `6c1df2f` | Add opt-in reuse of validated tree glue topology and extend regression suite |
| `893a6b5` | Audit batched draft roots against independent full-precision reference |
| `715d961` | Distinguish failed workers from stale running result files |
| `ec8bc97` | Record full-model round-two results and remaining batched-tree scope |
| `7af3f5c` | Connect request-isolated batched dynamic tree serving and exact verification |
| `33dda1e` | Share batch tree attention scratch and capture root scoring and acceptance |
| `72475b8` | Capture per-request tree input preparation across the batch |
| `d7fea9f` | Use short target query buckets for batches with smaller tree views |
| `c8f7b94` | Respect configured sibling capacity in batch verification and test stochastic walks |
| `413afca` | Include batched serving checks in the full regression suite |
| `800f011` | Ignore exhausted zero-probability WOR padding in batched acceptance |
| `225d1ab` | Preserve greedy tie breaking inside mixed-temperature batches |
| `30d8851` | Add opt-in ragged tree target queries with re-planned captured attention |
| `b90f118` | Preserve generation prefixes when capping batched verification trees |
| `68d0a26` | Record completed batched tree serving, full-model comparisons and handover |
| `696e92c` | Document merge provenance, baseline fairness, and remaining DUET work |
| `c0600ea` | Preserve tree proposal law and evaluate budget-aware DUET execution |
| `c4f05a6` | Record full-corpus DUET tree, miss, timing, and optimization evidence |
| `f041ebd` | Use actual temperatures and improved token/position cache-root scores |
| `8ddfe62` | Record optional tree depth and utilization for parameter diagnosis |
| `baa1c2c` | Reuse measured ordered-sibling gain prefixes for narrower trees |
| `94676ae` | Keep live temperature buffers in float32 during model-dtype graph capture |
| `a329823` | Correct duplicate dtype keyword in mixed-temperature verification |
| `58326e9` | Cover model-dtype temperature capture and preserve CPU-only regression selection |
| `81ea777` | Allow independent phase depths in the unified DUET tree service |
| `c02acee` | Expose optional tree preparation, acceptance, and commit timing spans |
| `69192a5` | Add opt-in elimination of unused tree ladder copies and terminal work |
| `cbdc41f` | Add opt-in overlap of batched tree proxy scoring and target post layers |
| `709c95b` | Reuse the proxy completion event across batched tree steps |
| `1db69b5` | Overlap the TP1 proxy head from independent normalized hidden storage |
| `641f5ce` | Label optional target and glue graph captures in detailed tree profiles |

**부록 B. Code·bench·test·dependency 변경 파일 전체**

`a82f7d2..641f5ce -- ssd results/mlsys_coverage/run_regressions.sh`의 numstat다. 결과·분석 script·artifact 목록은 각 round inventory/manifest와 Git에 보존한다. Add/delete 줄 수는 기여 규모나 성능 지표가 아니다.

| 파일 | 추가 | 삭제 |
|---|---:|---:|
| `results/mlsys_coverage/run_regressions.sh` | 25 | 0 |
| `ssd/bench/batched_tree_microbench.py` | 265 | 0 |
| `ssd/bench/mlsys_campaign.py` | 124 | 0 |
| `ssd/bench/mlsys_coverage.py` | 319 | 0 |
| `ssd/bench/mlsys_greedy_audit.py` | 57 | 0 |
| `ssd/bench/mlsys_preemption_audit.py` | 50 | 0 |
| `ssd/bench/mlsys_summarize.py` | 84 | 0 |
| `ssd/bench/mlsys_tree_observer.py` | 38 | 0 |
| `ssd/bench/mlsys_verify_bench.py` | 55 | 0 |
| `ssd/pyproject.toml` | 1 | 0 |
| `ssd/ssd/config.py` | 28 | 11 |
| `ssd/ssd/engine/block_manager.py` | 10 | 1 |
| `ssd/ssd/engine/draft_runner.py` | 135 | 133 |
| `ssd/ssd/engine/helpers/batch_tree_common.py` | 86 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_draft.py` | 434 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_forward.py` | 166 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_inputs.py` | 68 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_roots.py` | 61 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_sampling.py` | 224 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_verify.py` | 142 | 0 |
| `ssd/ssd/engine/helpers/batched_proxy.py` | 90 | 0 |
| `ssd/ssd/engine/helpers/batched_tree_executor.py` | 154 | 0 |
| `ssd/ssd/engine/helpers/cudagraph_helpers.py` | 65 | 30 |
| `ssd/ssd/engine/helpers/p1_tree.py` | 5 | 7 |
| `ssd/ssd/engine/helpers/p2_tree.py` | 111 | 35 |
| `ssd/ssd/engine/helpers/p2_tree_executor.py` | 42 | 13 |
| `ssd/ssd/engine/helpers/packed_tree_forward.py` | 139 | 0 |
| `ssd/ssd/engine/helpers/packed_verify.py` | 110 | 0 |
| `ssd/ssd/engine/helpers/root_policy.py` | 99 | 0 |
| `ssd/ssd/engine/helpers/tree_expansion_policy.py` | 74 | 0 |
| `ssd/ssd/engine/helpers/tree_fused_math.py` | 66 | 0 |
| `ssd/ssd/engine/helpers/tree_gain_allocation.py` | 71 | 0 |
| `ssd/ssd/engine/helpers/tree_host_topology.py` | 20 | 0 |
| `ssd/ssd/engine/helpers/tree_proxy_stream.py` | 29 | 0 |
| `ssd/ssd/engine/helpers/tree_rerank_gpu.py` | 4 | 1 |
| `ssd/ssd/engine/llm_engine.py` | 32 | 46 |
| `ssd/ssd/engine/model_runner.py` | 62 | 11 |
| `ssd/ssd/engine/scheduler.py` | 10 | 5 |
| `ssd/ssd/engine/sequence.py` | 9 | 3 |
| `ssd/ssd/engine/speculator_async.py` | 19 | 0 |
| `ssd/ssd/engine/step.py` | 13 | 1 |
| `ssd/ssd/engine/verifier.py` | 93 | 80 |
| `ssd/ssd/layers/attention.py` | 24 | 14 |
| `ssd/ssd/layers/layernorm.py` | 9 | 0 |
| `ssd/ssd/layers/sampler.py` | 8 | 2 |
| `ssd/ssd/utils/misc.py` | 17 | 0 |
| `ssd/ssd/utils/verify.py` | 31 | 13 |
| `ssd/ssd/utils/verify_fast.py` | 42 | 0 |
| `ssd/tests/test_b_gt1_jit_subset.py` | 7 | 0 |
| `ssd/tests/test_b_gt1_m2.py` | 12 | 2 |
| `ssd/tests/test_b_gt1_m4.py` | 19 | 5 |
| `ssd/tests/test_b_gt1_m6_verify_window.py` | 1 | 0 |
| `ssd/tests/test_batch_tree_serving.py` | 340 | 0 |
| `ssd/tests/test_batched_proxy.py` | 60 | 0 |
| `ssd/tests/test_batched_tree_executor.py` | 89 | 0 |
| `ssd/tests/test_cached_prefill.py` | 90 | 0 |
| `ssd/tests/test_colocated_norm.py` | 34 | 0 |
| `ssd/tests/test_greedy_sampler.py` | 52 | 0 |
| `ssd/tests/test_greedy_tree.py` | 42 | 0 |
| `ssd/tests/test_greedy_verify.py` | 61 | 0 |
| `ssd/tests/test_model_pair_contract.py` | 50 | 0 |
| `ssd/tests/test_output_accounting.py` | 97 | 0 |
| `ssd/tests/test_p1_dynamic_tree.py` | 1 | 1 |
| `ssd/tests/test_p2_tree_alloc.py` | 19 | 3 |
| `ssd/tests/test_packed_verify.py` | 90 | 0 |
| `ssd/tests/test_phase_budget.py` | 54 | 0 |
| `ssd/tests/test_root_policy.py` | 189 | 0 |
| `ssd/tests/test_stochastic_verify.py` | 65 | 0 |
| `ssd/tests/test_tree_fused_math.py` | 98 | 0 |
| `ssd/tests/test_tree_host_topology.py` | 22 | 0 |
| `ssd/tests/test_tree_round4.py` | 109 | 0 |
| `ssd/uv.lock` | 26 | 0 |

**부록 C. 주석별 답변 위치**

| 사용자 주석 | 이 문서의 답변 |
|---|---|
| 1 | 1·14·16절: 여섯 수정의 출처/노출 조건, SSD 공통 적용과 공정 비교 |
| 2 | 2절: dtype/wire/workspace |
| 3 | 3절: 과대평가 방향과 재집계 |
| 4 | 4절: 같은 batch의 query 차이와 packed 표현 |
| 5 | 5절: mixed miss, AR fallback, 추가 threshold 여부 |
| 6 | 6·19·20절: 기존 논문 profile과 새 full 모델의 phase deadline/SSD 튜닝 |
| 7 | 7절: soft score와 deterministic proposal 분리 |
| 8 | 8·9·19·20절: 이전 tree 연구와 실행기 재사용, full 모델 결합 검증 |
| 9 | 10절: serving 전체 연결의 의미 |
| 10 | 10절: miss JIT chain과 K2/Kmax |
| 11 | 0·10절: 기존 B1 tree attention/최신 원격 기준 확인 |
| 12 | 11절: 기존 ladder 보존과 새 경로 수정 범위 |
| 13 | 12·15·19·20절: graph 재사용, 실제 실행 최적화 대조, 남는 구조적 비용 |
| 14 | 8절: 이전 selector 연구의 online/offline/미실행 구분 |
