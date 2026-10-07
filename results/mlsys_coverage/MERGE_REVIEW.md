**DUET MLSys 확장 — 통합 변경 기록, 질문 답변, 다른 서버 merge 검토표**

작성일: 2026-10-07. 이 문서는 이번 systems branch의 작업을 한곳에서 검토하기 위한 기준 문서다. 과거 보고서의 시점별 미완료 설명은 역사적 기록이며, 현재 상태는 이 문서와 round3 결과로 판단한다. 이번 문서 작업에서는 source/history와 저장된 실험을 재검토했고, 새로운 GPU 성능 실험이나 runtime 변경은 하지 않았다.

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
- 이번 branch는 논문 기존 root/continuation 점수를 유지했다. `e(1-q)`, 새로운 위치 점수, calibrated reach/gain selector를 이번 성능 수치에 포함하지 않았다.

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

**15. 남은 작업과 우선순위 — 현재 시점의 완료/미완료 구분**

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

**17. 현재 기본값·지원 범위·재현 진입점**

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


**부록 A. Systems branch commit 전체 — 논문 기준 이후**

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

**부록 B. Code·bench·test·dependency 변경 파일 전체**

`a82f7d2..68d0a26 -- ssd`의 numstat다. 결과 artifact 전체 목록은 각 round inventory에 보존한다. Add/delete 줄 수는 기여 규모나 성능 지표가 아니다.

| 파일 | 추가 | 삭제 |
|---|---:|---:|
| `ssd/bench/batched_tree_microbench.py` | 265 | 0 |
| `ssd/bench/mlsys_campaign.py` | 124 | 0 |
| `ssd/bench/mlsys_coverage.py` | 237 | 0 |
| `ssd/bench/mlsys_greedy_audit.py` | 57 | 0 |
| `ssd/bench/mlsys_preemption_audit.py` | 50 | 0 |
| `ssd/bench/mlsys_summarize.py` | 84 | 0 |
| `ssd/bench/mlsys_verify_bench.py` | 55 | 0 |
| `ssd/pyproject.toml` | 1 | 0 |
| `ssd/ssd/config.py` | 11 | 9 |
| `ssd/ssd/engine/block_manager.py` | 10 | 1 |
| `ssd/ssd/engine/draft_runner.py` | 93 | 36 |
| `ssd/ssd/engine/helpers/batch_tree_common.py` | 86 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_draft.py` | 345 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_forward.py` | 133 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_inputs.py` | 68 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_roots.py` | 61 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_sampling.py` | 203 | 0 |
| `ssd/ssd/engine/helpers/batch_tree_verify.py` | 120 | 0 |
| `ssd/ssd/engine/helpers/batched_proxy.py` | 83 | 0 |
| `ssd/ssd/engine/helpers/batched_tree_executor.py` | 154 | 0 |
| `ssd/ssd/engine/helpers/cudagraph_helpers.py` | 65 | 30 |
| `ssd/ssd/engine/helpers/p1_tree.py` | 5 | 7 |
| `ssd/ssd/engine/helpers/p2_tree.py` | 58 | 14 |
| `ssd/ssd/engine/helpers/p2_tree_executor.py` | 25 | 9 |
| `ssd/ssd/engine/helpers/packed_tree_forward.py` | 139 | 0 |
| `ssd/ssd/engine/helpers/packed_verify.py` | 110 | 0 |
| `ssd/ssd/engine/helpers/tree_host_topology.py` | 20 | 0 |
| `ssd/ssd/engine/llm_engine.py` | 23 | 46 |
| `ssd/ssd/engine/model_runner.py` | 59 | 9 |
| `ssd/ssd/engine/scheduler.py` | 10 | 5 |
| `ssd/ssd/engine/sequence.py` | 9 | 3 |
| `ssd/ssd/engine/speculator_async.py` | 19 | 0 |
| `ssd/ssd/engine/step.py` | 3 | 1 |
| `ssd/ssd/engine/verifier.py` | 60 | 24 |
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
| `ssd/tests/test_output_accounting.py` | 88 | 0 |
| `ssd/tests/test_p1_dynamic_tree.py` | 1 | 1 |
| `ssd/tests/test_p2_tree_alloc.py` | 19 | 3 |
| `ssd/tests/test_packed_verify.py` | 90 | 0 |
| `ssd/tests/test_stochastic_verify.py` | 65 | 0 |
| `ssd/tests/test_tree_host_topology.py` | 22 | 0 |
| `ssd/uv.lock` | 26 | 0 |

**부록 C. 주석별 답변 위치**

| 사용자 주석 | 이 문서의 답변 |
|---|---|
| 1 | 1·14·16절: 여섯 수정의 출처/노출 조건, SSD 공통 적용과 공정 비교 |
| 2 | 2절: dtype/wire/workspace |
| 3 | 3절: 과대평가 방향과 재집계 |
| 4 | 4절: 같은 batch의 query 차이와 packed 표현 |
| 5 | 5절: mixed miss, AR fallback, 추가 threshold 여부 |
| 6 | 6절: timing ratio와 phase slack, 기존 profile 재분석 |
| 7 | 7절: soft score와 deterministic proposal 분리 |
| 8 | 8·9절: 이전 tree 연구와 실행기 재사용 계약 |
| 9 | 10절: serving 전체 연결의 의미 |
| 10 | 10절: miss JIT chain과 K2/Kmax |
| 11 | 0·10절: 기존 B1 tree attention/최신 원격 기준 확인 |
| 12 | 11절: 기존 ladder 보존과 새 경로 수정 범위 |
| 13 | 12·15절: 실제 graph 재사용, 계측·최적화 미완료 |
| 14 | 8절: 이전 selector 연구의 online/offline/미실행 구분 |
