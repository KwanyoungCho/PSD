# DUET 최적화 2차 기록 — 논문 tree 범위 정정 포함

기준 branch: `feat/duet-p2tree-g0@a82f7d2`; 작업 branch: `feat/duet-mlsys-coverage`.
날짜: 2026-10-07. 재생성 수치: [NUMBERS.md](NUMBERS.md), 원본 집계: [NUMBERS.json](NUMBERS.json).

**전체 요청 완료 상태는 아니다.** B>1 chain의 추가 최적화와 B1 greedy tree 구현·전체 입력 검증은 수행했다. B>1 dynamic tree의 end-to-end 통합과 throughput 검증은 남아 있다.

## Dynamic tree가 논문의 어떤 부분인가

- 사용자 PDF **4.3절 「다중 검증 결과를 위한 분기 구성」, 식 (4), 표 2의 Tree**다.
- root 후보 `(i,v)` 아래 노드의 점수:

  `s(x_d) = P(i,v) × ∏_{j=1..d} p_D(x_j | (i,v), x_1:j-1)`.

- 매 draft forward에서 모든 root의 frontier를 비교하여 상위 W개 노드만 확장한다. 결과적으로 root마다 깊이와 폭이 달라진다.
- P1/DS에는 proxy가 없으므로 draft의 위치 도달 점수 × root token 확률을 초기 점수로 사용한다.
- root 후보 선정은 cache hit, root 아래 분기 구성은 AL 향상이 목적이다. 여기서 dynamic은 후보별 tree 형태와 노드 배분이 실행 중 달라진다는 뜻이다.
- 논문 표 2: Chain **66.4 TPS / AL 4.33**, Tree **68.1 TPS / AL 4.68**. 논문 자체는 tree를 포함한다.
- **1차 보고서에서 서버 가이드의 chain throughput 설정을 이유로 논문 확장 전체를 chain 검증으로 대표한 것은 범위가 부족했다.** 이 기록에서 정정한다.
- 논문 원실험은 Llama2-70B+TinyLlama1.1B / target TP2+draft / RTX PRO 6000 Blackwell이다. 아래 7B·8B/4090/TP1 수치를 그 표의 재현으로 해석하지 않는다.

## 실제 구현한 변경

| 변경 | 의도 / 동작 | 현재 범위 |
|---|---|---|
| Packed chain verification | `valid_k+1`만 모아 target forward. 정렬 padding은 KV 쓰기가 없는 별도 dummy sequence가 소유 | B>1 chain, SGL attention, opt-in |
| Mixed-miss direct target | 일부 hit가 있는 배치에서 miss 행의 JIT를 기다리지 않고 `valid_k=0`으로 전달 | chain, opt-in; AL과 대기시간의 교환 |
| Greedy tree | draft 자식은 untempered score 상위 C개, target은 argmax와 일치하는 자식 경로만 수락 | B=1 P1/P2 dynamic tree |
| Qwama P1 workspace | 별도 P1 workspace 48→64 MiB; widest P1 FA2 임시 V가 56 MiB를 요구하는 초기화 오류 수정 | 두 phase tree 활성화 가능 |
| Batched tree executor | 요청별 arena·top-W 예산은 유지하고 매 round의 transformer forward를 하나로 묶음 | 독립 primitive; serving 전체 연결 미완료 |
| Host topology 재사용 | tree 응답을 보낼 때 이미 검증한 CPU parent/length를 glue에서도 재사용. 노드별 GPU scalar read와 GPU metadata 조립 제거 | opt-in A/B 검증 |

Packed 경로도 proxy callback, target 전·후반 겹침, 기존 acceptance를 유지한다. packed logits는 기존 dense 논리 row로 다시 매핑하므로 padded row가 proposal로 수락되지 않는다. LM head가 context 수에 따라 3차원 출력을 반환하는 경우도 명시적으로 flatten하여 index를 맞췄다.

Mixed-miss는 같은 proposal을 더 빨리 검증하는 변경이 아니다. miss 행에서 제안을 생략하여 **AL을 낮추고 대기시간을 줄이는 scheduling 정책**이다. miss의 다음 token은 residual이 아닌 해당 prefix의 target P에서 직접 뽑는다. proxy 역시 그 행에는 draft residual을 적용하지 않는다. 기본값을 임의로 모든 모델에 켜지 않았다.

Greedy tree의 untempered draft 확률은 후보 순위용이다. T=0에서 one-hot 분포를 비복원으로 여러 번 뽑는 정의를 사용하지 않으며, target 검증에서 그 순위 점수를 p/q 분모로 사용하지 않는다. T>0 target + T=0 branching draft 조합은 별도 proposal 법칙이 필요하므로 명시적으로 차단한다.

## 전체 입력 B=8 실험

- full dense weights, 양자화 없음. 두 모델 모두 target 1 GPU + draft 1 GPU, RTX 4090.
- repository corpus의 **480개 첫 turn 전부**, input cap512, output cap64, 자연 EOS. 원 데이터셋 전체나 560개 turn 전체는 아니다.
- K1/K2=4/2, exit21, draft/proxy fan-out=2/1, T=0.7.
- 모델별 동일 GPU pair에서 base→packed→mixed를 seed2026/2027/2028마다 실행. target/draft seed 모두 명시, 각 run 새 프로세스.
- foreign GPU PID가 없는 성공 실행만 집계. host 전체 독점은 아니다.
- campaign 기록 총62개 중59개 성공,3개 실패. 실패를 포함한 목록: [RUN_INVENTORY.csv](RUN_INVENTORY.csv). 별도 unit test/HF 진단/microbench는 이62개에 포함하지 않는다.

| 모델 | Base TPS | Packed TPS | Packed+mixed TPS |
|---|---:|---:|---:|
| Llama2-7B + AMD135M | 634.15 ± 2.45 | **651.70 ± 4.91** | 651.04 ± 4.08 |
| Llama3-8B + Qwama0.5B | 646.76 ± 1.17 | 647.58 ± 2.82 | **672.26 ± 4.02** |

- seed별 대응 개선율 평균: Llama2 packed **+2.77%**, Llama3 mixed **+3.94%**.
- Llama3에서 packed만으로는 **+0.13%**여서 실질 개선의 근거가 부족하다.
- Llama2는 mixed가 packed보다 더 좋다는 근거가 없다. AL도 함께 확인해야 한다.
- 이전 보고서의 617/630 TPS와 직접 비율을 계산하지 않는다. 위 base는 이번 시점에 다시 측정한 이미 1차 최적화가 적용된 경로다.
- 추가 최종 비교는 Llama2 packed+P1 fan-out4, Llama3 packed+mixed, SSD K3/fan-out2를 같은 GPU pair에서 번갈아 실행한다. 결과는 `llama*_final/`과 [NUMBERS.md](NUMBERS.md)에 기록한다.

최종 후보와 SSD를 교대로 실행한 **480문항 × 3회 비교도 완료**했다.

| 모델 | 최종 DUET 후보 TPS | 튜닝한 SSD TPS | seed별 대응 개선율 평균 |
|---|---:|---:|---:|
| Llama2 | 652.45 ± 3.79 | 649.08 ± 1.70 | +0.52% |
| Llama3 | 673.06 ± 5.54 | 652.33 ± 4.16 | +3.18% |

- Llama2의 차이는 작다. 3개 seed의 대응 개선율로 계산한 t 구간도 0을 포함하므로 확실한 우위라고 주장하지 않는다. packed만으로 얻는 이득이 주효하며 fan-out4의 추가 이득은 미미하다.
- Llama3는 이 조건의 세 대응 실행 모두 SSD보다 빨랐다. 다만 표본 반복은3회이고, AL은 base 약2.373→최종 약2.119로 감소한다. AL 개선 기법으로 설명하지 않는다.
- 따라서 기본 파라미터를 유지하려면 Llama2는 packed만 사용해도 된다. Llama3의 mixed-miss는 throughput 우선일 때 선택하는 정책이다.
- 그래프: [TPS와 AL PNG](figs/batch8_tps_al.png), [PDF](figs/batch8_tps_al.pdf). 오차막대는 표본 표준편차이며, prototype microbench 수치는 포함하지 않았다.

## Tree와 greedy 검증

- 최초 16문항/32-token 실행에서 Llama2의 P2-only tree와 P1/P2 tree 모두 AR 출력과 **16/16 동일**했다.
- Llama3 P2-only tree는 **14/16 동일**. 다른 두 first-divergence를 독립 HF로 확인한 결과 하나는 동점, 다른 하나는 DUET token이 HF argmax였다. 이를 모든 입력의 bitwise 일치 보장으로 확대하지 않는다.
- Qwama의 P1/P2 활성화 실패 로그와 수정 후 재실행 모두 보존했다. 실패한 실행을 성공/성능 통계에 넣지 않는다.
- `llama*_tree_full/`: B=1, T=0, 480문항, P1/P2 tree와 chain. `tree_verify_events`가 실제로 발생했는지 확인하며 이름만 tree인 fallback을 성공으로 세지 않는다.
- `llama*_tree_host/`: 같은 tree 설정에서 CPU topology 재사용만 켠 대조 실행. 출력 일치와 TPS를 함께 비교한다.
- Tree가 AL을 올려도 query 수·draft 작업·metadata 처리 비용 때문에 TPS는 낮아질 수 있다. 작은 모델에서의 tree/chain 결과를 70B 원논문 설정으로 일반화하지 않는다.

전체 입력의 실제 결과(한 번씩 측정):

| 모델 | Chain TPS / AL | P1+P2 tree TPS / AL | CPU topology 재사용 tree TPS |
|---|---:|---:|---:|
| Llama2 | 123.57 / 2.2413 | 108.22 / 2.3772 | 106.94 |
| Llama3 | 128.96 / 2.7589 | 104.18 / 2.8570 | 101.85 |

- 실제 tree verification event: Llama2 **11,083**, Llama3 **6,319**. 두 phase가 tree인 결과다.
- Host topology 변경은 두 모델 합계 **960/960 출력 및 모든 phase event가 동일**했다. 그러나 TPS 개선은 재현되지 않았고 측정값은 각각 -1.18%/-2.23%였다. 따라서 **기본 OFF를 유지**한다. 구현상 동기화 감소만으로 성능 개선을 주장하지 않는다.
- Tree와 chain의 token 완전 일치: Llama2 **458/480**, Llama3 **382/480**. 서로 다른 attention/query shape를 사용한다.
- 독립 HF first-divergence 진단: Llama2 22개 모두 두 token이 HF top2, logit 차이 최대0.015625. Llama3 98개 모두 HF top5(85개는 둘 다 top2), 차이 최대0.25. 근접 후보의 수치 차이를 시사하지만 bitwise 동등성을 인증하지 않는다.
- 원본: `llama*_tree_full/chain_tree_audit.json`, 요약: [greedy_comparison.json](validation/greedy_comparison.json).

## 정확성 검사와 한계

- Packed attention: 서로 다른 길이와 page 배치를 captured replay로 바꾸면서 각 요청의 독립 causal attention과 비교.
- `valid_k=0`: 20,000개 stochastic 샘플에서 recovery가 target P를 따르는지 확인. greedy의 zero-proposal row도 검사.
- Greedy tree: 두 번째 sibling과 grandchild를 따라가는 예시 및 없는 자식에서 정확한 target recovery를 반환하는 예시.
- Executor: 실제 paged attention을 사용하는 미니 모델에서 개별 실행과 batched eager/captured replay를 비교; prefix 변경 반영도 확인.
- Full draft microbenchmark는 합성 prefix와 전체 AMD/Qwama 가중치를 사용한다. **그 속도는 DUET end-to-end TPS가 아니다.** 큰 배치의 후보 차이는 별도 진단 결과를 확인해야 한다.
- AMD의 같은 text prefix를 prefill한 B8 primitive에서는 topology/token이 일치했고 최대 logit 차이는0.03125였다. Qwama의 큰 logit 차이에 대해서는 실제 prefix, strict matmul, fp16, GEMM row-padding, layer별 중간값 추적과 독립 HF float32 검사를 추가했다. 아래 표는 서로 다른8개128-token 문맥, 문맥당 상위5개 root(총40개)의 별도 소규모 진단이다.
- 통합 **263개 회귀 검사 통과**. B8/GQA·host topology 검사 포함, 최종 로그: [regressions.log](validation/regressions.log).
- 부동소수점 batch shape 변경으로 near-tie argmax가 달라질 수 있다. greedy algorithm의 정확성과 특정 HF/batch 실행과의 token 완전 일치를 구분한다.
- Packed 경로는 현재 SGL 전용이다. Blackwell 자동 FlashInfer 경로의 packed 지원과 실제 하드웨어 성능은 검증하지 않았다.

추가 draft 실행기 진단:

| Draft / B8 | 개별 대 배치 root 분포 TV 평균 / 최대 | HF float32와 TV 평균: 개별 / 배치 | HF argmax 일치: 개별 / 배치 |
|---|---:|---:|---:|
| AMD fp16 | 0.00095 / 0.00330 | 0.03906 / 0.03907 | 39/40 / 39/40 |
| Qwama bf16 | 0.01074 / 0.03103 | 0.02239 / 0.02180 | 39/40 / 40/40 |

- TV는 untempered softmax(T=1)의 비교다. 실제 greedy verifier의 확률 오차라는 뜻이 아니다.
- Qwama의 최대 logit 차이8은 token128008의 -524 대 -516에서 발생했고 두 값의 softmax 확률은 사실상0이다. 같은 입력에서 첫 layer norm 출력은 정확히 같았으며 첫 QKV projection부터 bf16 차이가 있었다. 최대 절댓값 하나만으로 attention mask/요청 혼합 오류라고 판단할 근거는 부족하다.
- 다양한 prefix의 Qwama에서는 tree 토큰/구조가 일부 달라졌다. 그럼에도 위 독립 reference 대비 root 분포 오차가 커지지는 않았다. 이는 수치 민감성을 시사하는 제한된 진단이며 모든 tree 컨텍스트의 정확성 증명은 아니다.
- 동일 문맥만 반복했을 때 Qwama B4/B8 tree 토큰/구조는 모두 동일했다. 다양한 입력의 검사와 구분한다.
- 위 B8 실행기 replay만의 latency: AMD 개별13.59ms→배치6.14ms, Qwama37.00ms→10.38ms. **cache/target/통신 없는 primitive 측정이므로 DUET TPS 개선으로 사용하지 않는다.**
- 원본: [AMD 독립 reference](batch_tree_amd_diverse_hf.json), [Qwama 독립 reference](batch_tree_qwama_diverse_hf.json), [Qwama 같은 문맥 검사](batch_tree_qwama_hf.json), [layer 추적](batch_tree_qwama_layers.json).

## 완료로 볼 수 없는 부분

**B>1 dynamic-tree의 cache→통신→target 검증→KV commit 전체 경로는 아직 완성되지 않았다.** Batched executor primitive를 추가했다고 이 기능이 지원된다고 표현하면 안 된다. 현재 harness는 B>1 `duet-tree`를 거부한다.

남은 핵심은 요청별 cache view/parent-q/terminal-node namespace, mixed chain/tree response wire, batched target ancestor mask와 검증, 종료·refill·preemption 시 accepted-path KV 복원이다. 기존 코드의 B=1 guard를 제거하는 것으로 해결되지 않는다. 구체적인 연결 지점과 재검증 순서는 [HANDOVER.md](HANDOVER.md)에 남긴다.
