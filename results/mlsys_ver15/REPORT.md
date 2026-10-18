# MLSys-ver1.5: 경로 점수 top-M tree 선택

2026-10-19 KST. **구현과 최종 기능 검증을 완료했다.** 기준 v1.4는 `dedc85ccb299df2d76fcf34d50f05f2d8747a32a`. v1.4 worktree는 변경하지 않았다. Runtime 커밋은 `d9029f27bb14bdc3876a76d606a0a51f029ffb03`다. 새 전체 성능 측정은 이 버전의 깨끗한 커밋에서 시작하며, 완료한 seed만 [live 보고서](performance/REPORT.md)에 누적한다. 전체 AL/TPS의 개선을 이미 입증했다는 뜻은 아니다.

## 무엇을 바꿨는가

| 구간 | v1.4 | v1.5 |
|---|---|---|
| Forward 확장 점수 | root 점수 × 원래 draft q-path | 유지, 마지막 깊이 경쟁 |
| Root별 생성 제한 | G=M | K·실제 입력 폭·C에 따른 자연 상한. 별도 G 튜닝 없음 |
| 검증으로 남길 tree | 생성 순서 앞 M | 경로 점수 상위 M, 부모 보존 및 index 재작성 |
| 최종 선택 구현 | prefix 포장 | GPU CUDA graph에서 선택·부모/형제 remap, 선택된 M개만 bulk CPU readback |
| Tree-enabled run의 verifier | 원래 q 기반 p/q | target에서 직접 token sampling 후 tree의 자식과 일치 여부 확인 |
| B=1 실행 | 전용 경로 | B>1과 같은 공용 tree 경로. 레거시 `SSD_BATCHED_TREE=0`으로 옛 verifier에 우회하지 않음 |
| Miss chain | JIT chain | 생성 유지, 검증은 해당 run과 같은 새 target sampler |
| Pure chain mode | 기존 p/q | 그대로 유지 |

원래 q에서 뽑은 토큰을 점수로 사후 pruning하면 선택된 토큰의 법칙이 q에서 달라질 수 있다. 원래 q를 p/q에 그대로 사용하는 구현은 채택하지 않았다. 같은 p=q에서도 TV=0.1273의 편향이 생기는 유한 열거 반례를 검사했다. 새 방식은 어떤 tree에서도 현재 문맥의 target draw를 그대로 출력하므로 target 분포를 보존한다. [증명과 선행연구](THEORY.md).

**중요한 해석 변화:** 기존 `e(1-q)` 및 위치식은 cache root 우선순위 휴리스틱으로 유지한다. 새 verifier에는 rejection residual이 없으므로 기존 residual/종료 분포의 해석을 그대로 적용하지 않는다. AL/hit/TPS 변화는 선택법과 verifier 변경을 함께 포함한다. 점수 선택만의 효과로 주장하지 않는다.

## Latency 확인 및 반영

CPU에서 생성 노드를 모두 포장하고 점수 정렬하는 최초 구현은 추가 비용이 있어 제외했다. 최종 선택·index remap을 GPU graph로 옮겼다. Target 수락과 경로 선택도 GPU graph 한 번으로 수행하며 depth별 CPU 왕복은 없다. 새 모델 forward나 token 재샘플링을 추가하지 않는다.

RTX4090, PyTorch2.8.0+cu128, BF16, 고정 큰 arena 입력을 모사한 항목별 검사다. 각 arm 10회 준비 뒤 60회 측정, 순서를 번갈아 실행했다. 두 phase의 전체 arena 2,340행, M1/M2=23/15. GPU 선택과 독립 CPU 참조의 토큰·부모·형제·parent-cell·생성 수가 일치했다.

| 선택·전송 준비 처리 | 중앙값 (ms) | p95 (ms) | 판정 |
|---|---:|---:|---|
| 같은 큰 arena를 CPU에 읽은 뒤 prefix 포장하는 참조 | 2.310 | 2.400 | 비교용, 과거 v1.4 전체 runtime 측정은 아님 |
| 전체 arena readback + CPU score top-M | 3.547 | 3.623 | 미채택 |
| **GPU score top-M + 선택된 M metadata bulk readback** | **0.764** | **0.778** | 채택 |

위 시간에는 선택 graph 실행과 최종 metadata readback·CPU dictionary 포장이 포함된다. 대기 중인 transformer kernel을 먼저 비운 조건이며, 실제 inference의 대기시간을 모두 분해한 결과는 아니다. 0.764ms를 단순히 전체 step의 절감량으로 환산하지 않는다.

같은 B1/V32000/logits shape에서 수락 graph만 비교한 결과:

| 노드 수 | 기존 p/q GPU 참조 (ms) | 새 target sampler (ms) |
|---|---:|---:|
| 15 | 0.364 | 0.183 |
| 23 | 0.440 | 0.239 |

서로 다른 검증 알고리즘이며 AL coupling이 다르다. 이 표는 계산 비용 검사이지 AL 동등성이나 end-to-end 개선 증거가 아니다. [원시 수치](checks/COMPONENT_GPU.json), [검사 도구](tools/component.py).

추가로 staging-only executor에서 쓰지 않는 거대한 root별 backbone/parent finalize를 생략했고, M1=23/M2=15일 때 PS target/glue가 23노드까지 padding되지 않도록 phase별 shape를 준비했다. Graph 준비는 decode 집계 밖에서 끝낸다.

## 최종 검증

| 범위 | 결과 |
|---|---|
| 최종 runtime GPU 회귀 16개 suite | **149개 통과, skip0** |
| 기존 q verifier + 사후 선택의 편향 | 유한 열거로 실패 반례 재현 |
| 새 target verifier | arbitrary pruned tree의 두 토큰 결합분포를 exact 열거, empirical sampling 검사 |
| Greedy | target argmax, 난수 상태 불변 |
| CUDA graph | B1/B4, dense/flat, T=0/T>0 경로 통과 |
| Score 선택 | 부모 보존·동점·q=0·index remap·자연 생성 bound 검사 |
| GPU exporter | 독립 CPU 참조와 일치. 입력 q 변경 후 replay, 외부 constant 수명·allocator 재사용 검사 |
| 실제 LayerSkip7B + llama68m, T=.7 | 9 raw 턴, 첫3 warmup, 출력 상한96, 측정219 steps, P1/P2/miss 모두 실행 |
| 같은 실제 모델, T=0 | 같은9턴, 측정140 steps, P1/P2/miss 모두 실행 |
| CPU-score 구현 대비 최종 GPU-score 구현 | 각 온도 **9/9 출력 동일**. v1.4나 AR와의 출력 parity를 의미하지 않음 |
| 입력·집계 | 입력 ID 보존, 실제 반환 token 수 일치, 명시적 무디코딩 분류, preemption 없음 |
| Source | 두 최종 모델 실행 모두 runtime hash 변화 없음. 개발 worktree 실행임을 기록 |
| 연속 seed 도구 | 컴파일, 우선 seed 순서, 중복 방지, 비어 있는 GPU 쌍만 사용 확인 |

최종 증거는 [CHECKS.json](checks/CHECKS.json), [회귀 로그](checks/final_regression.log), [실제 모델 검사](checks/MODEL_SMOKE.json), [smoke 원시 자료](checks/MODEL_SMOKE_RAW.tar.gz)다. 처음 실패했던 import/glue shape 문제와 GPU constant 수명 문제를 수정한 **최종 소스**의 증거만 release 표에 사용했다. 중간 개발 로그는 raw의 `development_checks/`에 보존했다.

모델 smoke의 상한96은 기능 확인용이다. 논문 조건의 전체 성능 결과로 제시하지 않는다. BF16 수치 차이와 병렬 난수 소비 때문에 동일 seed의 AR와 bitwise 같다는 보장도 하지 않는다.

## 시작할 전체 성능 측정

- 모델: LayerSkip-Llama2-7B + JackFram/llama-68m, BF16, B=1, target/draft T=0.7.
- Exit21. P1 K=10, 위치당 root4, continuation W cap56, M=23. P2 K=5, root24, W=28, M=15. C=3, last.
- 비동기 proxy 전송만 ON. Exit replica·별도 proxy stream·profiler OFF.
- 매 seed 전체 Spec-Bench480문항/560 raw 턴. 문맥2048, 출력 min(1024,2048−입력), 자연 EOS·기존 문맥 보호. 추가 template/truncation 없음.
- 첫3 raw 턴은 중복 없이 처리하고 성능 제외. 완료 결과는 턴→문항→seed 평균. 기존 cap/clip·부분 batch·같은 요청/batch/source 구성 내 2×중앙값 초과 step 제외를 유지하고 제외 수를 공개한다. 원시 결과도 보존한다.
- Seed42→123→2026→0→1→2→… 순서로 배정한다. 병렬 실행의 완료 순서는 다를 수 있다. v1.4 seed 재사용 없음.
- 로컬 (4,5),(6,7),(1,2),(0,3) 중 비어 있는 허가된 쌍에 배정한다. 외부 GPU 작업은 건드리지 않는다.
- 사용자가 중단할 때까지 자동 누적하며, 오류나 검증 실패면 신규 seed를 중단한다. 디스크35GiB 미만이면 신규 seed 대기.

Live: [performance/REPORT.md](performance/REPORT.md). Raw: `/data/chokwans99/duet-ver15-score-seeds-20261019`. 실행 당시 commit·모델 경로·GPU·telemetry·입력/소스 해시를 seed별로 저장한다. 완료 seed의 전체 TPS/AL/hit/step latency, P1/P2/Miss별 rate/AL/step latency와 제외 수를 표로 공개한다.

최초 campaign 시작 시 benchmark adapter의 상대 경로에 고정 집계기 `performance.py`가 없어 모델 로딩 전에 종료했다. Runtime 오류나 완료 seed가 아니다. 서버 절대 경로 symlink 대신 같은 집계 파일 한 개를 명시적으로 포함하고 prepare 단계에서 runtime 집계기와 byte 일치를 확인하도록 수정했다. 실패한 시작 기록은 development_checks에 보존하고 같은 seed를 다시 시작한다.

## 검토·재현

[인수인계](../../MLSYS_VER15_HANDOVER.md)의 파일별 순서로 검토한다. `checks/CHECKS.json`은 최종 runtime hash를 담고, full run validator는 공개된 clean commit과 현재 파일 해시를 함께 확인한다. 모델·GPU layout을 검토하지 않고 다른 서버의 launcher를 그대로 덮어쓰지 않는다.

재현 예시(경로는 환경에 맞게 확인):

```bash
export PYTHONPATH="$PWD/ssd:$PWD/ssd/tests"
export SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp
CUDA_VISIBLE_DEVICES=4 /path/to/python -B -m unittest -v test_score_tree_v15
CUDA_VISIBLE_DEVICES=6 /path/to/python -B results/mlsys_ver15/tools/component.py --output /tmp/ver15-component.json
```

연속 seed는 환경 변수 `VER15_CHECKOUT`, `VER15_RAW`, `VER15_REPORT`를 지정하고, clean commit을 push한 다음 `tools/prepare.py`로 계획·release gate를 만든 뒤 `tools/controller.py`를 실행한다. `VER15_TARGET`, `VER15_DRAFT`, `VER15_CACHE_ROOT`, `SSD_CUDA_ARCH`로 서버의 자원 경로를 명시한다. 이 연속 controller는 로컬 orion 전용이다.
