# MLSys ver1.6 적용 안내

MLSys ver1.6은 **ver1.4의 DUET 생성 및 검증 알고리즘을 유지하면서 target TP 검증 실행 경로를 최적화한 버전**이다. Chain과 tree 모두 적용된다. 기존 `MLSys-ver1.5`의 다른 트리 알고리즘 변경은 포함하지 않는다. 기준 엔진은 `MLSys-ver1.4@dedc85ccb299df2d76fcf34d50f05f2d8747a32a`이며 runtime 변경은 `bec21211a`와 `42f616fd5`에 있다. 연구 버전은 Git branch와 commit으로 구분하며 `ssd/pyproject.toml`의 upstream 패키지 버전과는 별개다.

## 변경 내용

2026-10-23부터 P1 draft-source tree의 API 및 benchmark CLI 기본값을 `off`에서 `on`으로 바꿨다. P2는 기존처럼 `on`이므로 DUET의 두 phase가 기본적으로 동적 tree를 사용한다. Chain 실험은 `duet_p1_tree_policy='off', duet_p2_tree_policy='off'` 또는 CLI의 `--duet_p1_tree_policy off --duet_p2_tree_policy off`를 모두 명시한다. 기본값에 의존한 과거 P2-only 실행을 재현하려면 P1을 명시적으로 `off`로 둔다. 기존 주 실험은 두 값을 이미 `on`으로 지정했으므로 이 기본값 변경으로 설정이 바뀌지 않는다.

초기 배포 `bcb4a1e443`에는 `enforce_eager=True`의 dense attention planner 회귀가 있었다. 실행 모드를 prefill 여부와 혼동하여 decode/verify의 `cu_seqlens_k=None`을 읽는 오류다. 후속 수정은 planner의 layout 분류를 ver1.4와 동일하게 복구하고, non-mixing eager 안전 동기화는 `run_model` 진입부에 둔다. 이 회귀는 최적화 flag를 꺼도 eager 실행에서 발생했으므로 초기 commit 대신 수정된 최신 ver1.6을 받는다. 일반 CUDA graph 검증의 분기와 수락 수식은 이 수정으로 바뀌지 않는다.

기존 TP 검증은 captured transformer graph 사이에 eager 중간층 LM head와 TP gather를 실행했다. 이 서버에서는 NCCL의 captured/eager collective 혼용 경로에서 `cudaGraphLaunch`의 CPU 제출 시간이 수십 ms까지 늘어났다. 고정 graph에서는 특정 폭의 지연이 반복됐지만, graph를 다시 구성하면 느린 폭도 이동했다. 따라서 특정 G 자체가 항상 느리다는 의미는 아니다.

ver1.6은 중간층 head/gather를 pre graph에, 최종 head/gather를 post graph에 포함한다. 이 경로에서는 NCCL graph mixing을 끌 수 있다. 실제 입력이 tree bucket보다 짧으면 실제 행 수에 맞는 보조 head graph를 사용한다. Eager prefill 및 진단 fallback에는 안전한 경계를 둔다. **NCCL 설정만 기존 엔진에 적용하면 안 된다.** Captured collective가 실행 중일 때 같은 communicator의 eager collective를 시작하는 것은 안전하지 않다. [NCCL CUDA graph 안내](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)와 [graph mixing 제약](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support)을 참고한다.

| 파일 | 변경 |
|---|---|
| `ssd/ssd/engine/helpers/verify_logit_graphs.py` | Opt-in 조건 검사, 원래 norm/head/TP gather capture, 안전한 eager 경계 |
| `ssd/ssd/engine/helpers/cudagraph_helpers.py` | Chain 및 tree의 pre/post head capture와 실제 행 수 보조 graph |
| `ssd/ssd/engine/model_runner.py` | 해당 graph dispatch, 짧은 tree의 최종 head 입력 trimming, startup bucket 선택 |
| `ssd/ssd/engine/helpers/target_verify_buckets.py` | 선택적 짧은 query bucket 추가와 가장 작은 수용 bucket 선택 |

Verifier, draft 생성기, P1/P2 후보 점수, q-path 경쟁, ordered WOR, 수락/보정 수식, sampler와 모델 가중치는 ver1.4 그대로다. Config 기본값 중 P1 tree 스위치만 위와 같이 `on`으로 바뀌었다. G는 여전히 생성되는 **root 제외 노드 상한**이다. Tree를 생성한 뒤 점수순 top-M으로 자르는 알고리즘은 도입하지 않았다.

## 다른 서버에서 받기

기존 dirty checkout을 바꾸지 말고 새 clean worktree를 만든다. 경로가 이미 있으면 새 이름을 쓴다.

```bash
git fetch origin MLSys-ver1.6
git worktree add --detach ../PSD-MLSys-ver1.6 origin/MLSys-ver1.6
cd ../PSD-MLSys-ver1.6
git log -1 --format='%H %s'
git status --short
python3 -B ssd/tools/verify_dispatch/check_release.py
```

기존 SSD Python/CUDA 환경과 같은 모델 checkpoint를 사용한다. 모델을 변환하거나 70B 가중치를 복제할 필요는 없다. 모델·데이터·결과는 이 release에 포함하지 않았다. `SSD_HF_CACHE`, `SSD_DATASET_DIR`, `SSD_CUDA_ARCH`, `TORCH_CUDA_ARCH_LIST`, GPU·포트·SHM 이름은 해당 서버에 맞춘다. 사용 중인 GPU를 중단하거나 공용 환경에 패키지를 덮어 설치하지 않는다.

## 최적화 활성화

기본 실행은 기존 경로다. **다음 두 설정을 함께 적용**하고, 지원 조건을 지킨다.

```bash
export SSD_VERIFY_GRAPH_LOGITS=1
export NCCL_GRAPH_MIXING_SUPPORT=0
export SSD_DUET_EXIT_REPLICA=0
export SSD_DUET_EXIT_TOPM_GATHER=0
export SSD_PROXY_STREAM=0
export SSD_BATCH_TREE_PROXY_STREAM=0
export SSD_ASYNC_PROXY_SEND=1
```

검증한 구성은 B=1, BF16, target TP2, full-vocabulary TP head, 기본 proxy stream이다. B>1, exit replica, top-M gather, 별도 proxy stream을 이 non-mixing 구성에 섞지 않는다. Runtime은 지원하지 않는 head 구성을 가능한 한 일찍 거절한다. 다른 TP 수·GPU·라이브러리 버전에서의 성능은 해당 서버 smoke와 동일 입력 검증 후 판단한다. 이 release가 검증하지 않은 조합을 일반적으로 지원한다고 간주하지 않는다.

`SSD_TREE_VERIFY_BUCKETS`는 별도 선택 사항이며 위 최적화의 필수 설정이 아니다.

```bash
# 기본값: phase별 G cap bucket 유지
export SSD_TREE_VERIFY_BUCKETS=''
# 선택 예시: cap12에 2/4노드 bucket을 추가
# export SSD_TREE_VERIFY_BUCKETS='2,4'
```

추가 bucket은 dummy transformer 행을 줄일 수 있지만 graph와 attention workspace의 GPU 메모리를 늘린다. 이 서버에서는 2/4 추가 시 GPU당 약4GiB가 늘었다. `exact`로 전체 폭을 무조건 capture하지 않는다. **이번 full 성능 재측정은 추가 bucket 없이 기본 phase cap을 사용한다.** 실제 tree가 G보다 짧을 때 transformer는 여전히 수용 bucket 폭으로 실행할 수 있고, head는 실제 행 수만 처리한다. Padding이 전부 제거됐다고 표현하지 않는다.

## 검증과 측정 범위

아래는 release 이전에 이 서버에서 완료한 검사다. [검증 요약](results/mlsys_ver16/RELEASE_AUDIT.json)에 범위와 분모를 보존했다.

- CPU 회귀15개 통과: head graph8개, query bucket7개.
- 동일 고정 입력의 기존/new chain·tree logits8쌍에서 최대 오차0.
- Captured head의 mixing on/off causal chain/tree control72개에서 최대 오차0.
- Physical13행에 실제2·3·5·7·9·11·13행을 넣은 짧은 tree7조건에서 proxy 및 최종 logits 최대 오차0.
- 실제 70B DUET tree·chain 각각 별도 cold-start smoke 통과. 각각 별도 warmup2턴과 측정6턴이며 원래 출력 토큰 합계·B1 event·문맥 guard를 검사했다.

검증 환경은 RTX PRO 6000 Blackwell SM120, target GPU2장과 draft GPU1장, driver580.178.04, Python3.11.15, torch2.8.0+cu128, NCCL2.27.3, FlashInfer0.5.2, Transformers4.57.1이다. Target은 LayerSkip-Llama2-70B revision `0a1815fcbd11543b0de227aa8dcbced952149411`, draft는 TinyLlama-1.1B-Chat revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`이다.

선택 폭 진단의 target 검증구간 평균은 N13에서60.05→57.47ms, N14에서60.40→57.49ms, N16에서59.84→58.59ms, N17에서63.04→60.17ms였다. N은 root를 포함한 실제 query 행 수이고 G=N−1이다. Prefix512, seed42·123·2026에서 보존한 실제 상태3개씩, 상태당15회 중앙값을 상태→seed 순서로 평균했다. **전체 decode step이나 TPS가 아니라 target component**이며 이후 전체 폭/성능 재측정과 구분한다.

폭이나 graph lifecycle을 바꾸면 BF16 kernel 선택으로 logits가 미세하게 달라질 수 있다. 동일 physical width의 head capture 변경과 width 자체 변경을 구분해 검사한다. 비동기 online 실행은 시간 변화로 cache hit와 RNG 진행이 달라질 수 있으므로 문장 전체의 bitwise 동일성을 보장하지 않는다. Graph span의 pre/post에는 이제 head/gather가 포함되므로 예전 timeline label과 그대로 비교하지 않는다.

## 재검증 순서

먼저 위 manifest 검사와 CPU 회귀를 실행한다. 이 명령은 모델 forward를 실행하지 않는다.

```bash
cd ssd/tests
python -B -m unittest -v test_verify_logit_graphs test_target_verify_buckets
```

Eager 회귀도 함께 검사한다.

```bash
python -B -m unittest -v test_flashinfer_eager_dispatch
```

그다음 확보한 GPU에서 기존 서버의 **동일 모델·입력·seed·K1/K2·G·온도·검증기**를 유지해 짧은 chain과 tree smoke를 각각 실행한다. 해당 서버의 원래 실행 명령에 활성화 환경변수만 더하고, 두 경로의 정상 종료·토큰 accounting·실제 행 수·source/AL event·source hash를 확인한 뒤 full 실행한다. 상세 계측은 성능 측정과 분리한다. 실행 중 소스나 환경을 바꾸지 않고 실패 attempt도 보존한다.

원복하려면 새 process에서 `SSD_VERIFY_GRAPH_LOGITS=0`으로 바꾸고 **`NCCL_GRAPH_MIXING_SUPPORT`도 unset하거나1로 돌린다.** 설정0으로 graph mixing만 끈 채 기존 경로로 돌아가지 않는다. 모델·입력·기존 ver1.4 checkout을 수정할 필요는 없다.
