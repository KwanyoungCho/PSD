# MLSys ver1.0 다른 서버 작업자 인수인계

> **이 브랜치는 `MLSys-ver1.5`다.** [ver1.5 인수인계](MLSYS_VER15_HANDOVER.md)를 먼저 확인한다. 점수순 subtree 선택과 target-sampling 검증을 함께 적용했으므로, 아래 ver1.0의 생성 순서 prefix·p/q 설명은 현재 tree 경로에 적용하지 않는다. 공개 파라미터 이름은 v1.1 계약을 유지하되 별도 생성 G는 자연 상한으로 대체했다. v1.4는 이전 결과 재현용으로 그대로 보존한다.

2026년 10월 15일 기준. 다른 서버에서는 **`MLSys-ver1.0`을 별도 worktree로 받아 아래 70B preset부터 검증**한다. 이 브랜치는 최신 ver0.2 위에 현재 서버의 확정된 후보 점수 설정과 calibration 없는 tree 확장을 통합한 버전이다. 기존 연구 브랜치의 파일이나 launcher를 통째로 덮어쓰지 않는다.

## 브랜치와 전달 상태

| 항목 | 기준 |
|---|---|
| 저장소 | `https://github.com/KwanyoungCho/PSD.git` |
| 통합 브랜치 | `MLSys-ver1.0` |
| 구현 기준 커밋 | `b8eac4cbeba2899409753a319f6e8556dffc72a5` |
| 부모 버전 | `MLSys-ver0.2@1b93104b` |
| 현재 서버 worktree | `/home/eslab/chokwans99/PSD-MLSys-ver1.0` |
| 원격 전달 | **문서 작성 시점에는 미 push**. 원격 브랜치 게시 후 아래 fetch 가능 |

이 문서는 구현 기준 커밋 이후의 문서 변경이다. 발신자가 최종 전달 커밋을 함께 알려주고, 수신자는 실제 checkout의 커밋을 기록한다. `ssd/pyproject.toml`의 패키지 버전 `0.2.0`과 연구 브랜치 이름 `MLSys-ver1.0`은 별개다. 버전 식별에는 Git 커밋을 사용한다.

원격 게시 후, 기존 PSD 저장소에서 실행한다. 같은 이름의 worktree나 브랜치가 이미 있으면 새 이름을 사용하며 기존 것을 삭제하지 않는다.

```bash
git fetch origin MLSys-ver1.0
git worktree add -b review/mlsys-ver1.0 ../PSD-MLSys-ver1.0-review origin/MLSys-ver1.0
cd ../PSD-MLSys-ver1.0-review
git log -1 --format='%H %s'
git status --short
git merge-base --is-ancestor b8eac4cbeba2899409753a319f6e8556dffc72a5 HEAD
```

## 반드시 유지할 알고리즘 설정

| 항목 | 기본 동작 | 변경 시 주의 |
|---|---|---|
| DS 후보 | 위치별 균등 선정 | 위치별 후보 수와 tree forward 폭은 다른 값 |
| DS 후보 점수 | 해당 위치까지의 draft 경로 확률 × 후보 토큰의 draft 확률 | 후보 토큰 확률만 사용하는 ablation과 혼동하지 않음 |
| PS 후보 | DS 중복을 제외하고 위치 점수 × 토큰 점수로 적응 배분 | DS와 PS는 별도 phase 안에서 tree 확장 경쟁 |
| 위치 점수 | `0.75 h_hat + 0.25 h_bar` | `duet_root_overlap_mix=0.25`는 **h_bar의 비율**. alpha 자체를 섞는 것이 아니라 종료 위치 질량을 혼합 |
| 토큰 점수 | correction 후보는 `e(1-q)`, bonus 후보는 `e` | 실제 target의 correction sampling을 이 식으로 바꾸지 않음 |
| 정규화 | **위치별 top-M 합으로 정규화** | `duet_root_normalization='topm'`. 전체 어휘 정규화는 명시적으로 `'full'` 지정 |
| 후보 온도 | e는 target 온도, q는 실제 draft 온도 사용 | 기본은 둘 다0.7. 후보 점수만 T=1로 고정하지 않음 |
| Tree 점수 | 후보 점수 × 원래 draft 경로 확률 | calibration 표 사용 안 함. 형제 제거 후 재정규화된 q로 경로 점수를 대체하지 않음 |
| 확장 범위 | **`last` 기본**, `frontier` 비교 옵션 | Last도 같은 phase의 여러 후보 사이에서 경쟁. Frontier는 이전 깊이의 미확장 노드도 경쟁 |
| 자식 생성 | 부모당 최대 C=3 | 남은 생성 공간에 따라 더 적을 수 있음. 주경로를 강제 보호하지 않음 |
| 검증 subtree | hit된 후보의 **생성 순서 prefix** | 점수 rerank 없음. 부모·형제의 검증 순서 보존 |

top-M의 M은 `duet_proxy_top_k`가 결정한다. Config가 전송 용량 조건 때문에 값을 올리면 실제 적용값을 기록한다. 이는 **후보 점수 정규화**이며 target의 실제 sampling 분포를 top-M으로 자르는 설정이 아니다. 아래 표의 tree 검증 노드 상한 M과도 서로 다른 파라미터다.

T=0은 별도 greedy 조건이다. 현재 구현은 후보 순위에는 soft 분포를 사용하고 실제 검증은 greedy로 수행한다. T>0의 residual sampling 실험과 같은 의미로 합치지 않는다.

## 70B 논문용 시작 설정

설정의 원본은 [ssd.presets.duet_70b_paper_kwargs](ssd/ssd/presets.py)다. **일반 `Config`·`LLM`·CLI 기본값 전체가 논문용 preset과 같지는 않다.** 예를 들어 DS tree는 preset에서 명시적으로 켠다. 새 실행기는 preset을 가져오고 필요한 실험 차이만 명시적으로 덮어쓴다.

| 분류 | 값 | API 설정 |
|---|---|---|
| Target | LayerSkip Llama2 70B | `facebook/layerskip-llama2-70B`의 로컬 snapshot 경로 |
| Draft | TinyLlama 1.1B Chat | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`의 로컬 snapshot 경로 |
| 정밀도 | target·draft 모두 BF16 | `dtype='bfloat16'` |
| 배치 | **B=1** | `max_num_seqs=1` |
| GPU 배치 | target TP2 + 독립 draft GPU1개 | `num_gpus=3`, 할당받은 장치3개만 노출 |
| 문맥 상한 | 2048 | `max_model_len=2048` |
| 출력 상한 | **1024** | `SamplingParams(max_new_tokens=1024)` |
| 온도 | target0.7, draft도0.7 | `temperature=0.7`, `draft_temperature=None`이면 target 값 사용 |
| Seed | 우선42 | launcher에서 Python·Torch seed 설정. preset에 포함되지 않음 |
| 비동기 경로 | SD·async·JIT·DUET 켬 | `speculate`, `draft_async`, `jit_speculate`, `duet_enabled` 모두 True |
| Exit | 56 | `duet_exit_layer=56` |
| Forward 깊이 | K1/K2=8/4, 합12 | `duet_phase1_k=8`, `duet_phase2_k=4`, `speculate_k=12` |
| 기본 fan-out | 3, DS도3 | `async_fan_out=3`, `duet_draft_fan_out=3` |
| DS·PS tree | 둘 다 on | `duet_p1_tree_policy='on'`, `duet_p2_tree_policy='on'` |
| DS 위치당 후보 | 3 | `duet_p1_roots_per_position=3` |
| DS forward scale | 1.5 | `duet_p1_tree_forward_scale=1.5` |
| DS 배분 | dynamic | `duet_p1_tree_allocation_policy='dynamic'` |
| PS forward 예산 | 15 | `duet_p2_budget=15` |
| PS 활성 시작 후보 상한 | 10 | `duet_tree_root_count=10` |
| PS 위치별 top-M 입력 | 28 | `duet_proxy_top_k=28`, 실행 로그의 적용값 확인 |
| 생성 상한 G / 검증 상한 M | DS12/12, PS8/8 | `duet_p1_tree_max_nodes=12`, `duet_p1_tree_verify_nodes=12`, PS는 각각8 |
| 최대 자식 수 C | 3 | `duet_tree_c_tensor=3` |
| 확장 범위 | Last | `duet_tree_expansion='last'` |
| 확장 threshold | DS·PS 모두0 | 두 phase의 start/proxy·confidence threshold 모두0 |
| KV 메모리 계수 | 0.7 | `gpu_memory_utilization=0.7` |
| 토큰 batch 용량 | 16384 | `max_num_batched_tokens=16384` |

`gpu_memory_utilization=0.7`은 **모델 로딩 후 남은 메모리에 대한 KV 비율**이다. GPU 전체 VRAM 사용률70%라는 뜻이 아니다. 다른 GPU의 메모리가 부족하면 이 값 등 자원 설정만 우선 조정하고 변경값을 남긴다. 모델을 양자화하거나 K·C·G·M·exit를 몰래 줄여 같은 조건으로 보고하지 않는다.

DS forward 입력 폭은 검증 문맥 수와 Config의 최소 폭 조건에 따라 결정된다. scale1.5를 적었다고 모든 실행의 물리 폭이 정확히1.5배가 되는 것은 아니다. `[tree-policy-resolved]`의 실제 roots·width·round_widths를 확인한다. PS의 예산15와 활성 후보10도 서로 다른 값이다.

이 preset은 **새 알고리즘의 검증 시작점**이다. 과거65 TPS의 C2·threshold 설정이나 다른 서버의 7B 실험 설정과 같지 않으며, 70B 최적 파라미터라는 뜻이 아니다. B>1은 배치 관련 검증에서만 별도로 변경한다.

## 다른 서버에서 준비할 항목

- 모델 가중치와 데이터셋은 Git에 포함되지 않는다. 경로·GPU 번호·통신 포트·CUDA architecture는 수신 서버에 맞춘다.
- 검증 환경은 Python3.11.15, PyTorch2.8.0, Transformers4.57.1, FlashInfer0.5.2, RTX PRO6000 Blackwell SM120이다. 의존성 기준은 [pyproject.toml](ssd/pyproject.toml)을 확인한다. 다른 환경에서는 자신의 실제 버전도 기록한다.
- 이 서버의 RTX PRO6000 Blackwell은 SM120이므로 `SSD_CUDA_ARCH=12.0`을 사용했다. 다른 GPU에서는 해당 장치 capability에 맞춘다. Blackwell이라는 이름만으로12.0을 선택하지 않는다. 이미 설정된 `TORCH_CUDA_ARCH_LIST`와의 불일치도 확인한다.
- 이전 환경의 calibration/gain, profiler·observer, tokenizer 교체, `LD_PRELOAD` GPU 장애 우회를 그대로 적용하지 않는다. 일반 실행의 `SSD_TREE_EXPANSION_POLICY`는 unset 또는 `q_path`다. `SSD_TREE_LEGACY_CALIBRATION=1`은 과거 재현에만 사용한다.
- `SSD_DEBUG_TOKENS=0`을 사용한다. Python `-O` 여부로 토큰 디버그를 제어하지 않는다. `SSD_TREE_EXEC_WARMUP=all`은 이번 검증과 같은 초기 준비 조건이다.
- 이 서버의 B1 SM120 넓은 tree verify graph는 기본256MiB workspace를 사용한다. 과거96MiB 기본이나 서버별224MiB override를 무조건 복사하지 않는다. workspace는 **개별 graph 기준**이라 전체 추가 VRAM이256MiB뿐이라는 뜻도 아니다. 부족하면 모델·노드·page shape와 요구 용량을 확인하고 capture 전에 조정한다.
- Tokenizer와 학습 가중치를 바꾸지 않는다. 일반 Llama2의 중복 RoPE 버퍼는 새 로더가 검증 후 처리한다. 모델 family 지원만 보고 scaled-RoPE 등 모든 checkpoint의 정확성이 확인됐다고 판단하지 않는다.

이번 native 검사에서 사용한 snapshot은 다음과 같다. 같은 이름의 다른 revision을 쓰면 별도로 기록한다.

| 모델 | Snapshot revision |
|---|---|
| `facebook/layerskip-llama2-70B` | `0a1815fcbd11543b0de227aa8dcbced952149411` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | `fe8a4ea1ffedaf415f4da2f062534de366a451e6` |
| 일반 모델 대조 `meta-llama/Llama-2-70b-hf` | `3aba440b59558f995867ba6e1f58f21d0336b5bb` |

## 실행 예시와 검증 순서

저장소 루트에서 사용한다. 아래 `/path/to/...`는 반드시 수신 서버 경로로 바꾼다. GPU는 실제 사용 허가를 받은 장치만 지정한다.

```bash
export DUET_PY=/path/to/venv/bin/python
export DUET_TARGET=/path/to/layerskip-llama2-70B/snapshot
export DUET_DRAFT=/path/to/TinyLlama-1.1B-Chat-v1.0/snapshot
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/processed_datasets
export SSD_CUDA_ARCH=12.0  # SM120일 때만 이 값 사용
export TORCH_CUDA_ARCH_LIST="$SSD_CUDA_ARCH"
export PYTHONPATH="$PWD/ssd:$PWD/ssd/tests"
export PYTHONDONTWRITEBYTECODE=1
export SSD_DEBUG_TOKENS=0
export SSD_TREE_EXPANSION_POLICY=q_path
export SSD_TREE_LEGACY_CALIBRATION=0
export SSD_TREE_EXEC_WARMUP=all
export SSD_DIST_PORT=29895  # 서버에서 충돌하지 않는 포트
export OMP_NUM_THREADS=2
```

1. **회귀 검사**: CUDA GPU 한 개가 필요하다. 새 결과 디렉터리를 사용한다.

   ```bash
   CUDA_VISIBLE_DEVICES=2 "$DUET_PY" -B \
     results/mlsys_ver01/extensions/check_suites.py \
     --output-dir results/local_ver10_checks
   ```

   이 서버의 결과는341개 중340통과·SM120 미지원 SGL 검사1개 제외였다. 실패·오류·예상하지 못한 skip이 없는지와 `CHECKS_PROVENANCE.json`의 `source_unchanged`를 함께 확인한다. 다른 GPU에서는 지원 범위와 skip 개수가 달라질 수 있다.

2. **70B 기본 기능 검사**: 네 고정 입력, 출력 최대32, seed42다. 실제로 DS·PS hit 경로가 실행되는지 확인한다.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2 "$DUET_PY" -B \
     results/ver10_integration_20261015/probe.py \
     --target "$DUET_TARGET" --draft "$DUET_DRAFT" \
     --output results/local_ver10_default.json
   ```

   결과의 `status=complete`, `source_unchanged=true`, 정상 종료 코드와 runtime manifest를 확인한다. 이32토큰 검사를 full dataset 결과나 TPS 개선 근거로 사용하지 않는다. Frontier 확인은 `--expansion frontier`와 다른 output 파일을 사용한다.

3. **논문 조건 실행기 연결**: 아래처럼 preset과 SamplingParams를 사용한다. 기존 논문 데이터 로더·prompt 형식·문항 ID를 유지한다.

   ```python
   import os
   import random
   import torch
   from ssd import LLM, SamplingParams
   from ssd.presets import duet_70b_paper_kwargs

   def main():
       random.seed(42)
       torch.manual_seed(42)
       kwargs = duet_70b_paper_kwargs(os.environ['DUET_DRAFT'])
       kwargs['stochastic_only'] = True  # 이 실행은 T>0만 평가
       llm = LLM(os.environ['DUET_TARGET'], **kwargs)
       params = SamplingParams(temperature=0.7, max_new_tokens=1024)
       try:
           # 기존 논문 데이터 로더가 만든 prompt_token_ids를 사용한다.
           # outputs, metrics = llm.generate(prompt_token_ids, params)
           # raw 출력·설정·문항 ID·metrics를 새 결과 경로에 저장한다.
           pass
       finally:
           llm.exit(hard=False)

   if __name__ == '__main__':
       main()
   ```

   위 Python은 **기존 실행기에 연결할 설정 예시**이지 데이터 평가 스크립트가 아니다. `bench.py --all`의 네 데이터셋 조합을 기존 Spec-Bench 전체 실험으로 대체하지 않는다. 본 평가의 출력1024와 전체 문맥2048은 서로 다른 한도다.

4. **성능 비교**: 먼저 동일한 소규모 문항 조합으로 해당 구간 지연과 전체 decode 시간을 확인한다. AL·cache hit 영향은 이후 동일 문항 집합의 충분한 데이터로 평가한다. 먼저 seed42로 조건을 완주하고 반복 seed를 추가한다. Last/Frontier 비교에서는 정규화·K·C·G·M·threshold·모델·온도·데이터를 고정한다. ver0.2와 ver1.0의 기본 설정 전체 비교는 순수한 확장 범위 비교와 별도로 표시한다.

PS가 target 종료를 넘기는지 분석할 때는 평균 시간뿐 아니라 **초과 빈도와 지연 분포**를 보고한다. profiler를 켠 실행은 성능 표의 일반 실행과 분리한다. 분석기는 [duet_profile_analysis.py](ssd/bench/duet_profile_analysis.py)를 사용하고, 요청별 정렬 실패나 긴 step을 임의로 제외하지 않는다.

## 결과 판정과 수정 원칙

- **현재 strict greedy AR 비교는9/12 일치**다. LayerSkip의 최초 불일치 두 문맥에서 AR/tree의 작은 logit 차이로 동점이 생겼고, 각 실행은 자기 target argmax를 출력했다. 원인 진단과 전체 토큰열 일치 보장을 구분한다. `run_models.py`가 이 비교로 실패하면 성공으로 바꾸지 말고 최초 분기 문맥·logits를 확인한다.
- 무디코딩 `context_limit_guard` 기록은 원시 데이터에 보존한다. TPS·AL·hit가 미정의인 기록을0으로 넣지 않는다. [mlsys_request_reporting.py](ssd/bench/mlsys_request_reporting.py)의 유효 분모·문항 내 turn 평균·공통 비교 문항 규칙을 유지한다.
- P1/P2 AL은 recovery token을 포함하는 보고 지표로 맞춘다. `accepted_lens_phase*_hit` 같은 기존 raw 필드는 recovery 제외 의미이므로 그대로 평균내서 섞지 않는다.
- 긴 입력에서의 보수적인 문맥 보호 종료는 남아 있다. 보호 조건 삭제나 상한 증가로 검증을 통과시키지 않는다. 변경이 필요하면 별도 엔진 수정과 경계 검사를 수행한다.
- NCCL 종료 경고가 관측됐지만 이번 실행은 정상 종료 후 GPU 메모리를 해제했다. 새 서버에서는 종료 코드뿐 아니라 worker 종료·메모리 해제도 확인한다. 다른 사용자의 프로세스나 semaphore를 일괄 삭제하지 않는다.
- 기본값을 바꿀 때는 Config·CLI·preset·graph 경로·문서·검사를 함께 갱신한다. 실제 proposal q·target 수락과 보정 sampling·RNG를 후보 점수 개선을 이유로 바꾸지 않는다.
- 실행 중 runtime을 수정하지 않는다. 원본 로그·실패 기록을 보존하고 새 attempt에서 재실행한다. 모델 revision, Git 커밋·dirty diff 또는 소스 해시, 적용 설정, 환경, seed, 문항 ID, 원시 결과를 함께 남긴다.

## 담당자에게 전달할 요약

```text
MLSys-ver1.0을 별도 worktree에서 검토해 주세요.
구현 기준은 b8eac4cb이며, MLSys-ver0.2@1b93104b 위에 통합한 버전입니다.
원격 게시 여부와 최종 전달 커밋은 먼저 확인해 주세요.

MLSYS_VER1_HANDOVER.md부터 읽고 ssd.presets.duet_70b_paper_kwargs를 사용하세요.
기본은 LayerSkip70B BF16, B=1, target TP2+draft GPU1개, exit56,
K1/K2=8/4, 출력1024, 문맥2048, T=0.7, DS/PS 모두 tree입니다.
후보식은 혼합 위치식 + e(1-q), 정규화는 top-M입니다.
Tree는 calibration 없는 q-path, Last 기본/Frontier 옵션, C3,
DS G/M=12/12, PS G/M=8/8, threshold0, 생성 순서 prefix이며 rerank는 없습니다.

기존 연구 launcher나 파일 전체를 덮어쓰지 말고 서버 경로·GPU·환경만 먼저 맞춰 주세요.
340개 회귀 통과와 70B 기능 검사는 완료했지만, strict AR 일치는9/12이며
이 통합본의 full dataset TPS 개선은 아직 별도 검증 대상입니다.
추가 수정은 원본 결과를 보존하고 독립 커밋으로 남겨 주세요.
```

상세 출처는 [통합 내역](MLSYS_VER1_INTEGRATION.md), [tree 알고리즘](TREE_EXPANSION_FINAL.md), [검증 보고서](results/ver10_integration_20261015/REPORT.md), [현재 서버 자료 목록](results/ver10_integration_20261015/SOURCE_INVENTORY.json)을 참고한다.
