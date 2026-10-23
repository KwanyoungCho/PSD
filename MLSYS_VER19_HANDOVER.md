# MLSys ver1.9 실행·인수인계

기준은 `MLSys-ver1.8@fc2f0cd93e66d82c148b1b3653c4e1ff74591582`이며,
배포 브랜치는 `MLSys-ver1.9`다. 이번 변경은 준비·응답 조립 오버헤드에 한정한다.
기존 checkout 및 진행 중인 실험을 자동 전환하지 않는다.

## ver1.8에서 달라진 내용

| 변경 | 기본값 | 적용 범위 |
|---|---|---|
| `SSD_SCOPED_DEFAULT_DEVICE` | **1** | 모델 초기화·warmup·CUDA Graph capture 동안에만 CUDA 기본 device context 사용 |
| `SSD_FUSED_RESPONSE_TOPOLOGY` | **0** | 선택 시 topology와 전송용 응답 packet을 한 Triton kernel에서 조립 |
| 실행 기록 | 항상 | startup runtime manifest 및 측정 결과에 두 옵션의 resolved 값 저장 |

**기본 적용한 최적화.** 기존 `torch.set_default_device("cuda")` 후
`torch.set_default_device("cpu")`는 CPU로 돌아와도 Python `DeviceContext` dispatch hook을
남겼다. 명시적 CUDA tensor 연산에도 hook이 반복 호출되어 root 선정, KV 복구, 입력 준비의
CPU 실행 비용이 늘어났다. 초기화 부분을 `with torch.device("cuda"):`로 감싸 초기화 종료 시
호출자의 mode stack을 복원한다. 초기화 중 tensor device와 dtype은 유지하며, dtype은 기존처럼
초기화 완료 후 복원한다. `SSD_SCOPED_DEFAULT_DEVICE=0`은 기존 경로를 재현한다.
매 step에 device context를 들어갔다 나오는 방식이 아니다.

**선택 옵션으로 포함한 최적화.** 기존 응답 조립은 topology를 CPU로 읽은 다음 NumPy/Python
반복문으로 packet을 만들고 다시 GPU로 보냈다. 새 경로는 두 GPU 버퍼를 함께 만들고 기존
host validation snapshot을 그대로 유지한다. 패딩·header·parent·형제 순서·q 참조는 기존과
동일하며, executor가 다음 작업으로 덮어써도 snapshot/packet이 변하지 않도록 독립 버퍼를 쓴다.
`views['_response_topology']`와 `views['_response_packet']`를 기존 송신 경로에 전달한다.
CPU readback 자체를 없앤 것은 아니다. 국소 조립 시간은 줄었지만 전체 step 개선이 확인되지
않았으므로 기본값은 0이다. CUDA/Triton 환경에서만 활성화한다.

검증/샘플링 수식, RNG 스트림, 후보와 노드 선택, tree 확장, G=M, KV 수락 경로는 변경하지
않았다. 추가 진단용 host probe, P1 root CUDA Graph 실험, CPU affinity 튜닝은 반영하지 않았다.
GPU 번호·CPU 코어·모델 경로를 runtime에 하드코딩하지 않았다.

## 채택 근거와 측정 범위

아래 수치는 **배포 전 ver1.8 기반 실험 브랜치의 profile ON 진단**이다. 로컬 RTX4090의
PIX 두 쌍에서 실행 순서를 반대로 한 A/B 및 B/A, 각 실행 첫 3 raw turn warmup 후 동일한
6 raw turn, 출력 상한 512, seed42, FP16이었다. K1/K2=10/5, P1 R=위치당4,
P2 R=16, W1/W2=32/32, G1/G2=31/31, C1/C2=3/3, exit21을 사용했다.
두 쌍 합계 조건당 유효 1,024 step이며, 첫 10 step·cap·clip·마지막 step을 제외했다.
꼬리 지연 진단 목적이라 긴 step의 2배 필터는 적용하지 않았다. 문항 평균을 구한 뒤 GPU 쌍을
동일 가중 평균했다. **현재 기본 G15 preset의 비계측 TPS 결과가 아니다.**

| 비교 | 전 (ms) | 후 (ms) | 해석 |
|---|---:|---:|---|
| Device scope: 준비 합계 | 3.039 | 2.252 | 약 25.9% 감소 |
| Device scope: 전체 step | 19.952 | 19.430 | 약 2.6% 감소, 두 쌍 모두 같은 방향 |
| Device scope: target compute | 16.447 | 16.445 | 실질적으로 동일 |
| Packet 조립: P1 해당 구간 | 0.787 | 0.207 | 국소 조립 비용 감소 |
| Packet 조립: P2 해당 구간 | 0.268 | 0.077 | 국소 조립 비용 감소 |
| Packet 조립: 전체 step | 19.467 | 19.527 | 전체 step 개선으로 주장하지 않음 |

Packet 비교는 두 조건 모두 scoped device ON이다. 각 A/B 쌍의 생성 출력과 phase event가
동일함을 확인했다. Device scope의 준비 합계는 KV 복구 + glue + P1 root + slot + 입력 준비다.
노드 생성 비용 또는 target latency 전체를 뜻하지 않는다. 순간 지연 및 deadline 초과를
모두 제거했다는 결과도 아니다. 논문 TPS는 profile OFF full-dataset matched-seed 평가로
별도 확인해야 한다.

원래 집계 결과와 정확성 검사:

- [Device scope A/B](results/mlsys_ver19/evidence/SCOPED_DEVICE_AB.json)
- [Packet 조립 A/B](results/mlsys_ver19/evidence/RESPONSE_PACK_AB.json)
- [기존 device scope 확인](results/mlsys_ver19/evidence/SCOPED_DEVICE_CHECK.json)
- [기존 packet 정확성·microbenchmark](results/mlsys_ver19/evidence/RESPONSE_PACK_CHECK.json)
- [이번 배포 코드 검사](results/mlsys_ver19/VALIDATION.json)

대용량 run/profile 파일 대신 요약 JSON을 포함했다. 원본은
`PSD-ver18-prep-overhead-20261023/results/prep_overhead/{scoped_device_ab,response_pack_ab}`에
보존되어 있다. 위 실험 코드를 배포 runtime과 혼동하지 않는다.

## 유지한 기본 성능 측정 설정

`ssd/bench/duet_measure.py`는 여전히 **performance 모드**가 기본이고 profile/probe는 OFF다.
이 preset은 7B B=1 용이며 일반 Config의 모든 모델 설정을 바꾸는 것은 아니다.

| 항목 | 기본값 |
|---|---|
| 모델·dtype | LayerSkip-Llama2-7B + JackFram/llama-68M, FP16 |
| B / TP | B1, target TP1 + draft TP1 |
| Exit | 21 zero-based, 32개 중 22번째 layer 뒤 |
| P1 | K10, R=위치당4, W40, G=M15, C2 |
| P2 | K5, R=전체16, W20, G=M15, C3 |
| Tree | P1/P2 ON, last-depth, q-path score, residual stochastic verification |
| Sampling | target/draft T=0.7, top-p=1, 자연 EOS |
| 입력 | SpecBench 480문항/560 raw turns, 기존 token IDs·순서 유지 |
| 문맥·출력 | 2048 / `min(1024, 2048−입력 길이)` |
| Warmup | 데이터 첫 3턴 생성 후 성능에서 제외 |
| Seed | 42 → 123 → 2026 → 0 → 1 → …, 실행당 한 seed |
| 집계 | `QUESTION_MEAN.json`의 문항 평균; 원시 값은 별도 보존 |

G는 recovery를 제외하며 G15의 최대 target 입력은 16행이다. 최근 G31/W32 진단 파라미터는
기본 preset을 덮어쓰지 않는다. 긴 step 제외 기준·제외 수 기록, native phase별 C,
timeline 도구, ver1.7 정확성 수정은 [ver1.8 인수인계](MLSYS_VER18_HANDOVER.md)를 따른다.

## 다른 서버에서 실행

```bash
git fetch origin MLSys-ver1.9
git worktree add -b work/ver19 ../PSD-ver19 origin/MLSys-ver1.9
cd ../PSD-ver19
python ssd/tools/verify_dispatch/check_release.py

export PYTHONPATH="$PWD/ssd"
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/datasets
export SSD_CUDA_ARCH=8.9 TORCH_CUDA_ARCH_LIST=8.9  # RTX4090; 실제 GPU에 맞게 변경
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5  # nvidia-smi topo -m으로 이 서버의 pair 확인
export SSD_DIST_PORT=49619
export SSD_SCOPED_DEFAULT_DEVICE=1 SSD_FUSED_RESPONSE_TOPOLOGY=0

python -O -B ssd/bench/duet_measure.py \
  --target /path/to/layerskip-llama2-7b \
  --draft /path/to/llama-68m \
  --prompts results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json \
  --seed 42 --output /path/to/new_run/seed42/run.json
```

선택 옵션 대조는 위 명령의 `SSD_FUSED_RESPONSE_TOPOLOGY=1`만 바꾼다. 이전 device 경로와
비교할 때는 `SSD_SCOPED_DEFAULT_DEVICE=0 SSD_FUSED_RESPONSE_TOPOLOGY=0`으로 한다.
측정 진입점은 명시한 두 환경값을 덮어쓰지 않는다. Runtime import 전에 설정해야 한다.
병렬 실행은 서로 다른 GPU pair·port·output 경로를 사용한다.
Timeline은 `--measurement-mode timeline`을 명시하고 기존 ver1.8 렌더링 도구를 사용한다.

가중치·CUDA/Python 환경은 Git에 포함하지 않는다. 이번 배포 검사는 Python 3.12,
PyTorch 2.8.0+cu128, RTX4090에서 수행했다. `ssd/pyproject.toml`의 upstream 패키지 버전은
MLSys 배포 브랜치 이름과 별개로 유지한다.

회귀 검사 예시(저장소 루트, 프로젝트 의존성이 설치된 Python):

```bash
PYTHONPATH=ssd:ssd/tests python -m unittest -v \
  test_scoped_default_device test_fused_response_topology \
  test_response_topology test_fused_tree_response test_response_snapshot \
  test_ver18_measurement test_p1_child_exclusion
```

CUDA가 없으면 GPU 테스트는 skip되므로 GPU 검증을 대신하지 않는다. 이번 배포에서는
회귀 테스트 41개가 모두 통과했다(skip 없음). GPU 패킷 비교의 200개 조합, executor 재사용 후
snapshot 보존, 난수 상태 불변, 초기화 device scope 복원을 확인했다. 실제 모델에서는
세 설정 각각 6턴(3턴 warmup), 출력 상한 128로 실행했다. 측정 3턴의 384 출력 토큰과
154개 step의 phase/수락 event가 일치했다(P1 76, P2 63, miss 15); warmup 출력도 일치했다.
짧은 모델 검사는 배포 정확성 검사이며 새로운 3-seed 논문 성능 결과로 사용하지 않는다.
