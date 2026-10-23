# MLSys ver1.8 실행·인수인계

기준: `MLSys-ver1.7@521810df73d2add8eb42564c4c2e769c881fdb00`. 새 브랜치는
`MLSys-ver1.8`이다. 기존 작업 트리와 다른 서버의 checkout은 자동 변경하지 않는다.

## 이번에 반영한 내용

1. **P1/P2 C를 native 설정으로 분리했다.** `duet_p1_tree_c_tensor`,
   `duet_p2_tree_c_tensor`와 동일 이름의 `bench.py` CLI를 추가했다.
   `None`은 기존 `duet_tree_c_tensor`를 상속하므로 기존 호출은 같은 설정을 유지한다.
   실행기는 생성 시 한 번 C를 읽으며, 공유 verifier/proxy의 sibling capacity는
   `max(C1,C2)`로 잡는다. B>1 chain arm은 해당 phase의 C를 명시적으로 1로 바꾼다.
   기존 초기화 어댑터와 달리 Config 객체 교체나 실행기 monkeypatch가 필요 없다.
2. **성능 측정 기본 진입점**은 `ssd/bench/duet_measure.py`다.
   기본 `--measurement-mode performance`는 import/capture 전에 모든 profiling,
   forward probe, token debug를 끈다. CUDA Graph·fused target heads·async proxy send는
   켜고 exit replica/side proxy stream은 끈다. 예전 실험용 `DUET_REVIEW_*` 설정도 제거한다.
   `--measurement-mode timeline`을 지정할 때만 DUET CUDA event와 forward probe를 설치한다.
   GPU별 forward 타이밍을 읽는 작업은 성능 모드에 없다.
3. **최근 측정 조건을 명시적 7B B1 preset으로 제공한다.** 아래 표는 이 실행기의 기본값이다.
   일반 `Config`나 다른 모델용 `bench.py`의 기존 tuning/dtype 기본값을 전부 덮어쓰지는 않는다.
   API용 `ssd.presets.duet_7b_measurement_kwargs(draft, ...)`도 같은 파라미터를 반환한다.
   이 함수는 Config 인자만 반환하므로 환경을 설정하는 측정 실행기와 구별한다.
4. **Draft KV 예약량을 native 입력으로 옮겼다.** `draft_gpu_memory_utilization=None`은
   기존 자동 규칙, 이 preset은 최근 실험과 같은 `.35`다. target은 `.45`다.
   이는 초기 KV/graph 메모리 예약이며 노드 선택·확률·계산 예산을 바꾸지 않는다.
5. **문항 평균 집계와 timeline 표시를 정식 도구에 반영했다.** 성능 결과는
   `QUESTION_MEAN.json`과 `QUESTION_MEAN_RAW.json`에 별도 저장한다. Timeline 표도
   턴 내 지표 → 문항 내 턴 평균 → 문항 평균으로 계산하고 metric별 분모를 저장한다.
   Target의 fused graph 안에 포함된 LM head를 legend에 표시한다. 연속 두 step,
   draft deadline, signed margin, 실제 forward probe는 기존 상세 도구를 사용한다.
   context guard로 decode하지 않은 요청은 profile epoch에 넣지 않는다.
6. **완료된 C/dtype 12회 및 서버 대조 2회의 결과를 동봉했다.**
   [결과 보고서](results/mlsys_ver18/evidence/FINAL_REPORT.md),
   [결과 그림](results/mlsys_ver18/evidence/RESULTS.pdf),
   [seed별 표](results/mlsys_ver18/evidence/SEED_REPORT.md),
   [PIX/NODE 대조](results/mlsys_ver18/evidence/PIX_NODE_REPORT.md).
   대용량 run/profile 원본과 서버 전용 controller는 새 배포에 복제하지 않았다.
7. Release runtime hash 검사 대상을 ver1.8로 갱신했다. ver1.7의 exact FP32 score
   wire와 맞지 않던 과거 log-bin 테스트는 완화하지 않고 **정확한 bit 보존 검사**로 수정했다.

Residual stochastic 검증, ordered sampling without replacement, 부모 q 전달,
확장/예산 배분 수식, G=M 규칙, P1 전체 자식 제외, phase/context/lane RNG 분리,
정확한 0점수 wire 및 scheduler queue 수정은 ver1.7의 것을 유지한다.
사후 점수순으로 뽑아 verifier를 교체하는 ver1.5 방식은 포함하지 않는다.

## 성능 측정 기본값

| 항목 | 값 |
|---|---|
| 모델 | LayerSkip-Llama2-7B + JackFram/llama-68M, 경로는 인자로 지정 |
| dtype | **FP16** target/draft weights·KV; `--dtype bfloat16`으로 대조 가능 |
| B / TP | B=1, target TP=1 + draft TP=1, GPU 2개 |
| Exit | 21 zero-based: 22번째 layer 뒤 / 전체 32 layers |
| P1 | K=10, R=위치당 4, continuation W=40, G=M=15, **C=2** |
| P2 | K=5, R=전체 16, W=20, G=M=15, **C=3** |
| Tree | 두 phase ON, last-depth, calibration 없는 q-path score |
| 검증 입력 | G/M은 recovery를 제외한다. 최대 target query는 16행 |
| Sampling | target/draft T=0.7, top-p=1, 자연 EOS |
| 입력 | 전체 SpecBench 480문항·560 raw turns, 기존 token IDs·순서 유지 |
| 문맥 / 출력 | 2048 / `min(1024,2048−입력 길이)`, 기존 context guard 유지 |
| Warmup | dataset 첫 3턴을 생성하되 성능에서는 제외, 추가 3턴을 반복하지 않음 |
| Seed | 기본 42; 비교는 42 → 123 → 2026, 후속 0 → 1 → 2 → … |
| 계측 | 기본 OFF; existing whole-step clock만 사용 |
| 집계 | 턴별 비율 → 문항 내 턴 평균 → 문항 평균; seed 간 동일 가중 |

이것은 최근 네 설정 중 선택한 시작점이며 전역 최적 파라미터라는 주장은 아니다.
현재 진입점은 B1·TP1·seed당 1회로 제한한다. batch 최적화의 일반 실행기는 별도다.

## 다른 서버에서 실행

```bash
git fetch origin MLSys-ver1.8
git worktree add -b work/ver18 ../PSD-ver18 origin/MLSys-ver1.8
cd ../PSD-ver18
python ssd/tools/verify_dispatch/check_release.py

export PYTHONPATH="$PWD/ssd"
export SSD_HF_CACHE=/path/to/huggingface/hub
export SSD_DATASET_DIR=/path/to/datasets
# RTX 4090 예시. 다른 GPU면 실제 compute capability로 변경한다.
export SSD_CUDA_ARCH=8.9
export TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5
export SSD_DIST_PORT=49618

python -O -B ssd/bench/duet_measure.py \
  --target /path/to/layerskip-llama2-7b \
  --draft /path/to/llama-68m \
  --prompts results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json \
  --seed 42 --output /path/to/new_run/seed42/run.json
```

기존 저장소에 추적된 `SPEC_BENCH_FULL.json`은 최근 `INPUTS_FULL.json`과 JSON 내용이
같음을 확인했다. Raw prompt와 paper tokenizer의 token IDs가 들어 있다. 선택한 모델의
vocabulary hash를 실행 시 검사한다. 임의 tokenizer로 새로 encode하거나 template를 씌우지 않는다.

Python 3.11, CUDA 12.8, PyTorch 2.8.0+cu128을 사용했다. `ssd/pyproject.toml` 및 기존
환경 구성을 따른다. 모델 가중치와 Python/CUDA 환경은 Git에 포함되지 않는다.
CPU affinity는 `PAPER_CPUSET=16-23,48-55`처럼 해당 서버 topology를 확인한 뒤 선택한다.
GPU 번호만 인접하다고 같은 연결이라고 가정하지 말고 `nvidia-smi topo -m`을 확인한다.
여러 pair를 병렬로 쓸 때는 각 프로세스에 다른 `CUDA_VISIBLE_DEVICES`, `SSD_DIST_PORT`,
output 디렉터리를 지정한다. 통신 shared-memory 이름은 PID로 구분된다.

`run.json`은 원시 step·출력·유효성/제외 표식을 보존한다. `summary`는 호환용 step 합산이므로
**논문 표의 문항 평균에는 `QUESTION_MEAN.json`을 사용한다.** source별 rate/AL/step/TPS와
지표별 유효 문항/턴 수를 함께 저장한다. 끝의 cap/clip step과 동일 요청·B·hit/miss 구성 내
중앙값 **2배 초과** step은 성능의 시간·토큰·AL·hit에서 함께 제외한다(최소 5개 표본).
`run.json.measurement.duration_filter.excluded_steps`에서 긴 step 제외 개수를 확인한다.
표본 부족·출력 2토큰 미만·decode 없음도 별도 공개한다. 원시 결과는 삭제하지 않는다.

## Timeline을 만들 때만

다른 output 디렉터리에서 위 명령에 다음을 추가한다.

```bash
--measurement-mode timeline --output-tokens 256
```

이 경우 `run.json`에 계측 실행임을 표시하며 비계측 TPS와 합치지 않는다. Figure 렌더링은
CPU만 사용한다(numpy, matplotlib, pypdf 필요). 다음 JSON을 `timeline_inputs.json`으로 저장한다.
경로는 이 JSON 파일의 위치 기준이다.

```json
{
  "schema": "duet-timeline-inputs-v1",
  "skip_initial_steps": 10,
  "datasets": [{
    "model_id": "duet7b",
    "model_name": "LayerSkip-Llama2-7B + llama-68M",
    "setting": "E21 P1 K10 R4 W40 M15 C2 / P2 K5 R16 W20 M15 C3",
    "measurement": "실제 commit / GPU topology / FP16 B1 T0.7 / 사용한 raw turns와 출력 상한",
    "fused_heads": true,
    "runs": [{"outcomes": "timeline/run.json", "profile_dir": "timeline"}]
  }]
}
```

```bash
python ssd/tools/duet_timeline/export_profile_timelines.py \
  --spec /path/to/timeline_inputs.json --output /path/to/new_timeline_bundle
```

P1/P2/Miss 대표 연속 두 step의 PDF·PNG·HTML과 문항 평균 표를 만든다. 다른 서버에서는
bundle 전체를 복사해 `build_model_timelines.py --manifest .../SOURCES.json --output ...`으로
재생성한다. 계측 없는 기존 실행에서 상세 timeline을 복원하거나 계측 overhead를 빼서
비계측 막대로 보정하지 않는다. [자세한 표·deadline 정의](ssd/tools/duet_timeline/README.md).

## 측정 근거와 검증 범위

완료된 14회는 **ver1.7 + 초기화용 phase-C 어댑터**의 결과다. ver1.8 native API로
14회를 모두 다시 측정했다고 쓰지 않는다. 세 seed 네 설정의 공통 문항에서 FP16 C1=2는
167.103 TPS / AL 3.067 / hit 91.403% / step 18.301ms였다. seed123은 19번 PIX,
42·2026은 20번 NODE이므로 이 평균은 설정 탐색용이며 단일 서버 논문 TPS가 아니다.
같은 seed123의 PIX/NODE 대조 2회는 557/557 측정 턴에서 출력 및 기록된 검증 경로가
일치했고 NODE 쪽 TPS가 약 2.7–3.2% 높았다. CPU·clock·동시 부하가 달라 topology만의
인과 효과로 해석하지 않는다. 모든 seed/서버별 수치는 동봉 결과를 따른다.

Native API 전환은 같은 12 raw turns, seed42, FP16, 출력 cap256의 GPU 대조에서
**12/12턴 출력 IDs와 927 step의 source·수락 길이·검증 폭·실제 반환 토큰이 일치**했다.
이는 전체 분포 증명이나 모든 B/dtype 조합의 full dataset 재검증을 대신하지 않는다.
검증 기록: [checks](results/mlsys_ver18/checks/),
[runtime hash manifest](results/mlsys_ver18/RUNTIME_MANIFEST.json).
