# ver0.2 리뷰 병합 후 검증

대상: `MLSys-ver0.2`, 최종 runtime `f6494f6`. 통합 이력과 다른 서버 지침은 [MERGE_HANDOVER_20261014.md](../MERGE_HANDOVER_20261014.md)를 먼저 읽는다.

## 검증 범위와 결과

**최종 상태: `complete_local_merge_validation`.** 전체 회귀 **33개 suite / 312개 검사 통과**, 실패·오류·skip 0개. Native 실제 모델 **6/6 실행 통과**, B1/B2 greedy 출력은 AR 기준과 **8/8 입력 완전 일치**했다. 실행 전후 runtime/test/bench 소스 해시도 동일하다. `VERIFICATION.json`에 기계 판독 가능한 최종 판정을 보존했다.

| 검사 | 결과 |
|---|---|
| 최종 runtime 전체 회귀 | 33 suite / 312개, 실패·오류·skip 0 |
| Native 실제 모델 | B1/B2 × (AR T0, tree T0, tree T0.7), 6/6 실행 정상 종료 |
| Greedy 출력 대조 | B1 4/4, B2 4/4: AR와 token ID 완전 일치 |
| 연구 adapter 연결 | Last와 Frontier 각4턴/최대32출력, 2/2 통과 |
| 동일 생성 예산 | Last174 + Frontier120 = 294개 exported phase forest. 모두 활성64행·생성192개·root 기록8개 |
| 이전 full 결과 재확인 | 14개 원시 run JSON 해시, 24개 산출물 해시, frozen source200개, 문항 행6,720개, paired 비교15개 일치 |
| 집계 코드 | CPU 회귀5개 통과 |

Forest 수에는 warmup 및 마지막 step 이후 준비 작업도 포함된다. 입력8턴이나 고유 검증 step 수와 같은 의미가 아니다. 예산 조건만 검사하며 이 짧은 실행의 AL로 Last/Frontier 우열을 판정하지 않는다.

검증은 **수정한 코드를 병합한 뒤에도 runtime과 실험 adapter가 동작하는가**를 확인한다. 기존 tree full 실험을 병합본에서 재실행하거나 성능 개선을 새로 측정한 작업은 아니다.

## 병합에서 추가로 해결한 사항

- Config의 정책 시작 로그 충돌을 해결하면서 사용자 요청의 calibration 없는 `q_path`를 유지했다.
- 별도 리뷰 브랜치에서 추가된 runtime manifest가 과거 `reach_gain_frontier`를 기본값으로 표시하던 부분을 `q_path`로 맞췄다. 명시 override는 그대로 기록한다.
- CPU 참조 rollout도 같은 기본값을 사용하도록 했다. 이전 guard는 환경변수를 생략하면 지원 가능한 q-path 요청도 calibrated 모드로 오인해 거부할 수 있었다. 실제 GPU 경로의 알고리즘 변경은 아니다.
- 연구 검사 fixture에 실제 executor의 phase/allocation/width/root/depth/node 속성을 제공했다. requested/resolved/fallback/calibration 로그도 검증한다.
- 두 브랜치의 회귀 목록을 모두 합쳐33개 suite를 구성했다. 기존 calibrated 재현 검사는 환경변수로 해당 모드를 명시한다.

## 기록 읽는 법

- `checks/CHECKS.json`, `checks/CHECKS_PROVENANCE.json`: 최종 전체 회귀와 모든 runtime/test/bench 소스 해시, 실행 전후 동일성, 실제 Python/GPU/명령.
- `models/status.json`, `models/*.json`, `models/*.log`: 별도 algorithm adapter 없이 native runtime을 호출한6개 실행. B1/B2 각각 AR T0와 tree T0/T0.7.
- `adapter/status.json`, `adapter/{last,frontier}.json`: 실제 paper 입력 앞4턴, 최대 출력32, warmup1, B1/T0.7/R=W=8/K1=K2=8/C3/M32로 새 merged runtime과 연구 adapter의 연결 확인.
- `adapter/*_profile/equal_forest.jsonl.gz`: 각 phase의 활성 forward 입력64행, 생성192개, root 기록8개를 검사한 원시 trace.
- `TREE_RESULTS_CHECK.json`: **이전 full 실험**의 원시 JSON14개 SHA256, 최종24개 파일, frozen source200개, 문항별 평균과15개 paired 비교 검산.
- `analysis_tests.log`: 원시 집계/압축 forest 검증과 문항 단위 통계의 CPU 회귀5개.
- `attempt1/`: CPU fallback 기본값 불일치를 발견한 뒤 멈춘 첫 병합 검사 기록. 완료 검사로 합산하지 않는다. 중단 사유는 `attempt1/ATTEMPT1_STOP.json`에 있다. 최종 검사는 새 디렉터리에서 처음부터 실행했다.

## 실제 실행 환경과 해석 제한

- GPU: RTX4090(SM89). 전체 회귀 GPU7, 실제 모델 GPU4·5. 사용 중인 다른 GPU 프로세스는 건드리지 않았다.
- 모델: LayerSkip-Llama2-7B + AMD-Llama-135m. Native 기능 검사와 tree full 연구/adapter 모두 FP16 runtime이다. 실제 dtype는 각 로그의 `[runtime-manifest]`로 확인한다.
- 공용 환경을 바꾸지 않고 `/home/chokwans99/PSD/ssd/.venv/bin/python`을 사용했다. Python3.12.7, torch2.8.0, FlashInfer0.5.2, sgl-kernel0.3.17.post1, transformers4.57.1, SentencePiece0.2.1이다. 저장소의 SentencePiece0.2.0 lock을 새로 설치한 환경의 검사와 구분한다. 이전 리뷰는 별도 Python3.11 환경으로 검증했다.
- Native 기능 검사는4입력×최대32출력, K1/K2=4/2, exit21, F=DS=3, P2 예산6, C2, node/verify8·4다. full Spec-Bench나 파라미터 최적화의 대체 결과가 아니다.
- 일부 정상 종료 시 기존 NCCL process group/resource-tracker 경고가 남는다. 종료 코드와 출력 검사 결과를 함께 보며, 모든 종료 경고가 제거됐다고 주장하지 않는다.
- 검증용 짧은 실행의 TPS/latency를 성능 결과로 인용하지 않는다. 실제70B/Blackwell SM120과 다른 서버의 launcher/adapter/report 통합은 별도 검증이 필요하다.

## 재확인 명령

현재 checkout이 검증한 소스와 같은지 확인:

```bash
python results/mlsys_ver02/merge_validation_20261014/summarize.py
python results/mlsys_ver02/merge_validation_20261014/verify_tree_results.py
```

새 GPU 검사는 기존 증거를 덮어쓰지 않고 새 output 디렉터리에 기록한다. 다음은 repository root 기준이며 Python과 장비 번호는 서버에 맞춘다.

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 \
  PYTHONPATH="$PWD/ssd" SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp \
  python results/mlsys_ver01/extensions/check_suites.py --output-dir /path/to/new-checks

python results/mlsys_ver02/merge_validation_20261014/run_models.py \
  --target /path/to/layerskip-llama2-7b --draft /path/to/AMD-Llama-135m \
  --gpus 4,5 --output-dir /path/to/new-native-model-checks

python results/mlsys_ver02/merge_validation_20261014/run_adapter.py \
  --target /path/to/layerskip-llama2-7b --draft /path/to/AMD-Llama-135m \
  --gpus 4,5 --output-dir /path/to/new-adapter-checks
```

GPU pair를 공유하는 native와 adapter 명령은 순서대로 실행한다. 원래 full 실험 소스의 재현과 merged runtime 검사는 별도로 기록한다.
