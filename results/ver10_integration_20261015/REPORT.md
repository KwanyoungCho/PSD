# MLSys ver1.0 통합 검증

2026-10-15. 기준 `MLSys-ver0.2@1b93104b`, 통합 브랜치 `MLSys-ver1.0`. [항목별 통합 내역](../../MLSYS_VER1_INTEGRATION.md)과 [원본 출처](SOURCE_INVENTORY.json)를 함께 본다. 기존 연구 디렉터리·원시 데이터는 변경하지 않았다.

## CPU와 Blackwell GPU 회귀

workspace 수정과 사용자 결정에 따른 top-M 기본값을 포함한 최종 소스에서 **37 suite / 341개 발견, 340개 통과, 지원되지 않는 SGL 검사 1개 제외, 실패·오류 0**이다. 실행 중 runtime·bench·test 소스의 해시가 변하지 않았다. GPU 2에서 약 243초 동안 검사했다.

- [결과 JSON](topm_release_checks/CHECKS.json), [소스·환경·명령](topm_release_checks/CHECKS_PROVENANCE.json), suite별 로그는 `topm_release_checks/`에 있다.
- SGL Hopper kernel은 SM120에서 지원되지 않으므로 전용 한 검사만 제외했다. auto/FlashInfer, cached prefill, 실제 attention shape·CUDA graph 검사는 통과했다.
- root 온도0.1·0.7·1, 세 token score, full/top-M, h_hat/h_bar/혼합의 독립 수식 검산과 dense/flat/graph 비교를 포함한다.
- Config의 target/draft 역할, CLI, 70B preset, chain/tree/flat 함수와 설정 없는 fallback까지 top-M 기본값이 일치하며, 명시적 full 옵션이 유지되는지 검사한다.
- last/frontier의 DS·PS, CPU rollout/arena, eager/capture, batch, q 보존 및 생성 순서 prefix를 확인한다.
- 무디코딩 guard·유효 분모·turn 집계, 출력 한도, worker 정리, dtype·vocab 계약과 workspace 진단을 포함한다.
- 새 오프라인 profile parser는 sparse tree span, 요청 순서, 누락·중복 거부, 긴 지연 표본 보존을 검사한다.

별도로 기존 실제 chain/tree profile을 새 parser에 연결한 회귀 검사 9개도 통과했다. 기존10개 profile의 모든 분석 대상 step이 정렬됐고, sparse tree fixture의291개 step을 복원했다. 이 검사는 기존 원시 데이터 재분석이며 새 GPU 실행이 아니다.

`checks/`와 `workspace_checks/`는 수정 중 수행한 선행 검사로 `source_unchanged=false`다. `final_checks/`는 workspace 수정 전 소스를 고정해 수행한 339개 검사다. `release_checks/`는 workspace 수정 후, 정규화 기본값 변경 전 340개 검사다. **최종 통합본의 근거는 `topm_release_checks/`**이며 이전 검사 횟수를 더하지 않는다.

## 실제 체크포인트 사전 확인

- LayerSkip Llama2 70B와 원본 Llama-2 70B는 모두 80 layers이며 TinyLlama와 token-to-ID mapping이 일치했다. 토크나이저 교체 없이 native tokenizer를 사용한다.
- 원본70B의 `rotary_emb.inv_freq` 80개가 모두 설정값을 FP16으로 왕복 변환한 결과와 정확히 일치했다. 이 보조 버퍼만 검사 후 생략하고, 학습 가중치는 정상 로더로 읽는다.
- LayerSkip checkpoint 기본 dtype은 BF16, 원본70B는 FP16이다. 이번 probe는 모두 `dtype='bfloat16'`을 명시한다. 가중치 파일을 변환·덮어쓰지 않는다.

## 실제 모델 검증

첫 실행은 B1 tree graph의 FlashInfer 작업 공간 부족으로 초기화 단계에서 종료됐다. [실패 원본](models_attempt1/status.json)을 보존했다. 96MiB 버퍼에197,132,288bytes가 필요했고, 과거 launcher의224MiB override 없이 실행하면서 드러났다. B>1용256MiB 수정과 다른 코드 경로였다.

SM120의 넓은 B1 tree graph 기본을 256MiB로 보완하고 capture planner의 오류 진단도 연결했다. KV보다 먼저 할당하며, 실행 중 captured buffer를 resize하지 않는다.

[재실행 원본](models_attempt2/status.json)의 **8개 추론 프로세스는 모두 종료 코드 0**, 유효한 출력과 고정 runtime 소스 검사를 통과했다. 그러나 마지막 strict greedy 토큰열 비교는 **9/12 일치**여서 runner 전체 상태는 `failed`다. 이 상태를 성공으로 덮어쓰지 않았다.

이 8개 실행과 뒤의 greedy 진단은 **top-M 기본값 확정 전**이며, 모든 실행에서 full/top-M을 명시했다. 각 JSON에 당시 소스 해시를 보존했다. 이후 변경은 정규화 기본값의 일관된 전환 및 해당 회귀 검사·설명 추가이며, 이 원본을 최종 소스의 재실행 결과로 바꾸어 표시하지 않는다.

| Greedy 대조 | AR과 같은 입력 | 전체 토큰열 일치 |
|---|---:|---:|
| LayerSkip70B Last/full | 4/4 | 2/4 |
| LayerSkip70B Frontier/top-M | 4/4 | 3/4 |
| 일반 Llama2-70B Last/full | 4/4 | 4/4 |

### Greedy 불일치의 추가 진단

[동일 문맥의 전체 logits 추적](greedy_diagnosis/DIAGNOSIS.json)에서 AR·Last·Frontier의 원래 출력이 모두 재현됐고, **최초 분기마다 실제 출력 토큰이 해당 실행의 target argmax와 일치**했다. 진단은 별도 프로세스의 측정용 wrapper이며 일반 추론 엔진에 hook을 설치하지 않았다.

| 입력 / 최초 불일치 offset(0부터) | AR의 두 경쟁 토큰 logits | Tree의 같은 두 토큰 logits | 해당 문맥의 전체 어휘 최대 logit 차이 |
|---|---|---|---:|
| 코드 / 16, Last와 Frontier 공통 | 18.125 / 18.000 | 18.000 / 18.000 | 0.125 |
| 번역 / 27, Last | 12.625 / 12.5625 | 12.625 / 12.625 | 0.21875 |

AR에서의 작은 우위가 tree forward에서는 동점으로 바뀌었다. 두 값의 차이 0.125와 0.0625는 각각 그 크기 구간의 BF16 한 단계 간격이다. **관측한 첫 분기는 verifier가 자기 target argmax를 잘못 고른 문제가 아니라 AR/tree forward의 수치 차이와 동점 선택에서 발생했다.** 단, 어느 kernel 연산이 최초 차이를 만들었는지까지 분리한 검사는 아니다.

3건의 불일치 비교는 서로 다른 문맥 2개에 해당한다. 이후 출력 전체의 AR 동등성이나 모든 문맥의 정확성을 증명한 것은 아니며, strict parity **9/12와 실패 상태는 그대로 유지**한다. 일반 Llama2의 4/4 일치도 이 네 입력에 한정한다. 수치·기능 진단 통과를 lossless 전체 검증 통과로 바꾸어 보고하지 않는다.

논문 시작 preset의 B=1·K1/K2=8/4·exit56·tree12/8·C3, target TP2와 별도 draft GPU를 사용한다. 기능 확인을 위해 출력만32토큰으로 줄였으며, full dataset 성능 결과가 아니다.

| 모델 | 조건 | 목적 |
|---|---|---|
| LayerSkip70B BF16 | B1 last/frontier, T0/T0.7, full/top-M | native config·tree·root·graph 경로 실행 |
| LayerSkip70B BF16 | B2 last, T0.7, full | 바뀐 설정의 배치 경로 확인 |
| LayerSkip70B BF16 | B1 AR, T0 | 두 tree 방식의 greedy 출력 대조 |
| 원본70B BF16 | B1 last와 AR, T0 | 보조 버퍼 native loader와 greedy 출력 대조 |

각 설정은 같은 모델 안에서 동일한 네 입력을 사용한다. root 정규화와 tree 범위를 섞은 검사는 경로 커버리지용이며 알고리즘 우열 대조가 아니다. sampler·loader·Config에 실험용 monkey-patch를 설치하지 않는다. GPU 장애 우회 `LD_PRELOAD`도 사용하지 않는다.

### 최종 top-M 기본값의 native 실행

사용자 결정 반영 후 [기본 설정 실행](default_topm_smoke.json)을 추가했다. `probe.py`에 정규화·tree 범위·온도·batch override를 전달하지 않아 **LayerSkip70B BF16 / B1 / Last / top-M / T0.7** 기본값을 사용했다. 네 입력 모두32토큰을 생성했고, DS hit18회·PS hit5회가 관측됐다. 종료 코드0, 소스 해시 불변이며 최종 회귀 검사와 같은 runtime 소스다. 이 소표본의 hit 횟수는 경로 실행 확인용이지 성능 추정치가 아니다.

[로그](default_topm_smoke.log)에 초기화와 실제 설정을 남겼다. 모델 로딩·준비·짧은 추론까지 약47초이며 decode TPS가 아니다. 종료 시 기존 NCCL process-group 정리 경고는 남았지만 프로세스는 정상 종료하고 GPU0·1·2 메모리가 모두 해제된 것을 확인했다. 이 경고의 정리 순서 개선은 이번 통합에서 변경하지 않았다.

최종 회귀 검사는 저장소 루트에서 아래처럼 새 결과 디렉터리로 재현할 수 있다. 70B 실행은 `probe.py --help`를 참고하고 현재 서버의 모델 경로를 전달한다. GPU 사용 가능 여부는 실행 전에 별도로 확인한다.

```bash
CUDA_VISIBLE_DEVICES=2 SSD_HF_CACHE=/tmp SSD_DATASET_DIR=/tmp SSD_CUDA_ARCH=12.0 \
PYTHONPATH=ssd:ssd/tests PYTHONDONTWRITEBYTECODE=1 \
/home/eslab/chokwans99/baseline/.venv-ssd/bin/python -B \
results/mlsys_ver01/extensions/check_suites.py --output-dir /tmp/duet-ver10-recheck
```

## 성능 해석과 남은 범위

통합과 기능 검증은 과거의 65 TPS 또는 새 AL·cache hit 개선을 입증하지 않는다. C·threshold·정규화가 다른 결과를 직접 비교하지 않으며, 성능 표 승격에는 같은 문항·출력 한도와 명시적 정규화에서 측정한 새 결과가 필요하다. 긴 입력의 context guard는 제거하지 않았다.

일반 Llama2 지원을 확인해도 Llama3 scaled RoPE 등 다른 모델 계약까지 자동 검증된 것은 아니다. 대규모 파라미터 sweep이나 기존 논문 결과 교체는 이 통합에 포함하지 않는다.
