# Blackwell ver0.2 리뷰 반영 및 재검증

작성: 2026-10-14. 기준 release: `MLSys-ver0.2`의 `76c98f946820179a121b5326d2eb8a2c4e3e3545`.
작업 브랜치: `fix/mlsys-ver02-blackwell-review`.
코드·회귀검사 commit: `8782668` (`fix(mlsys): address Blackwell configuration and execution review`). 결과·보고서는 다음 별도 commit으로 보존한다.

진행 중인 AL 실험의 `/home/chokwans99/PSD-MLSys-ver0.1` 런타임은 변경하지 않았다. 이번 수정은 `/home/chokwans99/PSD-MLSys-ver0.2-review`에서 준비하고, eslab18의 별도 소스 사본과 해당 AL 대기열이 끝난 GPU 2·3 및 4·5에서 검증했다. AL 실험용 q-only 소스와 release 검증 소스를 혼용하지 않는다.

## 결론과 적용 범위

- 리뷰의 설정·workspace·디버그 출력·실패 시 정리 문제는 코드에서 확인하여 수정했다.
- 기존 56 TPS와 과거 약65 TPS를 ver0.2의 성능 회귀로 비교하면 안 된다. 리뷰가 지적한 디버그 출력과 fan-out/MQ_LEN 조건이 다르다. 여기서는 그 비교를 재사용하지 않는다.
- 수치 검사를 강화하면서 **별도의 near-tie 후보 순서 문제를 발견했다**. 단순히 tolerance를 넓히지 않고 dense/flat의 정규화 합계 계산 배치를 통일했다. 수식은 같지만 기존 구현의 FP32 동률 부근 후보 순서가 달라질 수 있다.
- 이 서버의 검증 GPU는 RTX3090(SM86)이다. 70B의 TP2 attention head 모양을 검사해도 **Blackwell SM120의 실제70B 전체 실행 검증을 대신하지 못한다**.
- Blackwell 서버 전용 `report.py`와 검증 adapter는 이 작업 트리에 없다. 이식 가능한 상태 판정 helper와 적용 절차를 제공했으며, 다른 서버 파일을 직접 수정했다고 보고하지 않는다.

## R1–R10 반영표

| 리뷰 | 확인한 기존 동작 | 반영 | 검증 및 제한 |
|---|---|---|---|
| R1 | `__debug__`가 상세 토큰 출력까지 활성화. 일부 필수 설정/문맥 검사가 `-O`에서 사라짐 | `SSD_DEBUG_TOKENS=1`일 때만 토큰 출력과 문자열 변환. Config, scheduler 문맥/출력, 주요 host shape와 초기화 조건은 명시적 예외 | `-O`/일반 실행의 잘못된 exit 거부, AR/speculative 단계의 출력 집계·RNG 검사. 저장소의 모든 assert를 변경한 것은 아님 |
| R2 | DS<F 검사 때문에 별도 adapter가 F3→4로 변경 | 명시 split K1/K2와 양수 P2 예산이 있으면 DS=F 허용. 예산을 F−DS로 유도하는 경우에는 DS<F 유지 | F=DS=3, P2=15, K1/2=8/4에서 `[3]*13`, MQ_LEN=39를 target/draft 모두 확인. 원격 adapter의 자동 F 증가 코드는 그 서버에서 삭제 필요 |
| R3 | draft용 `dataclasses.replace` 후 exit56을 draft22층과 비교 | 내부 runner role과 원래 target config를 전달. exit는 0-based target 층 범위로 검사 | target80/draft22에서 exit56 통과, exit80 거부. Llama target/Qwen2 draft, head_dim 및 runtime dtype 계약 검사 |
| R4 | cached-prefill fixture가 SGL 강제, FlashInfer plan 생략 | auto/FlashInfer/SGL 검사를 분리. FlashInfer는 runner처럼 사전 plan. SM120의 SGL만 명시적 unsupported skip | FP16/BF16, cached+uncached 혼합, 비연속 물리 page와 dense reference. 기존 auto-SM120 fallback 자체를 새 기법으로 주장하지 않음 |
| R5 | 새 batch tree workspace128 MiB는 보고된188 MiB 요구보다 작음. draft64 MiB도82.25 MiB보다 작음 | batch target/glue 기본256 MiB, P1/P2 draft 기본128 MiB. 양수 정수 MiB 검사, 실패 시 역할·B·width·page·head·dtype·TP·가용량 및 원래 planner 오류 보존 | 고정 주소와 wrapper별 plan 상태 보존. B1/2/8, FP16/BF16, head32/KV4/d128 모양, graph 교대와 KV 격리 검사. 256 MiB는 실제 Blackwell B8 전체 shape 충분성 보증이 아님 |
| R6 | packed tolerance만 보면 raw 반올림과 순서 문제를 구분 못함 | raw tolerance 유지, 양수/zero mask·ID·위치·형제순서·출력·RNG 별도 검사, packed는 최대1 bin. 추가로 canonical normalization 적용 | 아래 발견/수정 및 component 비용 참조. 기존 모든 출력과 bitwise 동일하다고 주장하지 않음 |
| R7 | rank0 초기화/prewarm 예외가 정상 atexit 등록 전에 나면 worker 잔류. watchdog은 draft만 감시 | 초기화 예외 시 소유 child만 terminate/join/kill 후 최초 예외 재전파. target TP worker도 watchdog 감시 | 생성자 실패 및 target/draft worker 종료 검사. 다른 사용자의 PID, GPU reset, 전역 `/dev/shm` 삭제 없음 |
| R8 | 원격 report가 마지막 job명만 보고 실행 중 표시 | `mlsys_job_status.py`: PID 생존+heartbeat 기반 running/waiting/stopped/drained/needs_attention/complete, 검증 수준 구분 | CPU 상태 검사. 원격 report 호출부 이식은 별도 남음. 기존 무디코딩 분모 규칙은 변경하지 않고 회귀 검사 |
| R9 | tokenizer Python ABI/의존성 명세 부족 | `sentencepiece==0.2.0`과 uv.lock 갱신. 시작 로그에 ABI/라이브러리/dtype/head/vocab 기록 | 공용 venv 변경 없이 eslab18 Python3.11 전용 deps 디렉토리 사용. 실제 생성은 사용자 지정 Llama2-7B+AMD135M에 한정. Qwama 생성 재검증은 미실시 |
| R10 | 요청 정책과 실제 executor fallback 구분 부족 | rank별 runtime JSON 로그, executor별 요청/실제 정책·fallback 이유·calibration hash·물리 width 기록 | release의 complement/full/.25 및 calibration 기본값은 이 리뷰로 바꾸지 않음. q-only AL 실험은 별도 명시 설정. 원격 topm 실험과 동일하다고 부르지 않음 |

## 추가 발견: 양자화 이전의 near-tie 후보 순서

재현 조건: B8, N8, vocab128256, BF16, C2, 각 request의 유효 node4개, T0.3, seed712.

- Dense와 flat의 p, q, token score, 위치 점수는 이 사례에서 정확히 같았다.
- 자식 제외용 padding이 dense에는 request당1칸, flat에는 context당1칸 있어 합계 reduction의 stride가 달랐다.
- 분모: `0.99998939037323` 대 `0.9999894499778748`.
- 후보4168/83356의 최종 점수는 dense에서 동률 `0.00029398620245046914`; flat에서는 한 후보가 `0.0002939861733466387`로 달라져 두 후보 순서가 교환됐다.
- 이는 packed 양자화만의 문제가 아니다. 이전의 “모든 후보 ID가 동일” 결론을 이 GPU/입력에 그대로 확장할 수 없다.

수정:

1. 전체 어휘 정규화 분모는 contiguous `[rows,V]` 형태에서 같은 방식으로 계산한다.
2. Flat은 vocabulary 계산을 유효 context에만 수행하되, 마지막 작은 top-k 점수표를 원래 `[B,contexts,K]` 형태로 복원한 뒤 정규화/전역 순위를 계산한다.
3. raw 오차·후보 ID/순서·검증 출력·RNG 상태는 독립적으로 검사한다. 점수가 비슷하다는 이유로 ID 교환을 허용하지 않는다.

수정 후 240회 persistent graph 비교가 통과했다. 단, 이 검사는 유한한 입력 회귀 검증이며 모든 부동소수점 동률의 수학적 bitwise 보증은 아니다. Target의 실제 correction/accept 분포 수식은 변경하지 않았다. Cache root 우선순위의 극소수 근접 동률은 기존 코드와 달라질 수 있다.

## 정규화 일치 수정의 비용

동일 RTX3090에서 고정 입력의 before→after→after→before 순서로 proxy CUDA graph replay를 측정했다. 모델 forward, 통신, AL/cache hit/TPS는 이 측정 범위에 없다. 각 process 9회 측정, 측정당50회 replay. 아래 값은 각 arm의 process 중앙값을 평균한 값이다.

| B8 ragged 입력 | 수정 전 dense | 수정 후 dense | 수정 전 flat | 수정 후 flat | flat 수정 비용 |
|---|---:|---:|---:|---:|---:|
| vocab32000, FP16 | 1.131 ms | 1.157 ms | 0.839 ms | 0.853 ms | 약+0.014 ms |
| vocab128256, BF16 | 3.867 ms | 3.952 ms | 2.653 ms | 2.712 ms | 약+0.059 ms |

이 항목은 정확한 비교와 수치 일관성을 위한 비용이지 속도 개선이 아니다. 수정 후에도 flat의 proxy 구간은 같은 수정이 적용된 dense보다 약26.3%/31.4% 짧다. 장비 외부 부하를 통제한 논문용 신뢰구간 실험으로 확대 해석하지 않는다.

## 검증 기록

**최종 소스의 회귀 31개 스위트 / 302개 검사 전부 통과, 실패·오류·skip 0개.** `VERIFICATION.json`은 `complete_local_review`이며 실제 SM120/70B 검증 완료를 뜻하지 않는다. 최종 증거는 `checks_release/`, 수치 문제 조사·component 측정은 `checks_final/`, 앞선 실패는 `checks_attempt1/`과 `checks_attempt2/`에 보존했다. 과거 attempt의 ledger는 당시 기록이며 현재 상태로 사용하지 않는다.

| 검증 | 최종 결과 | 범위 |
|---|---|---|
| 전체 회귀 | 31개 스위트, 302개 검사 통과 | RTX3090 SM86, final runtime/test/bench source hash 일치, 실행 전후 변경 없음 |
| Dense/flat 후보·검증 | graph 비교240회 통과 | raw/packed, 후보 ID·순서, 출력·RNG를 각각 검사. 전체 회귀에 포함된 내부 비교 횟수이며302에 더하지 않음 |
| 실제 모델 tree 생성 | B1/B2 × T0/T0.7, 4개 실행 모두 통과 | 각4입력, 출력32토큰, LayerSkip-Llama2-7B + AMD-Llama-135m FP16 |
| AR 기준선 | B1/B2의 T0, 2개 실행 모두 통과 | 각4입력. Tree와 greedy 출력8/8 완전 일치 |
| 총 실제 모델 실행 | 6/6, 반환 코드0 | 모두 최종 소스 hash와 일치. T0.7은 생성·출력 경계 검사이며 분포 동일성의 통계적 증명은 아님 |

실제 모델 기능 검사의 설정은 K1/K2=4/2, F=DS=3, 명시 P2 budget6, MQ_LEN21, exit21, P1/P2 node8/4, C2, q_path다. 이는 작은 진단 구성으로, release의 calibrated tree 기본값에 대한 full 품질 평가나 현재 BF16 AL 캠페인의 대체 검사가 아니다. Qwama는 config 계약만 검사했으며 실제 생성은 사용자 요청대로 Llama2 조합에 한정했다.

- 초기 CPU 설정 검사10개 통과. 최종 suite에는 worker 감시·stride 검사를 추가했다.
- 첫 GPU 검사에서 새 Config 필드가 빠진 기존 fixture와 새 attention fixture의 누락 메서드를 발견해 수정했다. 실패 로그는 별도로 보존한다.
- 강화한 flat 검사에서 위 near-tie 문제가 드러났고, 첫 번째 수정안(마지막 작은 점수표의 layout만 복원)은 충분하지 않았다. 분모 reduction 배치까지 통일한 뒤 통과했다.
- 실제 모델 검사는 별도 adapter 없이 runtime을 호출한다. 짧은4입력/최대32출력의 기능 검사이며 full SpecBench 품질/최적 파라미터/성능 평가가 아니다.
- 일부 정상 종료 로그의 기존 NCCL process-group/semaphore 경고는 남아 있다. 이번 R7은 초기화 실패 시 소유 worker의 제한 시간 내 종료를 검증한 것이며 모든 종료 경고 제거를 주장하지 않는다.

저장된 최종 증거를 현재 checkout과 대조하는 명령:

```bash
tar -xzf results/mlsys_ver02/blackwell_review_20261014/RAW_RESULTS.tar.gz \
  -C results/mlsys_ver02/blackwell_review_20261014
python results/mlsys_ver02/blackwell_review_20261014/summarize_review.py
```

원시 로그·중간 실패·최종 모델 출력은 수정 없이 `RAW_RESULTS.tar.gz`에 묶었다. 해제 후 실제 모델 원시 로그와 출력 ID는 `checks_release/models/`, 전체 검사와 소스 출처는 `checks_release/full_final/CHECKS.json` 및 `CHECKS_PROVENANCE.json`에 있다. 최종 CHECKS 두 파일은 바로 열람할 수 있도록 본 문서와 같은 폴더에도 복제했다. `SOURCE_SHA256.json`에는 이식할 소스·환경 명세의 해시를 별도로 저장했다. `REVIEW_REQUEST.md`는 전달받은 본문을 보존하고 줄바꿈만 LF로 정리했다.

## Blackwell 서버에 이식하는 순서

1. 현재 실행 중 소스를 덮어쓰지 않는다. 이전 manifest와 실패 attempt를 보존하고, 새 checkout/attempt에서 재검증한다.
2. 공통 수정 commit을 가져온다. 서버별 launcher/모델 경로/manifest는 덮어쓰지 않는다.
3. adapter의 `async_fan_out` 자동 증가와 target/draft exit 검사를 우회하던 임시 hack을 제거한다. 원래 F3/DS3/P2예산15/K8+4/MQ39가 rank별 `[runtime-manifest]`에 나오는지 확인한다.
4. `SSD_DEBUG_TOKENS=0`(기본값)으로 실행한다. Python `-O` 여부도 기록하고 A/B 두 arm에서 같게 둔다. 비교 대상 ver0.1에도 동일한 디버그 출력 규칙을 적용해야 한다.
5. 실제70B/TP2/B1·B2·B8에서 선언한 모든 prewarm shape를 확인한다. B1의 기존 `SSD_TREE_VERIFY_WORKSPACE_MB=224`는 별도로 유지한다. batch workspace256 MiB가 부족하면 실패에 기록된 shape/요구량으로 초기 설정을 다시 정한다. 실행 중 graph buffer를 바꾸지 않는다.
6. `test_cached_prefill`을 auto/FlashInfer로 실행한다. SM120의 SGL skip은 지원 불가로 기록하고 통과 건수에 넣지 않는다.
7. 원격 report에서 `mlsys_job_status.queue_state`를 호출한다. PID만이 아니라 해당 attempt의 PID identity와 heartbeat를 확인하고, 마지막 job 이름으로 running을 추정하지 않는다. 진단 성공을 functional/performance 성공으로 올리지 않는다.
8. 원래 모델/입력/정책/파생 shape가 같은 ver0.1↔ver0.2 비교를 다시 수행한다. R6의 수치적 근접 동률 변화가 있는 경우 output·AL·hit 변화와 실행 지연 효과를 분리한다.
9. Qwama 및 실제70B 기능 검증, 480문항/560턴 알고리즘 비교는 각 서버의 별도 계획을 완료한 뒤에만 완료 처리한다.

## 병행 AL 실험과 merge 주의

이번 수정은 release76c98f9를 기준으로 작성했다. 현재 AL 캠페인의 q-only/last-depth/frontier 코드는 그 뒤의 별도 working tree 변경이다. 이번 commit을 원래 AL 경로에 즉시 덮어쓰면 다음 작업의 고정 source hash 검사가 실패한다. 캠페인 종료 후 공통 수정과 연구 코드를 통합하고, Config/정규화/정책 로그의 충돌을 검토한 뒤 같은 회귀 검사를 다시 수행한다.

검증된 실험 결과를 새 runtime에서 얻은 결과로 소급 표시하지 않는다. `REVIEW_REQUEST.md`는 사용자가 전달한 원문이며, 원문의 Blackwell 원시 파일은 이 서버에서 직접 열람하지 못했다.

다른 서버에서 먼저 검토할 때는 `git fetch origin fix/mlsys-ver02-blackwell-review` 후 해당 remote branch를 별도 worktree로 연다. 기존 연구 소스에 반영할 때는 `8782668`의 변경을 검토하여 cherry-pick하고, 결과·문서는 후속 commit으로 분리해서 가져올 수 있다. 이 작업은 `MLSys-ver0.2` 원격 HEAD를 변경하거나 실행 중 AL 작업 트리에 병합한 작업이 아니다.
