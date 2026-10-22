# ver1.7 best-3 검증

기존 논문 표에서 정한 seed **2, 0, 10**을 유지한다. 이번 결과로 seed를 다시
선택하지 않는다. 측정은 각 seed의 수정 전후를 같은 PIX GPU 쌍에서 수행한다.

| 항목 | 고정 조건 |
|---|---|
| 이전 / 수정 | `4c7e43382db6215166265ed1f19d0a5d6bdd72b3` / `0654bd68` |
| 모델 | LayerSkip-Llama2-7B + JackFram/llama-68m |
| dtype / batch / TP | BF16 / B1 / target1 + draft1 |
| Exit / K | exit21 (0-based), K1=10, K2=5 |
| P1 | 위치당4 roots, W56, G=M23 |
| P2 | R24, W28, G=M15 |
| tree | 두 phase on, last expansion, C3 |
| 실행 | graph on, async proxy send on, replica/별도 proxy stream off |
| 데이터 | Spec-Bench 480문항·560턴; 원래 best-3의 저장된 입력 ID와 순서 |
| 생성 | 문맥2048, 출력1024, `min(1024,2048-input_len)`, 자연EOS, 기존 context guard |
| sampling | Target/Draft T0.7, top-p1 |
| warmup | 560턴 안의 첫3턴; 성능 집계에서 제외 |

`INPUTS_FULL.json`은 과거 best-3 실행의 입력이다. 최근 다른 진단에서 다시
토큰화한 파일과 token ID가 다르므로, 문자열이 같다는 이유로 교체하지 않는다.
이 파일의 SHA와 실행별 실제 input ID를 모두 검사한다.
원격에는 원래 파일을 무손실 압축한 `INPUTS_FULL.json.gz`를 저장한다.
`prepare_adapter.py`가 원본과 동일한 byte/SHA의 JSON을 복원한다.

## 구성

- `campaign.py`: 3개의 GPU 쌍에서 각 seed의 before/after를 순서대로 실행한다.
  seed0만 after부터 시작한다. 각 실행의 소스와 측정 helper SHA를 전후 검사하며,
  실패한 결과를 덮어쓰지 않는다. 외부 GPU 작업과 겹치면 자신의 실행만 중단한다.
- `prepare_adapter.py`: 기준 커밋에서 69개 측정 helper를 복원하고 필요한 3개
  변경본을 적용한다. 런타임 엔진은 복제하지 않는다. 기존 파일이 다르면 덮어쓰지
  않고 실패한다. 전체 helper의 정확한 SHA는 `ADAPTER_MANIFEST.json`에 있다.
- `benchmark_wall.py`, `observe_runtime.py`, `capacity_adapter.py`: 양쪽에 같은
  측정 경로를 사용한다. Profile은 끄고 dtype/graph/용량은 초기화 시 감사한다.
- `question_mean.py`, `report.py`: 턴 비율 → 문항 내 평균 → 문항 평균 → seed 평균.
  공통 완전 문항과 실행별 전체 유효 문항, 필터 적용/미적용을 구분한다.
- `check_overhead.py`, `check_batch_roots.py`, `reproduce_review.py`: 전체 성능과
  별개로 코드 오류 및 작은 구간의 비용을 검증한다.

## 로컬 재현 절차

실행 조건과 경로는 `../../results/mlsys_ver17/best3_20261023/PROTOCOL.json`에
보존한다. 현재 controller는 orion의 경로·CPU affinity·GPU pair·모델 경로를
명시한다. 다른 서버는 이를 해당 서버에 맞게 준비하고 새 결과 디렉터리에서
실행해야 한다. 이미 완료된 결과 디렉터리를 재사용하여 덮어쓰지 않는다.

```bash
python3 -B experiments/ver17_best3_20261023/prepare_adapter.py
# PROTOCOL의 before/after를 각각 지정된 커밋으로 별도 worktree에 준비한다.
# 두 모델과 CUDA 환경, GPU의 PIX 연결 및 사용 가능 여부를 확인한 뒤:
python3 -B experiments/ver17_best3_20261023/campaign.py
```

실행 중 엔진·helper·입력·PROTOCOL을 변경하지 않는다. 각 실행의 `PROVENANCE.json`,
`ACTUAL_target.json`, `ACTUAL_draft.json`, `CHECKS.json`으로 실제 조건을 확인한다.
문맥 보호 종료를 일반 오류 또는 0 TPS로 바꾸지 않는다. 정상 종료·전체 UID와
순서·출력 상한·dtype·양 phase tree·graph·소스 불변·선점 없음이 모두 통과해야
완료로 취급한다.

`ACTUAL_draft.json`은 ModelRunner 초기화 직후의 관측이므로 나중에 생성되는
tree executor의 `rng_domains`가 빈 객체일 수 있다. 이를 tree 미사용 또는
RNG 검사 결과로 해석하지 않는다. Tree executor 준비는 `process.log`의
P1/P2 warmup 및 policy 기록으로 확인하고, RNG 분리는 고정 소스 SHA와 별도
실제 CUDA executor/graph 회귀 검사로 확인한다.

## 해석 범위

이번 A/B는 P1 제외, RNG, score wire, scheduler 수정을 묶어서 비교한다.
B1 성능 실험만으로 B>1 요청 누락 문제의 영향까지 측정했다고 주장하지 않는다.
작은 구간이 빨라져도 후보 및 난수열 변화로 AL/TPS가 감소할 수 있다.
원시 결과와 불리한 seed도 모두 보존하며, 과거 v1.4와의 차이를 이번 오류 수정
하나의 효과로 해석하지 않는다.
