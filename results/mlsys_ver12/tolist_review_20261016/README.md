# `fan_out_tensor.tolist()` 생략의 작은 수정 시험

**현재 상태: 성능 개선 근거 부족으로 무효 / 미채택. 사용자 지시에 따라 시험용
before/after worktree를 정리했다. 원래 `MLSys-ver1.2` runtime으로 후속 실험을 계속한다.**
아래 구현·재현 설명은 미채택 시험의 감사 기록이다. Frozen source는
`FROZEN_SOURCES.tar.gz`에 남겼으며, 재현 시899e4eb 기반의 별도 checkout을 복원해야 한다.

사용자는 현재 branch를 최대한 유지하면서 간단한 제거가 가능한지 시험하기를 요청했다.
현재 `MLSys-ver1.2` runtime은 변경하지 않는다. 같은899e4eb 기준의 detached worktree 두 개를
`/data/chokwans99/duet-tolist-review-20261016/before`, `after`에 만들었다.
Candidate는 `ssd/ssd/engine/draft_runner.py` 한 파일45줄 추가/15줄 삭제이며,
이 폴더의 `CANDIDATE.patch`에 보존한다. Benchmark, 검사, 분석 문서는 시험 자료다.

## 실제로 바꾼 것

- B1 P2 tree 고정 graph 경로에서 위치별 fan-out CPU 목록을 생략한다.
  `_update_phase2_layout_inplace`가 이미 만든 GPU `fan_idx_hit`을 사용한다.
- Chain prefix의 glue 가시성은 `column <= root_position`으로 GPU에서 만든다.
  Tree hit의 GPU ancestry override는 원래 값을 그대로 사용한다.
- Executor에서 CPU seed position은 쓰이지 않고 root 수만 필요하므로,
  실제 GPU tensor shape로 root 수를 얻는다. 토큰·점수·root 순서·mask는 유지한다.
- 일반 chain, B>1, EAGLE, stage1/lab 진단은 기존 CPU 목록을 유지한다.
  Executor가 지원하지 않아 eager arena로 fallback하면 정확한 목록과 legacy metadata를
  그 시점에 복원한다. 새 사용자 설정 옵션은 없다.

`tolist()` 한 줄을 지우는 것만으로는 CPU 목록의 기존 소비자 계약을 지킬 수 없다.
위45줄은 그 소비자를 연결하고 fallback을 보존하기 위한 수정이다. Selector의 underfill
검사 `.item()`, num_tokens scalar read, P2 state 판정 readback은 그대로다.

## 성능 측정

- 문제를 본 로컬 orion의 LayerSkip-Llama2-13B + JackFram/llama-68m.
- Target GPU4·5, draftGPU6. 같은 GPU와 CPU affinity16–23/48–55.
- B1/BF16/T0.7, exit26(0-based), K1=K2=3, DS위치당 root3,
  PSroot9, W1=W2=9, C3, G=M40, last-depth 경쟁.
- 이전 진단의 같은12개 입력×2회, 입력512/출력192, warmup3개.
- 실행 순서 before1→after1→after2→before2. 새 inference 네 번의 직접 비교다.
  과거 날짜의 timeline을 before 수치로 재사용하지 않는다.
- Native 상세 이벤트를 사용한다. 작은 CPU audit는 양 arm 모두 같은 wrapper로 기록하며
  추가 GPU 동기화나 timing kernel을 삽입하지 않는다.
- 초기10step/warmup/terminal/cap·clip/실제 compile·capture 영향을 제외하고,
  비교 쌍에서 공통으로 유효한 step만 사용한다. 느리다는 이유로 제외하지 않는다.
- `fanout_cpu_ms`는 layout 갱신 함수가 CPU로 반환될 때까지의 시간이다.
  Candidate의 GPU 작업이 이후 readback에서 완료될 수 있으므로 이것만으로 채택하지 않는다.
  Proxy 수신→PS graph 시작 준비 합과 proxy 수신→cache merge 완료의 PS 전체를 주로 본다.
- 출력 ID, cache source, AL, 검증 폭, prefix 길이를 쌍별로 확인한다.
  UID12개로 묶은 bootstrap은 두 passes와 두 A/B 반복을 함께 유지한다.

## 고정 입력 검증

`microbench.py`는 같은 process에서 동일 count tensor를 두 방식으로 갱신한다.
CPU+GPU profiler trace에서 기존 위치 목록 DtoH5회가 candidate0회가 됐는지 확인한다.
100호출 묶음 전후만 synchronize하므로 GPU에 미완료 작업을 남긴 호출 시간만을
비교하지 않는다. 전체 PS 성능과 별도의 작은 작업 측정이다.

Candidate의 새로운4개 검사는 GPU 위치 map/rope/mask 동일성, real root와 padding 구분,
`tolist()` 금지 범위, fallback 복원, chain/B>1/진단 gate를 확인한다. 기존 P2 준비12개와
dispatch 계약9개도 검사한다. 첫 test 실행은 필수 env인 SSD_HF_CACHE 미설정으로 import에
실패했으며, env를 설정한 후 기능 검사를 실행한다. 최초 로그도 보존한다.

## 재현과 자료

```bash
cd /home/chokwans99/PSD-MLSys-ver0.1
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/tolist_review_20261016/run.py
/home/chokwans99/PSD/ssd/.venv/bin/python -B results/mlsys_ver12/tolist_review_20261016/analyze.py
```

`run.py`는 기존 run을 덮어쓰지 않는다. 재실험은 별도 raw 폴더로 설정한다.
원시는 `/data/chokwans99/duet-tolist-review-20261016/runs`이며, 실행별 COMMAND,
SOURCE_MANIFEST, CPU fanout/state audit, CUDA timeline, telemetry를 보존한다.
최종 결과는 `REPORT.md`/`SUMMARY.json`, 완료 검증은 `CHECKS.json`을 확인한다.
고정 작업 micro profiler를 실제 전체 엔진의 DtoH가0회라는 뜻으로 해석하지 않는다.

이번 결과는13B B1의 지정 설정에서 작은 실행 변경을 확인하는 실험이다.
Full Spec-Bench 품질 검증이나 다른 모델·B>1의 성능 확정은 아니다.
