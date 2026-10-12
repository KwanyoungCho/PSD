# 다른 서버에서 검토하는 순서

1. `feat/duet-mlsys-coverage`의 이 문서가 있는 최신 commit을 가져온다.
   먼저 상위 `MERGE_REVIEW.md` 27절과 이번 `REPORT.md`를 읽는다.
2. 코드 적용 없이 저장된 근거를 재생성한다. 저장소 root에서 다음을 실행한다.

   ```bash
   python results/mlsys_coverage/round7/archive.py --restore
   python results/mlsys_coverage/round7/analyze.py
   python results/mlsys_coverage/round7/stratify.py
   python results/mlsys_coverage/round7/profile_analysis.py
   python results/mlsys_coverage/round7/make_figures.py
   python results/mlsys_coverage/round7/make_report.py
   ```

   GPU는 필요 없다. 그림 생성에는 Matplotlib/Numpy가 필요하다. 복원 시
   gzip·원시 파일의 SHA256을 모두 검사하고, 이미 있는 다른 내용의 파일은
   덮어쓰지 않는다. Runtime symlink를 복제할 필요가 없도록 모두 상대경로
   archive에 저장했다. `RUN_INDEX.csv`는 실행별 간단한 비교표다.
3. `PATCHES.md`에 따라 원 논문 `a82f7d2`의 별도 checkout에서
   `01_chain_fast.patch`, `03_kv_correctness.patch`를 각각 검토한다.
   `02_tree_graph_candidate.patch`는 효과가 재현되지 않은 실험 후보다.
   현재 systems branch 전체를 먼저 merge한 뒤 이것을 원본 대조군으로
   쓰면 이번 실험과 다른 질문이 된다.
4. GPU를 다시 쓰는 대조는 새 결과 디렉터리를 사용한다. 각 `*/plan.json`
   에 정확한 seed, 모델/입력/출력 설정과 옵션이 있다. 모델 경로·질문 파일
   경로·`PAPER_CHECKOUT`을 해당 서버의 경로로 바꾸고, 실제 사용 가능한 GPU
   pair와 port/SHM namespace를 정한다. `run_pair.py`는 시작 시 GPU가 이미
   사용 중이면 실행을 거부한다. 기존 완료 결과를 덮어쓰지 않는다.
5. 이번 준비용 Python은 `/home/chokwans99/PSD/ssd/.venv/bin/python`이었다.
   원 코드 helper는 targetTP1/draft1의 `num_gpus=2`를 사용한다. 원 논문의
   70B/Blackwell TP2는 기존 production runner에서 최소 patch 전후를 대조해야
   하며, 이 helper의 device 설정을 확인 없이 그대로 옮기면 안 된다.

## 재실험에서 유지할 대조 조건

- 같은 GPU pair·attention backend·dtype·모델 weight·입력·sampling·root/tree
  parameter로 비교한다. 가능하면 A→B/B→A를 교차하고 독립 process로 반복한다.
- TPS token/time, AL, hit는 동일한 nominal-B·non-boundary step을 사용한다.
  Tail과 terminal step을 포함한 raw TPS도 함께 보관한다. 실제 반환한 token을
  넘겨 세지 않는다. GPU 미세 구간만 빠르다는 이유로 채택하지 않는다.
- G>M, 실제 온도 적용, 모델 wire 호환 등 이미 승인된 수정은 양쪽에 동일하게
  적용해 verification 최적화의 효과와 분리한다. 이번 주 대조는 G=M/T1이라
  그 기존 결함을 피하면서 원래 정책을 유지했다.
- Prefix 문제를 평가할 때는 출력이 정확한 cold/cached를 비교한다. 잘못된
  KV를 읽는 과거 결과는 latency baseline이 아니다. Tree glue 교체와 prefix
  재사용을 한 항목으로 묶지 않는다.
- Chain fast는 SSD에도 같은 방식으로 적용한다. 이것으로 생긴 속도 향상을
  DUET 알고리즘만의 기여로 발표하지 않는다.

`ANALYSIS.json`의 source commit/parameter/hash, 각 raw의 source SHA와 env,
`ENVIRONMENT.json`, `REMOTE_BASELINE.json`, `FAILURES.md`를 보존한다.
두 tree+chain-fast 후보는 unified 비교의 원본 대조군을 공유하며, plan의
`control`이 해당 실행 이름을 명시한다. 같은 control을 새로운 반복으로
중복 집계하지 않는다.
