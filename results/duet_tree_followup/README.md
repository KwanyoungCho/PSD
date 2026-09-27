# Tree 후속 검증

- [REPORT.md](REPORT.md): 지금까지의 비교 결과와 실제 생성 실험 상태
- [THEORY.md](THEORY.md): C=3 gain 및 사전 fanout 최적화 유도
- [PLAN.md](PLAN.md), [plan.json](plan.json): 미리 고정한 비교 정책/실험 조건
- [comparison.pdf](comparison.pdf): **국소** C=3 비교, online AL이 아님
- [queue_status.json](queue_status.json): GPU 대기/실행/실패 상태와 PID

현재 CPU 분석과 한 GPU에서의 작은 모델 실행기 검사는 완료했다. 실제 target 70B
full inference는 GPU 5개가 필요하며, 사용자가 허용한 자동 대기열에서 실행한다.
`launch.py --wait`는 다른 작업을 중단하지 않고 GPU별 사용 메모리가 1GB 미만일 때
시작한다. Source hash가 바뀌거나 검사가 실패하면 중단하고 기존 attempt를 보존한다.

`STOP` 파일을 이 디렉터리에 만들면 대기 중 또는 다음 job 시작 전에 멈춘다.
이미 실행 중인 job을 강제 종료하지 않는다. 실행 실패 후에는 해당 `runs/*/run.log`를
검토해야 하며, 기존 디렉터리를 지우거나 완료 결과에 이어 쓰지 않는다.

완료 예정 산출물은 `online_analysis.json`, `audit.json`, `frontier_analysis.json`이다.
마지막에 `make_report.py`가 통합 보고서를 갱신한다. `queue_status.state=complete`는
예약한 online/trace 캠페인이 완료됐다는 뜻이며, 별도 미구현/실패한 shadow causal
replay까지 검증됐다는 뜻이 아니다.

24개 tree에서 시도한 draft-only shadow는 known-parent q 재현 기준에 실패했다.
`shadow_q_smoke*/validation.json`과 로그를 보존했고, 정책 효과 분석에서 제외했다.
실제 engine/KV 상태를 공유하는 shadow/super-tree의 다중 라운드 causal replay는
아직 해결되지 않은 별도 항목이다. 기존 hit-only data로 그 결과를 만들어 내지 않는다.

연구 hook은 이 디렉터리에만 있고 production 기본 설정을 변경하지 않는다.
기존 `duet_tree_al_full`, `duet_tree_posthoc`의 frozen 결과는 그대로 보존한다.
