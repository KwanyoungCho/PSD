# v1.5 연속 seed 성능 측정

사용자 중단 전까지 로컬에서 전체 Spec-Bench를 seed별로 반복한다. 실행 중 생성되는 REPORT.md, STATUS.json, SUMMARY.json, SUMMARY.csv, validated/는 Git에서 제외한다. 런타임 Git 상태를 clean으로 유지하기 위한 것으로, 원시 결과는 삭제하지 않는다.

현재 로컬 raw는 `/data/chokwans99/duet-ver15-score-seeds-20261019`다. 완료된 seed만 독립 검증 후 집계하며 미완료 seed나 v1.4 seed를 평균에 포함하지 않는다. 입력, 실행 명령, 소스 해시, GPU telemetry, 출력 ID와 원시 step 기록은 raw/runs/seedN에 보존한다.

실행 규칙·정확성 근거·이식 절차는 [인수인계](../../../MLSYS_VER15_HANDOVER.md)와 [검증 보고서](../REPORT.md)를 확인한다.

`touch /data/chokwans99/duet-ver15-score-seeds-20261019/STOP`은 진행 중 seed를 마친 뒤 새 seed 배정을 중단한다. 즉시 중단 요청은 해당 raw의 process.json과 실제 cmdline을 확인하여 이 실험의 프로세스만 종료한다. 다른 사용자의 GPU 작업은 건드리지 않는다.
