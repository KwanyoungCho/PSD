# 실패·범위 제한 기록

- 초기 `smoke_base.json`, `smoke_tree.json`: 원 논문 AMD draft 초기화가
  float32로 돌아가 FlashAttention dtype 오류로 worker가 종료됐다. 원 코드의
  fail-fast `os._exit` 때문에 JSON 상태는 running에 남았지만 실행 중인 것이
  아니다. Log와 종료한 process를 확인했다. Draft config의
  `dataclasses.replace` 이후 target runtime dtype을 보존하는 공통 adapter로
  해결했고 `_r1` 재실행은 완료했다. 이 호환 수정의 효과를 verification
  speedup에 포함하지 않는다.
- `tree_parity.json`/`prototypes/tree_v1.py`: 최초 작은 tree graph 이식안은
  고정 입력에서 scalar walk보다 느린 조건도 많았다. 작은 metadata의 여러
  H2D 복사를 하나로 합친 V2를 별도로 측정했다. V1은 삭제하거나 성공 사례만
  골라내지 않았다. V2도 full-model 지표로 채택 여부를 판단한다.
- `kv_correctness.json`: 원 논문 코드의 네 가지 실패는 의도적으로 재현한
  정확성 결함이다. 동일 검사가 좁은 KV 수정에서 통과한다. 잘못된 prefix
  출력이나 빈 query의 빠른 실행을 성능 기준으로 인정하지 않는다.
- 두 번째 전체 worktree checkout은 역사적 result 파일 복사 중 디스크
  여유 부족으로 실패했다. SSD 추론 소스만 sparse checkout하여 재구성했고
  모든 patch가 원 commit에 clean apply되는 것을 검사했다. 원본 연구/결과를
  지우지 않았고 실행 중 inference source의 내용은 바꾸지 않았다.

추가 실패가 있으면 이 파일과 raw log를 함께 갱신한다. 준비 단계 실패,
old-code defect 재현, 성능 회귀를 서로 구분한다.

- 2026-10-12 09:11–09:14 UTC경 root filesystem ENOSPC로 네 full 반복의
  logging/telemetry가 중단됐다. 완료된 R0 대조군·후보 결과는 유지했다.
  R1 candidate의 부분 실행(209/177/177/145 입력 저장)은 모두
  `failed_attempts/disk_full`로 격리하고 성능 판단에서 제외했다.
  `DISK_INTERRUPTION.json`에 기록했으며 같은 설정·seed로 fresh process에서
  R1을 처음부터 다시 실행해 완료했다. 고장 후 남은 **이번 실험의** GPU process만
  종료했으며 다른 작업은 종료하지 않았다.
- 이미 Git에 보존된 Round6 gzip archive와 SHA256이 일치하는 로컬 비압축
  사본124개(1817.91MiB)만 정리했다. 원본 byte는 archive에서 복원 가능하며
  `ROUND6_RESTORABLE_COPY_CLEANUP.json`에 파일별 증거를 남겼다. Round6의
  `archive_results.py --restore`로 복원한다. Round7의 실행 결과 디렉터리는
  `/data/chokwans99/duet_round7_runtime`으로 옮겼고 최종 인수인계 시 상대경로
  gzip archive/manifest로 함께 보존한다. 원 논문/source byte는 그대로다.

- `tree_noncompact_guard.log`는 허용 범위를 벗어나는 parent-q 참조를 넣은
  점검이다. 원 wire validation의 `q_ref < u_valid <= valid` 계약에 의해
  거부됐으며 실제 허용된 tree의 결함 재현이 아니다. 이를 근거로 추가하려던
  guard는 반영하지 않았고, 측정·portable patch 모두 V2 그대로 유지했다.
