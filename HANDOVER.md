# DUET 연구 브랜치 인수인계 — 2026-09-27

**2026-10-09 전체 작업 이관 보완:** 이 branch는 논문 기준 `a82f7d2` 이후의 root·위치 수식, calibration, tree AL·사후 분석 연구를 보존한다. 후속 시스템 구현과 Round1–5 실험은 별도 **`feat/duet-mlsys-coverage`** branch에 있다. 두 branch는 공통 논문 기준에서 갈라졌으므로 어느 하나만 받으면 전체 작업이 아니다.

- 전체 이관의 시작 문서: systems branch의 `results/mlsys_coverage/MERGE_REVIEW.md` **23절**. 다른 서버에 전달할 branch, 코드 겹침, 원시 데이터 복원, 검토 순서를 정리한다.
- 이번에 추가 보존한 자료: [10/01 root 보고서](results/residial_dist/root_progress_20261001/REPORT.md), [증거·증명](results/residial_dist/root_progress_20261001/EVIDENCE.md), [거리별 분석](results/residial_dist/root_progress_20261001/DISTANCE.md), 그림 6쌍 및 재생성 스크립트·집계표. 기존 결과의 후속 분석이며 새 GPU 실험이 아니다.
- 아래의 9/27 완료/미완료는 **당시 AWQ70B 연구 조건**이다. 후속 dense7B/8B·B>1·greedy·root/tree 통합 결과는 systems branch의 최신 Round5 보고서에서 확인한다. 다른 조건의 후속 실험을 기존 AWQ70B 미실행 job의 완료로 대체하지 않는다.
- 원시 데이터 약23.48GB의 `handoff_artifacts/duet_research_raw_20260927.tar`는 여전히 **Git 외 별도 전송** 대상이다. 복원·checksum은 [ARTIFACTS](results/handover_20260927/ARTIFACTS.md)를 따른다. 새 root 보고서의 작은 산출물은 Git에 직접 포함하여 frozen 9/27 raw archive를 변경하지 않았다.
- 이전 담당자가 서버 점검을 위해 GPU 실험을 중단했다. Branch를 checkout하거나 문서를 읽는 것만으로 queue/실험을 자동 재개하지 않는다.

이 문서는 `feat/duet-proxy-source-ablation`에서 진행한 9월 연구를 실제 실험 서버로
이관하기 위한 시작점이다. 기반 commit은
`a82f7d24fb36827a9a81a3567f344dccb71f193e`이다. 그 이후의 코드 변경, 연구 스크립트,
실험 계획, 원시 결과를 이용한 분석, 실패 기록을 이번 인계 commit으로 보존한다.
이전 보고서의 “미커밋”, “GPU 대기 중” 같은 표현은 당시 상태다.
**실행 상태는 이 문서와 [최신 상태](results/duet_tree_followup/STATUS_20260927.md)를 우선한다.**

## 먼저 알아야 할 결론

1. **Root 후보 선정의 목적은 cache hit, tree 생성의 목적은 hit 이후 accepted
   descendant length(AL)**이다. Tree 채택의 주지표를 TPS로 바꾸지 않는다.
2. Early-exit proxy는 항상 평평하지 않다. 기존 residual `[e-q]+`는 현재 주요
   조건에서 proxy-only `e`보다 coverage가 낮았다. 후보 배분과 정규화를 개선하면
   기존 proxy 정책보다 좋아졌지만, `e(1-q)`류 토큰 점수만의 추가 우위는 미확정이다.
3. 새 calibration은 제한된 chain 설정에서 탐색 축소 가능성을 보였다. 독립 속도
   확인에서 추천 설정은 기존 깊은 설정 대비 +42.19%였지만, 이는 후보 수식의 성과가
   아니며 다른 서버/정밀도/tree로 그대로 일반화할 수 없다.
4. Tree의 q 경로 곱을 보정 reach로 바꾼 **기존 두 정책의 full dataset 실험은 완료**했다.
   480문항/560턴 × 2정책 × 3seed에서 AL 2.0651→2.1003(+1.70%), 차이의
   95% CI [−0.00249,+0.07206]로 전체 우위는 미확정이다.
5. 사후 분석에서 fanout을 새 자식을 뽑기 전에 배분하는 개선안을 도출했다.
   C=3 CPU 수학 검사와 작은 모델의 GPU 실행기 검사는 완료했다. 국소 expected gain
   +1.66%는 **새 정책의 실제 online AL 개선 수치가 아니다.**
6. **새 여섯 정책의 full dataset 비교는 미실행**이다. 9/23 첫 70B smoke가 AWQ 모델
   로딩 OOM으로 실패했다. 다른 프로세스가 GPU 메모리를 차지한 상태였고, 그 뒤
   대기열이 중단됐다. 18개 online job과 별도 두 trace job은 시작되지 않았다.
7. **G>M의 사후 점수 기반 pruning에는 분포 편향 반례(TV=0.01875)가 있다.**
   신규 tree 실험은 G=M을 유지한다. 기존의 “closure를 지키면 lossless”라는 설명은
   충분하지 않았다. 여섯 정책의 full-model 정확성 검증도 아직 남아 있다.
8. 이 브랜치의 수치 실험은 **AWQ target 70B + BF16 TinyLlama** 기준이다.
   Dense BF16/FP16 target의 결과 또는 Mirror-SD 전체 시스템을 이긴 결과로 쓰지 않는다.

## 문서 읽는 순서

| 문서 | 포함 내용 |
|---|---|
| [실험 이력과 결론](results/handover_20260927/EXPERIMENTS.md) | 수행한 실험, 표본/분할, 수치, 정정, 주장 가능 범위, 원본 보고서 |
| [코드와 구현 지도](results/handover_20260927/CODE_MAP.md) | Production 변경, 연구 hook, CLI/env 의미, 데이터 흐름, 이식 시 수정 지점 |
| [Full-model 서버 검증 절차](results/handover_20260927/FULL_MODEL_VALIDATION.md) | 환경 이식→정확성→별도 calibration→full dataset→사후 진단 순서와 통과 기준 |
| [파일·데이터 이관](results/handover_20260927/ARTIFACTS.md) | Git 포함 범위, 원시 데이터 archive, SHA256 검증, 외부 모델/환경 의존성 |
| [이전 진행 기록](results/DUET_PROGRESS.md) | 이전 설명 및 연구 결정의 상세 이력 |
| [사용자 설명 TODO](results/residial_dist/training_free/TODO.md) | 사용자가 아직 확인하지 못한 수식 설명; 완료 처리하지 않음 |

“Full model”은 우선 **AWQ 대신 원본 70B를 BF16/FP16으로 실행**한다는 의미로
인계한다. 새 GPU 종류/개수, 실제 checkpoint, dtype는 아직 지정되지 않았다.
“Full dataset”은 별도 개념이며, 기존 완료 실험은 full 480문항이지만 출력 상한은
128이었다. 논문 조건의 1,024-token/chat 평가까지 재현한 것은 아니다.

## 다음 담당자가 우선 할 일

- Git branch를 받은 뒤 원시 데이터 archive도 별도로 가져와 checksum을 확인한다.
- 기존 `results/duet_tree_al_full`, `duet_tree_posthoc` 및 실패 attempt를 수정하지 않고,
  새 campaign 디렉터리/설정/manifest를 만든다.
- 모델·GPU를 확정하고, 스케줄러로 GPU를 확보한 후 dense baseline부터 로딩한다.
  기존 `launch.py --wait`나 `ssd/env.sh`를 그대로 실행하지 않는다.
- G=M, B=1, root 정책 고정 상태에서 실제 full-model smoke와 graph/eager parity를
  먼저 확인한다. 기존 작은 모델 검사 통과를 대신 쓰지 않는다.
- 새 precision의 disjoint calibration을 고정하고 여섯 tree 정책의 480문항 × 3seed
  AL 비교를 완료한다. 후보 정책/exit/budget 변경은 별도 실험으로 분리한다.
- 추가 frontier trace와 동일 engine/KV 기반 causal replay로 실패 원인을 확인한다.
  기존 draft-only shadow는 q 재현 gate 실패로 효과 판단에서 제외한다.

이번 인계는 새 GPU 실험을 수행한 것이 아니다. 기록/패키징 검사는
[검증 기록](results/handover_20260927/VALIDATION.md)에 남긴다. 저장소 push와
외부 원시 데이터의 새 서버 전송은 별도 작업이다. 원시 데이터는 현재 서버에
보존하며, 새 서버 주소가 없으므로 전송 명령과 archive를 제공한다.
