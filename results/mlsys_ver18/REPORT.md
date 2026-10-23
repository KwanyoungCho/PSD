# MLSys ver1.8 검증·측정 자료

배포 내용과 실행 명령: [MLSYS_VER18_HANDOVER.md](../../MLSYS_VER18_HANDOVER.md).

- [완료된 14회 실험 보고서](evidence/FINAL_REPORT.md): ver1.7 + 초기화 phase-C 어댑터의 C/dtype 12회 및 동일 seed PIX/NODE 2회 대조.
- [결과 그림](evidence/RESULTS.pdf), [seed별 결과](evidence/SEED_REPORT.md), [서버 대조](evidence/PIX_NODE_REPORT.md).
- [검사 요약](checks/CHECKS.json), [runtime/tool hash](RUNTIME_MANIFEST.json).

Native ver1.8과 기존 어댑터를 같은 12 raw turns·seed42·FP16·출력 cap256으로 GPU에서
대조했다. **12/12턴 출력 ID와 927 step의 기록된 검증 결과가 일치**했고 native 실행기의
P1 C2 / P2 C3 및 기본 profile/probe OFF를 확인했다.
[동등성 기록](checks/native_equivalence/EQUIVALENCE.json).

검사 실행은 CPU 119개(104개 통과·CUDA 전용 15개 제외), 별도 CUDA 회귀 35개 모두 통과,
최종 측정/timeline 34개 모두 통과다. 동일 검사가 suite 사이에 중복되므로 합계를 독립
테스트 수로 해석하지 않는다. 과거 log-bin wire를 가정하던 테스트는 ver1.7 exact-bit
wire에 맞게 고쳤다. 검증 수식은 바꾸지 않았다.

명시적 timeline 모드는 별도 GPU 실행으로 확인했다. 12 raw turns·출력 cap128의 profile에서
375개 유효 진단 step, P1/P2/Miss 대표 PDF 3쪽을 생성하고 최신 렌더러로 다시 그렸다.
[Timeline PDF](checks/timeline_bundle/ALL_MODELS.pdf),
[HTML](checks/timeline_bundle/index.html), [수치](checks/timeline_bundle/PHASE_AVERAGES.json).
계측된 짧은 진단 표본의 TPS는 전체 비계측 성능 결과로 사용하지 않는다.

Git에는 compact 검사·portable timeline bundle·최종 요약 결과를 포함한다.
원시 run/profile은 이 작업 트리에 보존하고 `.gitignore`로 push에서 제외한다.
Portable bundle 재생성은 `SOURCES.json`과 `data/`를 사용하므로 원래 서버의 profile 경로가 필요 없다.
