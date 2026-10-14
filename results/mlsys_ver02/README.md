# MLSys-ver0.2 현재 상태

2026-10-14 기준.

- 최초 검증 완료 release `0d53742`는 `origin/MLSys-ver0.2`에 push 완료했다.
- 이후 검토한 작은 추가 최적화의 코드 commit은 `99dec94`다. B>1 tree에서 중간 proposal padding tensor 복사를 제거했다. 연구 수식·tree 선정 정책은 그대로다.
- [최신 검토·적용 결과](remaining_review_20261014/REPORT.md): 25 suite/265개 검사, 두 모델 각각 full480문항×2회 A/B. 출력/AL/hit/step event 동일. 해당 준비 비용은 감소했지만 전체 TPS 향상은 주장하지 않는다.
- [최초 ver0.2 반영 목록](REPORT.md): 새 모델·B>1 실행기·확률 연산 압축·입력 준비·prewarm·집계/리뷰 수정 등 기존 전체 인수인계. 이 파일과 `MANIFEST.json`은 최초 release 당시 자료로 보존한다. 그 안의 “push하지 않았다”는 당시 상태다.
- 최신 문서/증거 commit과 원격 head는 `git log -1 origin/MLSys-ver0.2`로 확인한다. 다른 서버에서는 snapshot repository의 `fa76960`이 아니라 **`origin/MLSys-ver0.2`**를 검토/병합한다.

Full target packed tree, mixed-miss AR, replica/stream 자동ON은 계속 기본 미채택이다. 다른 서버에서 진행 중인 root/위치/tree 점수와 miss DS 알고리즘은 합치지 않았다. 최종 알고리즘·하드웨어의 parameter 재선정/70B 평가/SSD 최종 비교는 별도 단계다.
