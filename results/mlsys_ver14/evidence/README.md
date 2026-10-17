# 과거 실험 증거 보존

이 폴더의 보고서·해시·측정 버전은 당시 상태를 보존했다. Ver1.4 최종 코드의 실측으로 이름을 바꾸지 않았다.

- `response_async_20261017`: ver1.3 이후 topology prepack 채택 근거. 미채택 async/early/host-list 실험도 증거로 포함한다. Runtime에는 이들의 미채택 코드가 없다.
- `local_followup_20261017`: 작은 prep 최적화 및 미채택 bucket/q-overlap의 검토 기록. Unpacked raw를 중복 저장하지 않고 원래 `RAW_RESULTS.tar.xz`로 보존한다.
- `local_fused_response_20261017`: 통합 응답/metadata/glue 변경의 16회 실험 및 48문항/56턴 비교. 당시 base는 ver1.3 자체가 아닌 prepack 기준 `1dcb10ba`다.

Ver1.3과 최종 ver1.4의 직접 비교는 상위 `REPORT.md`, `DIRECT_COMPARISON.json`, `release_raw.tar.xz`를 확인한다. 과거 보고서의 'push하지 않음'은 당시 실험 상태다. 현재 배포 상태는 루트 `MLSYS_VER14_HANDOVER.md`가 정본이다.

과거 patch의 context 공백, raw 로그의 끝 공백, CSV의 원래 CRLF도 해시 보존을 위해 변경하지 않았다. Runtime과 새 문서의 whitespace 검사는 이 과거 증거 폴더를 제외하고 수행한다.
