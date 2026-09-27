# 2026-09-27 인계 검증 기록

이번 검증은 문서/파일/기존 CPU 검사의 재현에 한정한다. 새 GPU inference,
dense full-model 실험, 실패했던 6정책 online campaign을 완료한 것이 아니다.

## 역사적 결과 보존

[`frozen_integrity.json`](frozen_integrity.json)의 모든 checksum이 일치했다.

| 기존 manifest | 확인 파일 수 | 결과 |
|---|---:|---|
| `duet_tree_al_full/final_manifest.json` | 62 | 전부 일치 |
| `duet_tree_posthoc/final_manifest.json` | 1,273 | 전부 일치 |
| `duet_tree_followup/execution_manifest.json`의 source | 158 | 전부 일치 |
| `duet_tree_followup/offline_review.json`의 결과 | 10 | 전부 일치 |

마지막 두 manifest의 source/result만 검사했다. 외부 model weight의 내용 전체를
새로 hash하거나 새 서버에 복사한 것은 아니다. 기존 `REPORT.md`/코드를 덮어쓰지
않고 상태 정정은 `STATUS_20260927.md`와 루트 `HANDOVER.md`에 기록했다.

## CPU 재실행

실행 명령:

```bash
ssd/.venv/bin/python results/handover_20260927/validate_handoff.py \
  --output results/handover_20260927/cpu_validation
```

새로 재실행할 때는 다른 `--output` 경로를 사용한다. 도구는 기존 디렉터리를
덮어쓰지 않는다. 원래 `math_checks.json` 등을 보존하고 결과 destination만 바꿨다.

- Python source 185개 AST 문법 검사 통과. 기존 `make_figs.py`의 `\;` 문자열에
  Python SyntaxWarning이 있지만 parse failure는 아니며 역사적 source는 유지했다.
- Entropy 알려진 분포/variable-K/hazard/RNG 보존/마지막 chunk flush 검사 통과.
- C=3 484cases, 최대 오차 1.30673e−13.
- 사전 fanout 배분 vs 완전탐색 600cases, 최대 오차 2.22045e−16.
- 2-token 결합분포 100cases, 최대 오차 2.22045e−16.
- 잘못된 사후선택 negative control의 TV=.01875 확인.
- 실제 production selector 1,000cases, mixed-depth mask 3cases 통과.

상세: [`cpu_validation/summary.json`](cpu_validation/summary.json),
[`math_checks.json`](cpu_validation/math_checks.json),
[`frontier_checks.json`](cpu_validation/frontier_checks.json) 및 같은 폴더 로그.

## 패키지와 Git 대상

- Git 대상 연구 파일1,056개를 manifest와 대조해 누락/내용 불일치 없음.
- Archive 대상2,362개는 Git staging에 포함되지 않았음.
- 원시 tar 내부의 **모든 파일 내용**을 스트리밍으로 hash하고 원본 manifest와
  대조해 일치 확인: [`archive_contents_validation.json`](archive_contents_validation.json).
- Tar 자체 SHA256:
  `e22f04404c488d9187e2dc387069f0237d432f9b4bdf9945ada1c2682f0b4438`.
- Staging 점검 시 10MiB를 넘는 파일 없음. Private-key 및 흔한 credential token
  패턴 점검에서 발견 없음. 이 검사는 모델 출력의 의미 분석을 대신하지 않는다.
- 새 인계 문서의 상대 링크를 확인했다. Commit 전 최종 링크/staging 점검 결과는
  [`publication_checks.json`](publication_checks.json)에 기록한다.

기존 로그의 trailing space, CSV의 CRLF, followup REPORT의 마지막 빈 줄 때문에
전체 `git diff --cached --check`는 whitespace 경고를 냈다. 원본 checksum 보존을
위해 정규화하지 않았다. Code/new docs 검사는 다음 명령으로 통과했다.

```bash
git diff --cached --check -- '*.py' '*.sh' '*.md' .gitignore \
  ':!results/duet_tree_followup/REPORT.md'
```

Commit/push 대상은 `feat/duet-proxy-source-ablation`이다. `main`에 merge하거나
원격 history를 강제로 덮어쓰지 않는다. 배포 commit은 새 서버에서 `git rev-parse
HEAD`로 확인하며, 원시 tar는 [ARTIFACTS](ARTIFACTS.md)의 절차로 별도 전송한다.
