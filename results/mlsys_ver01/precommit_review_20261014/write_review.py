"""Append the final precommit review without replacing earlier evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    j = json.loads((HERE / 'TP2_FULL_RESULTS.json').read_text())
    checks = json.loads((HERE / 'final_suites/CHECKS.json').read_text())
    final = json.loads((HERE / 'FINAL_DEFAULT_CHECK.json').read_text())
    a, b = (j['arms'][k]['summary'] for k in ('base', 'auto'))
    c = j['comparison']
    vc = [100*x for x in c['verify_change_ci95']]
    tc = [100*x for x in c['tps_change_ci95']]
    if any(x['returncode'] or x['errors'] or x['failures'] or x['skipped'] for x in checks):
        raise AssertionError('Final regression is not clean')
    section = f'''
<a id="precommit-review"></a>

## 커밋 전 독립 리뷰 재검토와 최종 회귀 — 2026-10-14

### 1. 기존 완료 표현의 정정과 오류 수정

리뷰의 핵심 지적은 맞았다. **234회 통과는 승인 extension의 당시 frozen runtime 결과이고, 후속 최적화까지 포함한 최종 runtime 전체 회귀 결과가 아니었다.** 이후25개 CUDA 검사 및 full-model 대조가 있었더라도, 이를 전체 회귀 완료로 표현할 수 없다. 앞선 “검증 끝, 배포만 남음”은 이 구분을 빠뜨렸다. 이번에는 최종 runtime으로 전체 review inventory를 재실행했다.

- 기존 `split=False` glue/전체 forward가 사용하지 않는 `duet_exit_replica`를 읽어 최소 fixture에서 실패하는 것을 재현했다. Fixture에 속성을 억지로 추가하지 않고 **split proxy 호출에서만 설정을 확인**하도록 고쳤으며, 공통 선택 helper도 `getattr(..., False)`를 쓴다. 이 guard 수정 자체는 실제 Config의 선택 정책을 바꾸지 않는다. 별도의 기본값 철회는 아래2절과 같다.
- 두 오류는 독립적인 두 운영 버그가 아니다. `test_tree_extension_review`가 기본 `ForwardAttention` 클래스를 직접 import해 unittest가 그 클래스를 중복 발견했다. FixedPlan 파생 검사 자체는 재현 로그에서 통과했고, 원래 forward 검사가 중복 오류를 냈다. 모듈 import로 바꿔 중복 발견을 없앴으며 package/직접 discover 양쪽 import를 지원한다.
- 추가로 `test_flat_tree_review.py`는 함수형 검사여서 unittest가0건으로 지나갈 수 있었다. 12개 law/isolation 조건을 unittest subtest로 연결했다. Test runner는 **0건·skip도 실패**로 처리하며 각 suite를 새 interpreter에서 실행한다.
- Replica/stream 기본값과 기존 env override를 검사하는4개 회귀를 추가했다. 설정 로그와 runtime이 같은 선택 helper를 사용한다.

최종 결과: **{len(checks)}개 suite, {sum(x['tests'] for x in checks)}개 unittest 실행, failure/error/skip 모두0**. 실행 전후 runtime·test source hash 일치. Compact production 검사 안의240조건, CPU forest12조건 등은 내부 조건 수이며 unittest 수에 중복 가산하지 않았다. 별도 package import 실행도1/1 통과했다. 이는 명시한 review inventory 전체이며 저장소의 모든 가능한 테스트·하드웨어 조합을 뜻하지 않는다.

`extensions/CHECKS.json`과 `CHECKS_PROVENANCE.json`에 최종 결과를 복사했다. 개별 로그와 원본은 `precommit_review_20261014/final_suites/`, 묶음은 `CHECK_LOGS.tar.gz`다. 이전234회 기록과 원시 archive는 역사적 증거로 유지한다.

### 2. TP2 자동 기본값: 전체 데이터 대조 후 자동ON 철회

검토 전 자동 기본값은 단순히 서버에 GPU가3장 있다는 뜻이 아니라 **요청한 config가 B>1 tree·async draft·target TP>1**이고 다른 proxy mode 및 quantized LM head와 충돌하지 않는 경우였다. 그러나 아래 full 비교에서 기본 활성화 근거를 확인하지 못해 **이 자동ON 변경을 철회했다. 최종 default는 원 논문과 같이 replica OFF**다. 기존 `SSD_DUET_EXIT_REPLICA=1`의 명시 실행은 검토/운영 설정용으로 유지하며 새 옵션은 만들지 않았다. 시작 로그에 batch, 실제 target TP, replica/stream 선택값과 `default`/`env` 출처를 출력한다. Stream의 기존 자동/명시 선택도 별도로 기록한다.

아래 A/B는 **동일한 오류 수정 후·기본값 철회 전 runtime**에서 replica OFF와 환경변수 미설정의 자동 선택만 비교했다. 후속 최적화 전체를 동시에 켜고 끈 대조가 아니다. 두 arm의 source hash는 `pre_policy_suites`의 전체248회 검사 runtime과 일치한다. 기본값 철회 후 최종 runtime은 별도로248회 전부 재검증하고 full model을 다시 실행했다. 두 source snapshot을 혼동하지 않는다.

- 모델: full LayerSkip-Llama3-8B + Qwama-0.5B-Instruct, BF16. 로컬4090 GPU4/5 target TP2 + GPU6 draft, 같은 NUMA의 CPU16–23/48–55.
- B8, T0.7, K1/K2=2/2, exit index17, P1/P2 generation=verification4/4, roots2, sibling C2, max_model_len2048, input cap512, output cap64.
- 동일480문항 전부×2 pass, warm48, target seed10143·draft seed231. Cold-prefix 및 cap/clip/부분 B/준비 step 제외 규칙은 두 arm에 동일 적용했다.
- Control: `exit_replica=0` explicit, stream 자동OFF. Candidate: 관련 두 env를 모두 제거하고 replica/stream 자동ON. Controller의 옛 강제OFF 환경까지 제거하는 `unset_env` 지원을 bench에 추가했다. 새 runtime 튜닝 옵션은 추가하지 않았다.

| 지표 | Replica OFF | 자동 replica+stream |
|---|---:|---:|
| Verification ms | {a['verify_ms']:.6f} | {b['verify_ms']:.6f} |
| Step ms | {a['step_ms']:.6f} | {b['step_ms']:.6f} |
| TPS, 보조 지표 | {a['tps']:.6f} | {b['tps']:.6f} |
| AL | {a['al']:.8f} | {b['al']:.8f} |
| Request cache hit | {100*a['hit']:.6f}% | {100*b['hit']:.6f}% |
| 포함 decode steps | {a['steps']} | {b['steps']} |

- Verify 변화 **{100*c['verify_change']:+.3f}%**, prompt-group bootstrap95% 구간 [{vc[0]:+.3f}, {vc[1]:+.3f}]%.
- TPS 변화 **{100*c['tps_change']:+.3f}%**, 같은 구간 [{tc[0]:+.3f}, {tc[1]:+.3f}]%. 이전 요청대로 시스템 최적화의 채택 근거는 직접 구간 비용이며 TPS는 보조다.
- 같은 출력 **{c['equal_output_sequences']}/{c['output_sequences']}**. 이전48문항 TP2 smoke와 구별해 전체 평가 결과를 추가했다.
- **이득 미확정 및 정책 보류:** Verify와TPS의 구간이 모두0을 포함한다. AL·출력도 달라졌으므로 이 online 평균을 순수 실행 비용 감소의 증거로 볼 수 없다. `REPLICA_DIVERGENCE.json`에는 warm group24의 부분B5 step에서 P1/P2 source 선택이 달라진 최초 event를 남겼다. 이후 난수 소비 경로도 달라질 수 있으므로897개의 독립 정확성 실패라는 뜻은 아니지만, 로그만으로 최초 원인을 head 수치 차이 또는 비동기 순서 중 하나로 확정하지 않았다. 원인 분리와 일관된 serving 이득을 확보하기 전 자동 기본값에서 제외한다.
- Decode 준비가 겹친 step: control {j['arms']['base']['preparation']['affected_steps']}, auto {j['arms']['auto']['preparation']['affected_steps']}. 두 arm 모두 prewarm한 대조이므로 이 표를 prewarm 자체의 가속 효과라고 해석하지 않는다.
- GPU telemetry·실제 선택값·raw hash는 `TP2_FULL_RESULTS.json`, 명령/설정은 `TP2_FULL_PLAN.json`, source와 원시 결과는 아래 archive에 있다. 외부 GPU process는 두 arm 모두 관측되지 않았다.
- 두 pass는 같은480문항의 반복이며 독립960문항이 아니다. 한 GPU 구성에서의 paired campaign이고 bootstrap은 환경/다른 seed/모든 TP 크기의 변동을 포함하지 않는다. 기존 TP1 B8 stream 저하 결과를 지우거나 TP2 결과로 대체하지 않는다.

**철회 후 최종 확인:** 환경변수를 지정하지 않은 최종 코드에서 replica/stream은OFF였고, 전체480×2 실행에서 기존 explicit OFF와 같은 출력은 **{final['equal_output_sequences']}/{final['output_sequences']}**였다. 전체 회귀 runtime hash와 이 실행의 hash가 일치한다. 이 실행은 기능 확인이며, 별도GPU7에서 회귀가 일부 병행됐으므로 새 성능 우위 수치로 사용하지 않는다. Decode 준비 영향 step은 {final['preparation']['affected_steps']}건이다. `FINAL_DEFAULT_CHECK.json`을 근거로 한다.

### 3. Commit 단위 분리 — 아직 commit/push하지 않음

| 순서 | 제안 commit 범위 | 재현 자료 |
|---|---|---|
| 1 | 승인한 새 모델·B>1 tree extension의 production 이식 | `APPROVED_EXTENSION.patch`: HEAD f12d0f1 → 승인 frozen snapshot |
| 2 | 입력 준비/확률 압축/prewarm, 명시 replica의 stream 연결 및 이번 proxy 경로 수정·시작 로그. Replica 자동ON은 제외 | `FOLLOWUP_RUNTIME.patch`: 승인 snapshot → 최종 production |
| 3 | 실험용 전체 packed target helper, bench adapter·실행/집계 도구, 회귀 테스트 | `REVIEW_RUNTIME.patch`와 `COMMIT_PLAN.json` 파일 목록 |
| 4 | 단계별 보고서·설정·원시 결과·source/hash archive | `COMMIT_PLAN.json`의 결과 자료 목록 |

별도 임시 디렉터리에1→2→3의 runtime patch를 적용해 **현재 runtime을 정확히 재현**하는 것을 확인했다. 같은 파일이1과2에서 바뀌므로 현재 파일을 통째로 `git add`하는 것으로 이 경계가 유지되지는 않는다. 실제 commit 시 patch/hunk 단위로 분리해야 한다. 이번에는 working index·branch history를 변경하지 않았다.

이전 `tree_followup_20261014/ADOPTED_RUNTIME.patch`는 그때 검증한 버전의 증거로 그대로 보존했다. 이번 수정까지 포함한 최종 이식 근거는 이 절의 patch 및 `FINAL_SOURCE.tar.gz`다.

자동ON A/B에 사용한 source는 `PRE_POLICY_SOURCE.tar.gz`에 별도로 보존했다. 최종 코드의 정책·로그·테스트 변경만 역적용해 복원하고 당시 기록한 runtime/test/adapter SHA256 전부와 일치함을 확인했다(`PRE_POLICY_SOURCE.json`). 원시 세 실행은 `RAW_TP2.tar.gz`/`RAW_MANIFEST.json`에 보존한다. 옛234회/25회/자동ON A/B/철회 후248회 검사에 같은 “최종” source 표기를 재사용하지 않는다.

### 4. 서로 다른 캠페인의 TPS 해석 정정

리뷰의 “서로 다른 날”은 raw timestamp와 다르다. 한국시간2026-10-14에 `full_l2/flat_prob`는02:53:21, `candidate_full_l2/candidate`는03:28:19에 시작했다. **같은 날이지만 GPU UUID 쌍이 다르다.** 전자는701c7c85/947d0308 쌍, 후자는00cbad2d/97d3e0bb 쌍이다.

따라서615.36과610.21 TPS 차이(약0.84%)를 순수 run-to-run noise의 크기나 통합 최적화 손실로 확정하지 않는다. 각 캠페인 내부의 같은 GPU pair before/after를 비교한다. 기존 통합 Llama2 verify−2.89%와 그 구간의CI는 해당 matched A/B 근거로 유지하되, 이를 모든 모델/설정의 TPS 상승으로 일반화하지 않는다.

### 5. 검토 후 상태

전체 회귀 미실행·fixture 오류·0건 검사 누락·선택값 로그 부재를 보완했다. 위 전체 TP2 실행도 완료했고, 원격에만 있던 최종 검사 파일 문제는 이 작업 트리에 결과와 로그를 포함해 해결했다. **Commit·push는 이번 검토에서 수행하지 않았다.** 원 논문70B/G>M 영향, 다른 서버의 연구 수식 통합, 최종 SSD 재튜닝 비교 등 앞의 별도 후속 항목은 이 검토로 완료 처리하지 않는다.
'''
    path = HERE.parent / 'REPORT.md'
    s = path.read_text()
    marker = '<a id="precommit-review"></a>'
    if marker in s:
        s = s[:s.index(marker)].rstrip()
    s += '\n\n' + section.strip() + '\n'
    top = '**최신 상태는 [배포·현재 반영·남은 항목](#current-port-status)을 기준으로 한다.**'
    intro = '**최신 검증은 [커밋 전 독립 리뷰 재검토](#precommit-review)를 함께 확인한다.**'
    if intro not in s:
        s = s.replace(top, intro + '\n\n' + top)
    path.write_text(s)


if __name__ == '__main__':
    main()
