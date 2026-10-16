"""Finish the review only after both counterbalanced campaigns validate."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
comparisons=json.loads((P/'STEP_COMPARISONS.json').read_text())
assert comparisons['status']=='complete'
profiles=json.loads((P/'PROFILE_COMPONENTS.json').read_text());assert profiles['status']=='complete'
readme=(P/'REPORT.md').read_text()
handover=readme[readme.index('## 다른 작업자가 확인할 순서'):] if '## 다른 작업자가 확인할 순서' in readme else ''
readme=readme.replace('상태: **실험 진행 중. 최종 채택 판단 전.**','상태: **모든 예약 비교 실행과 분석 완료. 원래 ver1.2에는 미반영.**')
readme=readme[:readme.index('## KV 복원 순서 단독 변경의 전체 step 결과')] if '## KV 복원 순서 단독 변경의 전체 step 결과' in readme else readme[:readme.index('## 검증과 결과')]
rows=comparisons['comparisons'];bytag={r['tag']:r for r in profiles['runs']}
final=[r for r in rows if r['stage']=='final' and r['cohort'].startswith('2x')]
checks=json.loads((P/'raw/checks_restore_only/CHECKS.json').read_text())
assert all(not r['failures'] and not r['errors'] and not r['skipped'] for r in checks)
readme+='''## KV 복원 순서 단독 변경의 전체 step 결과

**아래 표가 성능 판정의 주 근거다. Profile OFF, 같은 GPU, ABBA 순서, 동일 출력·수락 기록의 step끼리 비교했다.** 전체 step에는 target/draft 응답 대기, 수락, 스케줄링 비용이 모두 들어 있다.

| Target +68M | Step 전→후(ms) | 감소 | 감소율95% CI | 반복 a / b 감소율 | 동일 step 수 |
|---|---:|---:|---:|---:|---:|
'''
for r in final:
 ci=r['reduction_pct_ci95'];reps=r['repeats']
 readme+=f"| Llama2-{r['model'].upper()} | {r['before_ms']:.3f} → {r['after_ms']:.3f} | {r['reduction_pct']:+.2f}% | [{ci[0]:+.2f}, {ci[1]:+.2f}]% | {reps[0]['reduction_pct']:+.2f}% / {reps[1]['reduction_pct']:+.2f}% | {r['matched_steps']} |\n"
readme+='\n| Target | Target 호출·수락 구간 전→후(ms) | 그 밖의 step 구간 전→후(ms) |\n|---|---:|---:|\n'
for r in final:
 readme+=f"| {r['model'].upper()} | {r['before_verify_ms']:.3f} → {r['after_verify_ms']:.3f} | {r['before_outside_verify_ms']:.3f} → {r['after_outside_verify_ms']:.3f} |\n"
readme+='''
신뢰구간은 **12 UID cluster bootstrap**이다. Process가 반복될 때의 환경 변동까지 모두 포괄하는 신뢰구간은 아니다. 따라서 반복별 결과와 warmed profile의 변경 구간도 함께 판단한다. **13B는 반복 b가 약0.047%로 거의 중립이므로, 큰 전체 step 향상이 확정됐다고 표현하지 않는다.** 응답·verification 바깥 구간 감소와 전체 평균의 작은 이득을 구분한다.

| Target | 현재 step source | 동일 n | 전체 step 전→후(ms) | 감소 |
|---|---|---:|---:|---:|
'''
for r in final:
 for source,label in [('1','P1 hit'),('2','P2 hit'),('0','Miss')]:
  v=r['source'][source];readme+=f"| {r['model'].upper()} | {label} | {v['n']} | {v['before_ms']:.3f} → {v['after_ms']:.3f} | {v['reduction_pct']:+.2f}% |\n"
readme+='\n엔진 내부 timer와 별도로 원래 `LLMEngine.step()` 호출 전체를 감싼 벽시계도 같은 감소율을 확인했다. [WALL_STEP_CHECK.json](WALL_STEP_CHECK.json).\n'
readme+='''
## 변경 구간을 직접 본 warmed profile

전체 step과 별도로, 초기10 step 및 cap/clip을 동일하게 제외한 profile-ON 표본이다. 위의 비계측 성능과 섞어 평균내지 않는다.

| Target | 구간 | 전→후(ms) | 감소(ms) |
|---|---|---:|---:|
'''
for model in ('7b','13b'):
 a=next(r for r in bytag[f'before_{model}_profile_warm']['summary'] if r['state']=='all')
 b=next(r for r in bytag[f'restore_{model}_profile_warm']['summary'] if r['state']=='all')
 for key,label in [('response_ms','요청 전송 후 응답 완료'),('whole_gap_ms','최종 logits 종료→다음 target forward'),('response_to_forward_ms','응답 완료→다음 target forward'),('graph_pre_ms','Target exit 이전 forward'),('graph_post_ms','Target exit 이후 forward')]:
  readme+=f"| {model.upper()} | {label} | {a[key]:.3f} → {b[key]:.3f} | {a[key]-b[key]:+.3f} |\n"
readme+='''
`response_ms`는 요청 send 종료부터 응답 대기 종료까지의 관측된 경과 시간이다. CPU 준비·NCCL·경우에 따라 JIT를 포함하므로 순수 통신 latency라고 부르지 않는다. KV 복원은 target에서 수행하는 일이 아니라 **draft의 요청 서비스**다. Hit는 응답을 먼저 보내고 복원하므로, target forward와 겹칠 수 있다. Miss는 복원 이후에 JIT하므로 lossless 동작에 필요한 prefix를 항상 먼저 확보한다. Cache 판단 뒤에 metadata를 읽게 되는 실행 순서 변화도 있으므로, 모든 source의 감소분을 KV copy 하나의 순수 비용이라고 해석하지 않는다.

![전체 step과 target forward 사이 공백](LATENCY_COMPARISON.png)

## 정보 재사용·입력 준비의 별도 비교

독립적으로 측정한 앞선 캠페인이다. 아래 변화와 KV 순서 변경의 효과를 더하지 않는다. **결합 코드는 마지막에 모델별12입력1회로 실행·정확성을 확인했다. 결합 코드의 반복 A/B 성능 검증은 하지 않았으므로 개별 효과를 합산하지 않는다.**

| Target | 비교 | 감소율 | UID95% CI | 반복 a / b |
|---|---|---:|---:|---:|
'''
for r in rows:
 if r['stage']!='perf' or not r['cohort'].startswith('2x'):continue
 ci=r['reduction_pct_ci95'];reps=r['repeats']
 readme+=f"| {r['model'].upper()} | {r['reference']} → {r['variant']} | {r['reduction_pct']:+.2f}% | [{ci[0]:+.2f}, {ci[1]:+.2f}]% | {reps[0]['reduction_pct']:+.2f}% / {reps[1]['reduction_pct']:+.2f}% |\n"
readme+='''
`snapshot`은 추가 동기화를 줄이는 방향이지만 반복별 효과를 확인해야 한다. `inputs`는 GPU 입력·geometry 버퍼 재사용까지 포함한 조합이다. Incremental 비교(snapshot→inputs)와 코드 변경량도 함께 봐야 한다. 이 결과만으로 임의의 모델/B/길이에서 개선을 보장하지 않는다. 구현은 해당 실험 커밋과 서버 variant에 보존했다.

## 2배 step 제외 내역과 raw 민감도

| Target | 반복 | before 제외 | restore 제외 | Raw 감소율 | Filtered 감소율 |
|---|---|---:|---:|---:|---:|
'''
for r in final:
 raw=next(v for v in rows if v['model']==r['model'] and v['stage']=='final' and v['cohort'].startswith('raw'))
 for i,rep in enumerate(r['repeats']):
  ex=r['excluded_long_steps'][i]
  readme+=f"| {r['model'].upper()} | {rep['repeat']} | {ex['before']} | {ex['after']} | {raw['repeats'][i]['reduction_pct']:+.2f}% | {rep['reduction_pct']:+.2f}% |\n"
readme+='''
2배 기준은 같은 request-group·batch·source 구성의 median을 사용한다. Step의 시간과 토큰을 함께 제외하고 raw를 보존했다. 각 arm에서 제외된 step이 다를 수 있으므로 filtered 결과는 **양쪽에 남은 step 교집합**이다. Raw 출력·수락 기록은 정확히 일치하지만, 각 arm의 native filtered AL/hit 평균은 제외 집합 차이 때문에 소폭 달라질 수 있다. 모든 arm의 제외 개수는 [EXCLUSIONS.json](EXCLUSIONS.json)에 있다. 준비 시간을 retained step에서 임의로 차감하지 않았다.

## 정확성·환경·브랜치 보호

'''
readme+=f"- KV 순서 변경 최종 runtime: {sum(r['tests'] for r in checks)}개 회귀 검사, failure/error/skip 모두0. [CHECKS.json](raw/checks_restore_only/CHECKS.json). 단편화 page, 이전 state 보존, seq 교체, EOS/output 절단을 검사했다.\n"
for model in ('7b','13b'):
 for stage in ('profile_warm','final_a','final_b'):
  folder=P/'raw/runs'/f'restore_{model}_{stage}'
  q=json.loads((folder/'PARITY.json').read_text());assert q['output_equal'] and q['events_equal']
  readme+=f"- `{folder.name}`: {q['outputs']}개 출력·{q['events']}개 수락 event 모두 기준과 동일.\n"
combined_checks=json.loads((P/'raw/checks_combined/CHECKS.json').read_text())
assert all(not q['errors'] and not q['failures'] and not q['skipped'] for q in combined_checks)
readme+=f"- 최종 결합 코드: GPU 기능 검사 {sum(q['tests'] for q in combined_checks)}개 모두 통과. [combined CHECKS](raw/checks_combined/CHECKS.json).\n"
for model in ('7b','13b'):
 q=json.loads((P/'raw/runs'/f'combined_{model}_integration'/'PARITY.json').read_text())
 assert q['output_equal'] and q['events_equal']
 readme+=f"- 최종 결합 {model.upper()}: {q['outputs']}개 출력·{q['events']}개 event 정확히 일치. 단일 process/profile OFF 평균 step {q['summary']['step_ms']:.3f}ms는 실행 확인용이며 반복 A/B 효과 크기로 인용하지 않는다.\n"
readme+="- Cache routing CPU13개는 동일 fixture로 기존/변경 runtime 각각 통과했다. 오래된 fixture의 `tree_cache_is_tree` 누락을 테스트만 보완했다. 최초에는 baseline에서도12건이 실패했고, fixture 보완 후 양쪽13건 모두 통과했다. 중간 wrapper의 import 순서 오류 로그도 별도 보존했다. Serving 수정으로 테스트를 우회하지 않았다. [BEFORE](raw/CACHE_CONTRACTS_BEFORE.json), [AFTER](raw/CACHE_CONTRACTS_AFTER.json).\n"
readme+='''- 모든 완료 run은 source hash가 실행 전후 동일해야 한다. GPU 초기 점유·동시 PID·clock·온도를 [TELEMETRY.json](TELEMETRY.json)에 보존했다. 동시 GPU residency 검사는 모든 호스트/하드웨어 잡음을 제거했다는 뜻은 아니다.
- 원래 브랜치·커밋·91개 runtime 파일·기존 git status는 [ORIGINAL_UNCHANGED.json](ORIGINAL_UNCHANGED.json)에서 확인한다. 기존 uncommitted 작업을 지우거나 덮어쓰지 않았다.
- 초기 cold-start 진단을 위해 중단한4개 실행은 `aborted`이며 성능/정확성 완료 수에 넣지 않았다. 컴파일/JIT 준비 차이를 inference 개선으로 주장하지 않는다.

## 최종 판단과 적용 범위

'''
verdicts=[]
for r in final:
 ok=r['reduction_pct_ci95'][0]>0 and all(v['reduction_pct']>0 for v in r['repeats'])
 verdicts.append(dict(model=r['model'],recommend_for_review=ok,whole_step_strength='consistent' if r['model']=='7b' else 'small; repeat b almost neutral',reason='both process repetitions improve and UID CI excludes zero' if ok else 'not a consistent positive whole-step result'))
 readme+=f"- {r['model'].upper()}: {'이번 B1 tree 설정에서 KV 복원 순서 단독 변경을 적용 후보로 추천한다.' if ok else '일관된 전체 step 개선이 확인되지 않아 채택을 보류한다.'}\n"
readme+='''
**기존 ver1.2에 자동 병합하지 않았다.** 독립 패치는 [KV_RESTORE_ORDER.patch](KV_RESTORE_ORDER.patch)이며 runtime 변경은 `ssd/ssd/engine/draft_runner.py` 한 파일이다. 현재 보호 대상에 `git apply --check`만 수행해 적용 가능함을 확인했다. 실험 브랜치 HEAD는 결합 코드이며 [COMBINED_RUNTIME.patch](COMBINED_RUNTIME.patch)에 해당한다. 개별 채택 판단과 결합 실행 검사를 구분해서 검토한다. 다른 서버에서 검토할 때는 원래 baseline의 pending 집계 변경까지 포함한 `BASELINE_WORKING_CHANGES.patch`와 혼동하지 않는다.

검증 범위는 RTX3090·B1·현재 두 모델·K3 tree·고정12입력이다. **B>1, T=0 전체 모델 실행, 13B chain, 다른 서버로의 성능 일반화는 검증하지 않았다.** 7B chain은 별도 K10/40 roots 조건에서 before↔inputs만 검증했다. Chain에는 tree KV 복원 구간 자체가 없으므로 이 단독 변경으로 chain response wait까지 줄였다고 주장하지 않는다. 다음 검토는 필요할 때 input 재사용과 KV 순서 변경의 결합 효과 또는 chain의 별도 큰 구간을 같은 방식으로 비교하는 것이다.

## 재검토·재생성

- [STEP_COMPARISONS.json](STEP_COMPARISONS.json): profile OFF, raw/filtered, source별 latency, 반복·UID bootstrap.
- [PROFILE_COMPONENTS.json](PROFILE_COMPONENTS.json): warmed before/after 및 초기 ablation traces. 초기 cold 7B baseline은 최종 개선 판단에 사용하지 않는다.
- [raw/VARIANTS.json](raw/VARIANTS.json): 서버 variant별 모든 runtime SHA256과 baseline 대비 변경 파일.
- `raw/runs/`: 모든 완료 run의 JSON·native profile·PROVENANCE·telemetry·parity. 서버 원본은 `/data2/chokwans99/target-gap-20261017/`.
- `analyze_steps.py`, `analyze_profiles.py`, `audit_telemetry.py`, `plot_results.py`: 저장된 결과만으로 재분석. 모델/GPU가 필요 없다.

```bash
python analyze_steps.py --raw raw --output STEP_COMPARISONS.json
python analyze_profiles.py --raw raw --output PROFILE_COMPONENTS.json
python audit_telemetry.py --raw raw --output TELEMETRY.json
python plot_results.py
```
'''
(P/'REPORT.md').write_text(readme+'\n'+handover)
(P/'DECISION.json').write_text(json.dumps(dict(input_reuse_results=[{k:r[k] for k in ('model','before_ms','after_ms','reduction_pct','reduction_pct_ci95','repeats')} for r in rows if r['stage']=='perf' and r['reference']=='before' and r['variant']=='inputs' and r['cohort'].startswith('2x')],status='complete',original_unchanged=True,automatic_merge=False,kv_restore_only=verdicts,kv_13b_caution='repeat b nearly neutral; component reduction is clearer than whole-step speedup',combined_model_parity_verified=True,combined_single_process_measured=True,combined_repeated_ab_measured=False),indent=2)+'\n')
print('Finished report; no automatic application',verdicts)
