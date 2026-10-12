"""Reconcile the completed paper-code audit into the canonical merge document."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent
d = json.loads((R / 'ANALYSIS.json').read_text())
if d['status'] != 'complete' or not (R / 'REPORT.md').exists():
    raise RuntimeError('Finish the report before updating the handoff')
full = {x['contrast']: x for x in d['aggregates'] if x['scope'] == 'full'}
names = {'chain_b1': 'Llama2 chain B1', 'llama3_chain_b1': 'Llama3 chain B1',
         'chain_b8': 'Llama2 chain B8', 'tree_b1': '원본 B1 tree + 최소 tree graph',
         'kv_b1': '원본 B1 tree + KV 정확성 수정', 'unified_b1': '원본 B1 tree → unified 실행기',
         'tree_chain_b1': '원본 B1 tree + chain fast만 적용'}
lines = ['**27. 실제 원 논문 checkout 성능 검증과 최소 이식 판정 — Round7 (2026-10-12)**', '',
         '기존 graph OFF는 현재 unified 실행기의 acceptance graph OFF였으며 원 논문 실행기가 아니었다. '
         '이번에는 원격 `feat/duet-p2tree-g0`가 여전히 `a82f7d2`임을 재확인하고 해당 commit의 clean sparse checkout을 실행했다. '
         '현재 systems runtime은 `df8ac6e`다. 기준 source와 원격 조회는 [REMOTE_BASELINE](round7/REMOTE_BASELINE.json)에 있다.', '',
         '**27.1 실제 추가 작업과 검증 범위**', '',
         '- 원 논문 checkout용 대조 도구, 독립 process/SHM/GPU pair 관리, 실제 반환 토큰 기반 집계, '
         '정확성 검사, 원시 결과 감사·복원 도구를 추가했다. Production `ssd/ssd`와 원 논문 branch는 변경하지 않았다.',
         f'- 표준 모델 실행 {d["completed_jobs"]}건 중 전체480입력 실행은 '
         f'{sum(x["scope"] == "full" for x in d["rows"])}건이다. 별도 full-model prefix 검사2개와 준비 smoke·수학적 대조를 보존했다. '
         '480은 저장된6개 group×80 전체이며 원 데이터셋 전체나 새 held-out 평가가 아니다.',
         '- 공통 adapter: AMD dtype 호환, SHM/종료 격리, prefix lookup OFF, B개씩 고정 요청 group, cap/clip·partial-B step 집계 제외. '
         '검증 외 변경 효과를 혼합하지 않았으며 [ADAPTERS](round7/ADAPTERS.md)에 전부 기록했다.',
         '- 고정 설정: T1, K1/K2=4/2, exit21/16, P1/P2 G=M=8/4, C3, 원래 residual/topm/mix0/q_path, 입력512/출력96. '
         '모델은 dense Llama2-7B+AMD135m(FP16), Llama3-8B+Qwama0.5B(BF16), RTX4090 targetTP1/draft1.',
         '- Chain512개의 coupled 출력, tree864개의 coupled 출력 대조 및 모든 최소 이식의 full 실행 출력 일치를 검사했다. '
         '원 KV 결함4개는 원본에서 실패·수정 후 통과했다. 별도 checkout에 물리적으로 patch를 적용한 chain/tree smoke도 원본 출력과 같았다.',
         '', '**27.2 성능 결론 — 아래 감소율은 전체 verification wall 기준**', '',
         '| 비교 | Verify ms 원본→후보 | Verify 변화 | TPS* 변화 | 판정 근거 |',
         '|---|---:|---:|---:|---|']
for key, name in names.items():
    a = full[key]
    note = '출력·AL·hit 동일' if a['outputs_equal'] else 'RNG/생성 경로 차이; 묶음 비교'
    lines.append(f'| {name} | {a["verify_base_ms"]:.3f} → {a["verify_candidate_ms"]:.3f} | '
                 f'{a["verify_change_pct"]:+.2f}% | {a["tps_change_pct"]:+.2f}% | {note} |')
lines += ['', 'Unified 및 tree+chain-fast는 PIX2,3와 NODE0,1에서 각각 수행한 대조의 기술적 평균이다. '
          '각 pair 안에서 원본/후보를 비교했고 두 후보는 원본 대조군을 공유한다. 다른 항목은 고정 pair에서2–3회 반복했다. '
          '개별 반복, 전체 AL/hit/TPS, 포함률과 raw TPS는 [Round7 REPORT](round7/REPORT.md)에 있다. '
          'TPS*는 nominal-B·non-boundary decode step의 값이며 serving 전체 TPS가 아니다.', '',
          '**27.3 다른 서버에 무엇을 옮길지**', '',
          '- **01 chain fast:** 이식 추천. SSD와 DUET 공통 적용. 양의 온도 JIT만 최적화하고 greedy/mixed/non-JIT 일반 경로는 원본 유지. '
          'Tree workload에서는 cache-miss chain 검증에도 쓰인다. 새 B>1 tree executor 전체가 필요한 수정은 아니다.',
          '- **02 최소 tree graph 후보:** 이식 비추천. Full 반복에서 전체 latency 개선이 재현되지 않았다. '
          '원본을 보존한 실험 후보와 음성 결과를 보관하되 기본 활성화하지 않는다.',
          '- **03 KV 정확성 수정:** 성능 기여가 아닌 정확성 수정으로 이식 추천. 실제 page table, 마지막 query, 최초 출력 경계, 올바른 block hash만 수정한다. '
          'Tree glue/promotion/defragmentation을 교체하지 않는다. 일반 decode 성능 차이는 반복마다 방향이 달랐다.',
          '- Prefix cold/cached는 두 full 모델에서 B1/B8×길이512/768/800×3회 모두 greedy 출력이 같았다. '
          '512-token B8 prefill은 약369–376ms→19–20ms였다. 이는 반복 prefix 재사용의 진단이며 tree glue 대체 효과가 아니다.',
          '- B>1 tree graph는 새 기능 내부에서 별도 판단한다. Round6 graph OFF→ON+trim의 B8 verification은 '
          'Llama2−6.48%/Llama3−4.08%, TPS는+0.96%/+1.25%였다. 원본 B>1 dynamic tree가 없어 원 논문 대비 개선율로 바꿔 쓰지 않는다.',
          '', '이식 파일과 적용 검사는 [PATCHES](round7/PATCHES.md), 최종 판단과 한계는 [DECISIONS](round7/DECISIONS.md)를 따른다. '
          'Verification/KV의 **실험 완료와 사용자 채택 승인은 별개**다. 자동 merge나 production 기본값 변경은 수행하지 않았다. '
          '기존 승인 항목(실제 온도, G>M, 집계, B 제한, wire)도 이 세 패치에 섞어 넣지 않았다.', '',
          '**27.4 자료 보존·재현·남은 적용 범위**', '',
          '- [REPORT](round7/REPORT.md)의 표·그림, [RUN_INDEX](round7/RUN_INDEX.csv)의 실행별 지표, '
          '[ANALYSIS](round7/ANALYSIS.json)의 설정·감사, [STRATIFIED](round7/STRATIFIED.json)의 phase/길이별 사후 분석, '
          '[PROFILE_BREAKDOWN](round7/PROFILE_BREAKDOWN.json)의 상세 계측을 함께 본다.',
          '- `round7/archive.py --restore`로 raw/log/telemetry/profile을 복원한 뒤 '
          '`analyze.py`, `stratify.py`, `profile_analysis.py`, `make_figures.py`, `make_report.py` 순서로 재생성한다. '
          'gzip SHA256 manifest로 파일을 검증한다.',
          '- 초기 AMD dtype 실패와 디스크 부족 부분 실행4개는 [FAILURES](round7/FAILURES.md)에 남겼고 성능 판단에서 제외했다. '
          '중단된 반복은 같은 설정으로 처음부터 다시 완료했다. 기존 Round6 비압축 사본 정리는 이미 보존된 gzip과 byte/SHA 일치를 확인한 복원 가능한 파일에 한정했다.',
          '- 원 논문70B/Blackwell TP2에서의 효과 크기는 아직 실측하지 않았다. 해당 서버의 production runner에서 '
          '동일 입력·파라미터·집계로01/03을 각각 대조한다. 현재 benchmark helper는 targetTP1/draft1 고정이므로 '
          '이를 그대로 TP2 재현 도구라고 부르지 않는다. 새로운 장비에서 빠르다는 보장은 하지 않는다.',
          '- Whole unified executor 채택은 B>1 기능과 연결된 별도 결정이다. 원본 B1의 작은 성능 차이를 이유로 '
          'KV/통신/scheduler/tree 실행기 전체를 무조건 가져오지 않는다. 최종 topology별 결과를 확인한 뒤 선택한다.', '', '']
section = '\n'.join(lines)
path = R.parent / 'MERGE_REVIEW.md'
text = path.read_text()
start = text.find('**27. 실제 원 논문 checkout')
end = text.index('**부록 A.')
text = text[:start if start >= 0 else end] + section + text[end:]
notice = ('**최신 성능 이식 판정은27절과 [Round7 보고서](round7/REPORT.md)를 따른다 (2026-10-12).** '
          '원 논문 checkout과 직접 대조해 chain fast·KV 정확성 수정·tree 후보를 분리했다. '
          '원 논문 branch 자동 merge는 하지 않았으며 미승인 verification/KV 채택은 검토 대상으로 유지한다.\n\n')
if notice not in text:
    pos = text.index('\n\n') + 2
    text = text[:pos] + notice + text[pos:]
text = text.replace('이후 GPU 검증과 집계 변경은26절을 따른다.', '이후 집계 변경은26절, 실제 원 논문 성능 대조는27절을 따른다.')
text = text.replace('**10/12의 집계 변경과 verification/KV 검증은 26절 및 [Round6 보고서](round6/REPORT.md)가 최신이다.**',
                    '**10/12 집계는26절/Round6, 원 논문 checkout과의 성능 대조·최소 이식은27절/Round7이 최신이다.**')
text = text.replace('최신 승인과 후속 검증은26절을 따른다.', '최신 승인과 후속 검증은26–27절을 따른다.')
path.write_text(text)
print('Canonical handoff updated')
