"""Render only after all frozen controls and physical-port checks finish."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
d=json.loads((R/'ANALYSIS.json').read_text())
expected={'chain_b1':3,'llama3_chain_b1':2,'chain_b8':3,'tree_b1':2,'kv_b1':2,'unified_b1':2,'tree_chain_b1':2}
full={x['contrast']:x for x in d['aggregates'] if x['scope']=='full'}
if d['incomplete']:raise RuntimeError('Incomplete controls remain')
for key,n in expected.items():
 if key not in full or full[key]['repeats']!=n:raise RuntimeError('Missing full confirmation '+key)
for model in ('llama2','llama3'):
 p=json.loads((R/(model+'_paper_prefix.json')).read_text())
 if p['status']!='complete' or not all(c['outputs_equal'] for c in p['cases']):raise RuntimeError('Prefix verification incomplete')
port=json.loads((R/'PORT_INTEGRATION.json').read_text())
if not all(x['outputs_equal'] and x['warm_outputs_equal'] for x in port):raise RuntimeError('Physical patches disagree')
decisions=(R/'DECISIONS.md').read_text()
stratified=json.loads((R/'STRATIFIED.json').read_text())['pairs']
causal=['','같은 phase와 proposal/context 길이 구간을 대조한 full480의 상세 값은 다음과 같다. Source0=miss chain,1=P1 hit,2=P2 hit다. 두 GPU 연결 조건의 phase별 평균을 요약하며, unified는 같은 토큰 문맥의 counterfactual은 아니다.','',
        '| 후보 | Phase | Verify ms 원본→후보 | 차이 ms |','|---|---|---:|---:|']
for contrast in ('unified_b1','tree_chain_b1'):
 for source in (0,1,2):
  entries=[s for p in stratified if p['contrast']==contrast for s in p['phase_means'] if s['source']==source]
  b=statistics.mean(s['base_ms'] for s in entries);c=statistics.mean(s['candidate_ms'] for s in entries)
  causal.append(f'| {contrast} | {("Miss chain","P1 hit","P2 hit")[source]} | {b:.3f} → {c:.3f} | {c-b:+.3f} |')
causal+=['','상세 profile은 [PROFILE_BREAKDOWN.json](PROFILE_BREAKDOWN.json)에 있다. 4개 warm prompt 뒤16개 측정 prompt에서 nominal B/cap/clip 기준을 적용하고 capture step과 다음 step을 제외했다. Label이 발생한 step의 조건부 평균이며, CUDA event span에는 CPU launch 간격이 포함될 수 있다. 서로 겹치는 span들을 더해서 전체 wall time으로 해석하지 않는다.','',
          '그림의 원본 acceptance span에는 KV commit dispatch가 포함되고, unified acceptance 뒤에는 별도 commit span이 있다. 따라서 두 acceptance 막대의 차이를 동일 범위의 순수 kernel 단축량으로 쓰지 않는다. 전체 verification 비교는 위의 host wall 표를 따른다.','',
          '![Target execution spans](figs/05_target_spans.png)','']
names={'chain_b1':'Llama2 chain B1','llama3_chain_b1':'Llama3 chain B1','chain_b8':'Llama2 chain B8',
       'tree_b1':'Llama2 최소 tree graph B1','kv_b1':'Llama2 KV 정확성 수정 B1','unified_b1':'Llama2 unified tree B1',
       'tree_chain_b1':'Llama2 원본 tree + chain fast B1'}
lines=['# Round7 — 원 논문 코드 기준 latency 검증과 선별 이식', '',decisions,'',
'## 비교 기준과 완료 범위','',
'- 논문 분기 기준 `a82f7d24fb36827a9a81a3567f344dccb71f193e`를 별도 clean checkout에서 실행했다. Systems 소스는 `df8ac6e`다.',
'- Round6 graph OFF는 현재 unified 실행기의 acceptance graph OFF였다. 이번 기준군은 원 논문 실행기이며, 두 비교의 개선율을 혼용하지 않는다.',
f'- 표준 모델 실행 {d["completed_jobs"]}건의 결과·전체 입력 완료·실제 반환 토큰·최대 B·동일 집계 모집단을 감사했다. 별도 prefix 모델2개와 준비 smoke 결과도 보존했다.',
'- Full은 저장된6개 group×80=480개 입력 전체다. 원본 각 benchmark 데이터셋 전체 또는 새 held-out 데이터라는 뜻은 아니다.',
'- Chain32조건/512개의 같은 RNG 출력 대조, tree36조건/864개 같은 coin 출력 대조, KV 원본4개 실패→좁은 수정4개 통과를 확인했다.',
'- 원 논문70B/Blackwell TP2 환경을 재현한 것은 아니다. 이 서버의 full-weight7B/8B와 RTX4090에서 측정한 코드 효과다.',
'', '## 고정 설정', '',
'| 항목 | 설정 |','|---|---|',
'| 모델 | LayerSkip-Llama2-7B + AMD135m FP16; LayerSkip-Llama3-8B + Qwama0.5B BF16 |',
'| GPU | 각 A/B 비교별 동일 pair. Tree2,3 PIX; Llama2 chain4,5 PIX; KV/chain B8은6,7 PIX; Llama3 chain0,1 NODE. Unified 및 tree+chain fast는 R0=2,3 PIX/R1=0,1 NODE. Target TP1/draft1 |',
'| Sampling | T=1. 원 코드의 proxy T=1도 실제 온도와 같아 온도 정책을 바꾼 효과가 섞이지 않음 |',
'| Draft | K1=4/K2=2, exit Llama2=21/Llama3=16, draft/proxy fanout2/1 |',
'| Tree | P1/P2 G=M=8/4, C=3, P1 roots2, dynamic allocation, q_path, 원래 residual/topm/mix0 |',
'| Input/output | 입력 최대512, 출력 최대96, EOS 존중. Full 앞16입력 warm;48입력 진단 앞8입력 warm |',
'| Repeats | Full chain Llama2 B1/B8은3회, 나머지 full 대조2개. 주 대조는 독립 process/seed와 A→B/B→A. 후속 tree+chain fast는 앞선 원본 control을 재사용 |',
'| B | 원 scheduler를 수정하지 않고 B개씩 묶어 입력. 모든 요청은 완료하고 partial-B tail은 성능 집계에서만 제외 |',
'',
'공통 benchmark adapter는 SHM/종료 격리, AMD dtype 호환, prefix lookup 비활성화(잘못된 과거 cached 출력과 성능을 비교하지 않기 위함), 실제 출력 집계다. Decode KV/tree glue는 유지한다. [ADAPTERS](ADAPTERS.md)에 모두 명시했다. 이 비교를 원 논문의 역사적 TPS 재현이나 최선의 root/tree 설정 탐색으로 해석하지 않는다.',
'', '## 전체480 입력 — verification과 TPS', '',
'Verification은 target forward를 포함하는 기존 verifier의 host wall 범위다. Step 시간은 scheduling/draft wait/verification/postprocess를 포함한다. TPS*는 실제 B가 nominal B이고 cap/clip이 없는 batch-step의 반환 token/시간이다. AL/hit도 같은 모집단이며 prefill·tail을 포함한 전체 serving TPS와 다르다.', '',
'| 비교 | 반복 | Verify ms: 원본→후보 | Verify 변화 | TPS*: 원본→후보 | TPS 변화 | 모든 출력 일치 |',
'|---|---:|---:|---:|---:|---:|---|']
for key in expected:
 a=full[key];lines.append(f'| {names[key]} | {a["repeats"]} | {a["verify_base_ms"]:.3f} → {a["verify_candidate_ms"]:.3f} | {a["verify_change_pct"]:+.2f}% | {a["tps_base"]:.2f} → {a["tps_candidate"]:.2f} | {a["tps_change_pct"]:+.2f}% | {"예" if a["outputs_equal"] else "아니오: RNG/실행 경로 차이"} |')
lines+=['','평균 변화율은 각 동일-seed 대조의 변화율을 평균했다. 다음 범위는 process 반복의 관측 최솟값/최댓값이며 통계적 신뢰구간이 아니다. 작은 차이를 보편적인 우위로 해석하지 않는다. Unified 및 tree+chain fast는 PIX/NODE 각각1회 대조의 기술적 평균이며 같은 topology의 반복 평균이 아니다. 두 후보는 같은 원본 대조군을 공유한다. Tree+chain fast는 원본 tree executor를 유지하고 cache-miss chain 검증에만01을 적용한 후속 확인이다.','',
'| 비교 | Verify 변화 범위 | TPS 변화 범위 |','|---|---:|---:|']
for key in expected:
 a=full[key];v=a['verify_change_pct_range'];t=a['tps_change_pct_range'];lines.append(f'| {names[key]} | {v[0]:+.2f} ~ {v[1]:+.2f}% | {t[0]:+.2f} ~ {t[1]:+.2f}% |')
lines+=['','GPU 연결별 직접 대조:','','| 후보 / GPU pair | Verify 원본→후보 | 변화 | TPS 원본→후보 | 변화 |','|---|---:|---:|---:|---:|']
for p in d['pairs']:
 if p['contrast'] in ('unified_b1','tree_chain_b1') and p['scope']=='full':
  topology='PIX' if p['gpu_pair']=='2,3' else 'NODE'
  lines.append(f'| {p["contrast"]} / {p["gpu_pair"]} {topology} | {p["verify_base_ms"]:.3f} → {p["verify_candidate_ms"]:.3f} ms | {p["verify_change_pct"]:+.2f}% | {p["tps_base"]:.2f} → {p["tps_candidate"]:.2f} | {p["tps_change_pct"]:+.2f}% |')
lines+=['','[전체 그림 PDF](figs/ALL_FIGURES.pdf) · 각 그림의 개별 PNG/PDF는 figs/에 있다.','',
'![Actual paper latency and TPS](figs/01_paper_latency_tps.png)','',
'![Step breakdown](figs/02_step_breakdown.png)','',
'## AL·cache hit·제외율', '',
'| 비교 | AL 원본→후보 | Hit 원본→후보 |','|---|---:|---:|']
for key in expected:
 a=full[key];lines.append(f'| {names[key]} | {a["al_base"]:.4f} → {a["al_candidate"]:.4f} | {a["hit_base"]*100:.2f}% → {a["hit_candidate"]*100:.2f}% |')
lines+=['','| 비교 | 집계 포함 step 원본→후보 | 참고 raw decode TPS 원본→후보 |','|---|---:|---:|']
for key in expected:
 a=full[key];lines.append(f'| {names[key]} | {a["included_fraction_base"]*100:.2f}% → {a["included_fraction_candidate"]*100:.2f}% | {a["raw_tps_base"]:.2f} → {a["raw_tps_candidate"]:.2f} |')
lines+=['','![AL and hit](figs/03_al_hit.png)','',
'모든 실행의 포함/전체 step 수와 실제 출력 수는 [ANALYSIS.json](ANALYSIS.json)의 rows.summary에 있다. B8은 고정8개 요청 group 안에서 한 요청이 먼저 끝나면 이후 partial-B step을 제외한다. 실제 continuous serving의 요청 교체 정책 성능을 측정한 것이 아니다.',
'', '## B>1 tree와 기존 Round6 결과의 해석', '',
'원 논문 실행기는 B>1 dynamic tree를 지원하지 않았다. 따라서 아래는 현재 실행기 안에서의 최적화 검증이며, 이번 원 논문 checkout 전후 비교에 합치지 않는다. Round6의 T=.7, 출력128 및 continuous input 설정도 이번 T=1/출력96/fixed-group 설정과 다르다.', '',
'| 현재 unified 실행기 B8 | Graph OFF → graph+trim ON: verification 변화 | TPS* 변화 |',
'|---|---:|---:|',
'| Llama2 | −6.48% | +0.96% |',
'| Llama3 | −4.08% | +1.25% |',
'',
'출처: [Round6 full480 결과](../round6/REPORT.md). 두 process 반복 평균이다. Faster verification이 줄인 시간이 추가 draft 대기로 일부 상쇄될 수 있으므로 graph kernel 개선율을 TPS 개선율로 쓰지 않는다. B1 기존 논문 이식과 B>1 기능 도입의 의사결정도 분리한다.',
'', '## 사후 원인 분석', '',
'[STRATIFIED.json](STRATIFIED.json)은 phase(source0/1/2), 실제 proposal 길이, prefix 길이128-token bin별로 비교한다. Unified의 RNG 차이 때문에 AL/shape 분포가 달라진 영향을 진단하기 위해 공통 bin에서 pooled-weight 표준화 값을 별도로 계산했다. 공통 지지영역 비율도 기록하며 원래 전체-population 표를 대체하지 않는다.',
*causal,
'', '## KV: 정확성 수정과 glue 대체는 별개', '',
'| 항목 | 원본에서 재현된 문제 | 이식하는 수정 |','|---|---|---|',
'| Paged prefix | 불연속 page를 단순 flatten하여 다른 위치의 KV를 읽음 | 실제 page table/요청별 KV 길이로 attention |',
'| 완전 cached prompt | query 길이0으로 다음 logits를 계산할 수 없음 | 마지막 query1개 유지 |',
'| Preemption | re-prefill 후 이미 만든 출력과 남은 출력 예산의 기준이 바뀜 | 최초 prompt 경계 보존 |',
'| 완료 block hash | 실제 완료 block 대신 마지막 할당 page에 hash 기록 | block_index로 page 및 이전 block 선택 |',
'',
'위 수정은 전체 KV 재배열/defragmentation이나 draft tree glue를 없애는 KV promotion이 아니다. 원 논문의 tree stage/restore/accepted-path commit은 유지한다. 잘못된 출력이 나오는 cached 원본을 빠른 기준군으로 삼지 않았다.',
'', '![Prefix prefill](figs/04_paper_prefix.png)','',
'Prefix 실험은 좁은 KV 패치를 적용한 원 논문 engine에서 cold/cached를 비교한다. 두 모델의 B1/B8×길이512/768/800×3회, 각18조건에서 greedy 출력이 일치해야 최종 보고서를 생성한다. 반복 prompt를 인위적으로 만든 진단이며 자연 workload의 prefix hit율이나 DUET root hit율이 아니다.',
'', '## 이식·인수인계', '',
'- [PATCHES.md](PATCHES.md):01 chain fast,02 opt-in tree 후보,03 KV 정확성 수정. 최종 판정에 따라 필요한 파일만 적용한다.',
'- [REPRODUCE.md](REPRODUCE.md): 다른 서버에서 자료 복원·선별 patch 검토·GPU 재대조 순서와 유지할 조건.',
'- [PORT_INTEGRATION.json](PORT_INTEGRATION.json): 별도 checkout에 물리적으로 patch를 적용한 chain/tree smoke의 원본 대비 출력 대조.',
'- [FAILURES.md](FAILURES.md), [DISK_INTERRUPTION.json](DISK_INTERRUPTION.json): dtype 초기화 실패 및 디스크 부족으로 중단·재시작한 전체 실행. 실패 부분을 결과에 섞지 않았다.',
'- [ENVIRONMENT.json](ENVIRONMENT.json): 패키지, GPU UUID/topology, 외부에서 가져온 함수의 source SHA. 각 raw에는 원본 source/adapter SHA, 입력, seed, 설정을 보존했다.',
'- [RUN_INDEX.csv](RUN_INDEX.csv): 실행별 모델/B/온도/GPU/원본 commit, latency·AL·hit·TPS·포함률. 정확한 model kwargs와 seed는 ANALYSIS.json 및 raw 결과에 있다.',
'- [ARCHIVE_MANIFEST.json](ARCHIVE_MANIFEST.json): 원시 결과·log·telemetry·상세 profile의 gzip과 SHA256. 원 논문 브랜치에 자동 merge하지 않았다.',
'', '재분석:', '',
'```bash', 'python results/mlsys_coverage/round7/archive.py --restore',
'python results/mlsys_coverage/round7/analyze.py', 'python results/mlsys_coverage/round7/stratify.py',
'python results/mlsys_coverage/round7/profile_analysis.py',
'python results/mlsys_coverage/round7/make_figures.py', 'python results/mlsys_coverage/round7/make_report.py','```','']
(R/'REPORT.md').write_text('\n'.join(lines));print('Report written')
