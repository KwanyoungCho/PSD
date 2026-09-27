"""Data-derived report and standalone research figures; no online gain claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
LABELS={'q_path':'기존 q-path','frozen':'기존 frozen reach (8문항)',
    'same8_depth':'동일 8문항 + depth','same8_rich':'동일 8문항 + depth/entropy',
    'refit_q':'Cross-fit phase/sibling/q','depth_q':'Cross-fit + depth',
    'cond_depth_q':'Cross-fit conditional q + depth','draft_rich':'Cross-fit + depth/entropy',
    'direct_beta':'Cross-fit 직접 local transition'}


def ci(item):
    return f"{item['mean']:+.5f} [{item['ci95'][0]:+.5f}, {item['ci95'][1]:+.5f}]"


def main():
    x=json.loads((HERE/'analysis.json').read_text());audit=json.loads((HERE/'audit.json').read_text())
    examples=json.loads((HERE/'examples.json').read_text())
    fanout=json.loads((HERE/'fanout_analysis.json').read_text())
    bound=json.loads((HERE/'sibling_bound.json').read_text())
    m=x['scores']['nonfinal:all']['means'];allm=x['scores']['all_events:all']['means']
    b=x['budgets']['all:4'];g=x['prospective_onechild']['all']
    total=sum(r['trees'] for r in x['coverage']);nodes=sum(r['nodes'] for r in x['coverage'])
    body=[
        '# DUET tree 사후 분석 — 전체 데이터 진단',
        '',
        '2026-09-22. Root는 cache hit, tree는 hit 이후 descendant AL을 목표로 구분한다.',
        '',
        '**이 보고서의 MSE·고정-pool 선택 개선은 새 online tree 정책의 AL 개선율이 아니다.** '
        '기존 관측 없는 3-seed 성능 실험은 AL 2.0651→2.1003 (+1.70%), 차이 CI [-0.0025,+0.0721]로 그대로 보존했다. '
        '이번 진단은 왜 차이가 작고 무엇을 더 개선할 수 있는지 조사한다.',
        '',
        '## 1. 실행 범위와 신뢰성',
        '',
        f'- 전체 Spec-Bench 480문항/560턴 × 두 생성 정책 × seed 1 = **1,120턴**, '
        f'**{total:,} served trees / {nodes:,} nodes**를 기록했다. Warmup 제외, 모든 hit를 한 번씩 수집했다.',
        '- 같은 모델, exit56, T=.7, depth4/2, node8/6, G=M, 기존 root policy를 사용했다. '
        '출력 상한128, natural EOS, 실제 이전 응답을 포함한 두 번째 turn이다. 공식 chat-template/1024 cap 재현은 아니다.',
        '- Target/draft 전체 분포로 alpha·reach·terminal 확률과 coin 분산을 계산했다. '
        f"매32번째 tree의 원본 분포 **{sum(r['raw_audits'] for r in audit['jobs']):,}개**를 재계산해 검증했다.",
        '- 진단 observer의 CPU/GPU 동기화는 비동기 스케줄에 영향을 준다. '
        '기존 무관측 성능 결과와 절대 AL/TPS를 합치지 않는다. 새 점수들은 **같은 tree**에서 비교했다.',
        '- 기본 진단 가중치는 task 균등 → original question 균등 → 해당 question의 tree 평균이다. '
        '두 생성 정책을 pooled하며 policy별 민감도도 제공한다. CI는 task 내 question cluster 2,000회 bootstrap; '
        '양 turn과 두 정책을 같은 cluster로 유지한다.',
        '- Cross-fit CI는 산출된 보정 table에 조건부인 탐색적 문항 불확실성이다. '
        '각 bootstrap에서 보정 table을 다시 학습하지 않으며 calibration 표본/다른 seed의 불확실성까지 포괄하지 않는다.',
        '- 표의 기본값은 종료 event 제외다. 종료 여부는 수락 결과에 의존할 수 있어, '
        'coin 변동/예측 분해는 전체 event도 함께 확인한다. 많은 보조 비교의 CI에는 다중 비교 보정이 없다.',
        '',
        '| 수집 정책 | 문항/턴 | 모든 tree | 비종료 tree | node |',
        '|---|---:|---:|---:|---:|']
    for r in x['coverage']:body.append(f"| {r['policy']} | {r['questions']}/{r['turns']} | {r['trees']:,} | {r['nonfinal_trees']:,} | {r['nodes']:,} |")
    body += ['', '## 2. 현재 수식에 의미가 있는가', '',
        '정확한 목적함수는 E[AL|tree,p,q]=Σ rho(v)다. '
        'rho(child)=rho(parent)×앞선 형제 전부 거절 확률×현재 형제 수락 확률이므로 '
        '기존 q-path보다 수락 법칙과 직접 연결된다. 그러나 보정한 alpha와 그 곱은 여전히 추정값이다.', '',
        '**현재 frozen table의 형제 순위 한계는 수식으로 확인됐다.** 첫 형제의 예측 alpha는 '
        f"P1에서 최소 {bound['phases']['1']['first_alpha_lower']:.6f}, "
        f"P2에서 최소 {bound['phases']['2']['first_alpha_lower']:.6f}로 모두 .5보다 크다. "
        '뒤 형제의 도달 질량을 전부 합쳐도 1−alpha0보다 클 수 없으므로, '
        '같은 부모의 첫 형제가 대안에 있으면 frozen 점수는 항상 그 형제를 우선한다. '
        '343개 q-bin 조합×2 phase를 모두 검사했다. 다른 root 간 순위까지 고정이라는 뜻은 아니다.', '',
        '| 점수 | Node reach MSE ↓ | 기대 AL 예측 bias | 예산4 subset의 실제 기대 AL ↑ |',
        '|---|---:|---:|---:|']
    for name,label in LABELS.items():body.append(f"| {label} | {m[name+'_reach_mse']:.6f} | {m[name+'_al_bias']:+.4f} | {b['means'][name]:.4f} |")
    body += ['', f"Frozen−q-path MSE 차이: **{ci(x['scores']['nonfinal:all']['reach_vs_q']['frozen'])}**.",
        f"Cross-fit rich−frozen MSE 차이: **{ci(x['scores']['nonfinal:all']['reach_vs_frozen']['draft_rich'])}**.",
        f"Cross-fit rich−같은 자료로 refit한 q table MSE 차이: **{ci(x['scores']['nonfinal:all']['reach_vs_refit']['draft_rich'])}**.", '',
        '동일8문항 보정은 기존 77 calibration trees만 쓴다. Cross-fit은 각 task의 문항을 '
        '5개 fold로 나눠 최대 384문항의 관측 tree로 보정하고 나머지 96문항에 평가한다. '
        '평가 문항의 양 turn·양 생성 정책을 보정에서 모두 제외했다. '
        'Hit가 없는 문항은 전체 실행 coverage에는 포함하고 undefined tree 지표를 0으로 대체하지 않는다. '
        'Cross-fit 개선을 원래 8문항의 저비용 calibration만으로 달성한 결과처럼 해석하지 않는다. '
        '새 neural head나 추가 inference target forward는 없고 scalar lookup 통계 보정이다.', '',
        '예산4 subset 값은 원래 node 수가 4보다 큰 tree에서만 계산한다. '
        'MSE 표본·다른 예산·기존 무관측 성능 실행과 절대 AL을 직접 비교하지 않고 '
        '같은 pool/예산 안의 정책 차이를 비교한다.', '',
        '## 3. Phase·깊이에서 무엇을 못 맞추는가', '',
        '| Phase | q-path MSE | Frozen MSE | Rich cross-fit MSE | Frozen AL bias | Rich AL bias |',
        '|---|---:|---:|---:|---:|---:|']
    for p in [1,2]:
        v=x['scores'][f'nonfinal:{p}']['means']
        body.append(f"| P{p} | {v['q_path_reach_mse']:.6f} | {v['frozen_reach_mse']:.6f} | {v['draft_rich_reach_mse']:.6f} | {v['frozen_al_bias']:+.4f} | {v['draft_rich_al_bias']:+.4f} |")
    body += ['', '| 평가 tree 생성 정책 | q-path MSE | Frozen MSE | Pooled rich MSE | q-path 자료만 보정한 rich | Frozen-policy 자료만 보정한 rich |',
        '|---|---:|---:|---:|---:|---:|']
    for policy,v in x['per_policy'].items():
        body.append(f"| {policy} | {v['q_path_reach_mse']:.6f} | {v['frozen_reach_mse']:.6f} | {v['draft_rich_reach_mse']:.6f} | {v['rich_from_q_path_reach_mse']:.6f} | {v['rich_from_phase_sibling_q_bin_reach_mse']:.6f} |")
    body += ['', 'Policy별 보정에서도 평가 question은 다른 정책의 기록까지 보정에서 제외했다. '
        '이는 관측된 두 정책 사이의 transfer 검사이며 새 알고리즘의 모든 frontier에 대한 off-policy 보장은 아니다.']
    body += ['', '아래 depth 표는 node를 합산한 보조 진단이다. 위 question-balanced 주 표와 가중치가 다르다.', '',
        '| Phase/depth | Nodes | 정확 alpha | Frozen alpha | q-path MSE | Frozen MSE | Rich MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in x['strata']:
        if r['axis']=='depth':body.append(f"| P{r['phase']}/d{r['bin']} | {r['nodes']:,} | {r['alpha']:.4f} | {r['predicted_alpha']:.4f} | {r['q_path_mse']:.5f} | {r['frozen_mse']:.5f} | {r['rich_mse']:.5f} |")
    body += ['', '| Phase / 형제 순서(0부터) | Nodes | 정확 alpha | Frozen alpha | q-path MSE | Frozen MSE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in x['strata']:
        if r['axis']=='sibling':body.append(f"| P{r['phase']}/s{r['bin']} | {r['nodes']:,} | {r['alpha']:.4f} | {r['predicted_alpha']:.4f} | {r['q_path_mse']:.5f} | {r['frozen_mse']:.5f} |")
    body += ['', '| Phase / target-draft overlap 구간 | Nodes | 정확 alpha | Frozen alpha | Frozen reach MSE |',
        '|---|---:|---:|---:|---:|']
    overlap_labels=['≤.25','(.25,.5]','(.5,.75]','(.75,.9]','>.9']
    for r in x['strata']:
        if r['axis']=='overlap_bin':body.append(f"| P{r['phase']} / {overlap_labels[r['bin']]} | {r['nodes']:,} | {r['alpha']:.4f} | {r['predicted_alpha']:.4f} | {r['frozen_mse']:.5f} |")
    body += ['', 'Overlap=Σmin(p,q)는 이 분석에서만 사용하는 target 기반 설명 변수다. '
        'Online scalar alpha table의 입력에는 넣지 않았다. 전체 문맥의 overlap과 이미 샘플된 특정 '
        '토큰의 alpha도 같은 양은 아니다.']
    body += ['', 'Alpha 평균은 exact attempt mass로 가중했다. 뒤 형제가 실제로 시도되지 않았다고 '
        '거절 label 0을 주지 않았다. 같은 phase/q라도 depth나 문맥의 p/q 불일치가 달라질 수 있다.', '',
        '| Oracle 치환 진단 | Reach MSE |', '|---|---:|',
        f"| Frozen 전체 | {m['frozen_reach_mse']:.6f} |",
        f"| 진짜 parent reach × frozen local transition | {m['oracle_parent_frozen_local_mse']:.6f} |",
        f"| Frozen parent reach × 진짜 local transition | {m['frozen_parent_oracle_local_mse']:.6f} |", '',
        '치환 결과는 오류 요인을 알아보는 보조 실험이다. 두 감소량을 더해 전체 오차의 '
        '인과적 기여율이라고 부르지 않는다. 곱의 상호작용이 있다.', '',
        '## 4. 실제 실패 사례와 정보 부족', '']
    for label,title in [('confident_draft_wrong','Draft가 확신했지만 target은 수락하지 않는 경우'),('unlikely_draft_accepted','Draft 확률은 낮지만 target은 수락하는 경우')]:
        entry=examples[label];e=entry['examples'][0]
        body += [f"**{title}**: 첫 형제, attempt≥.3 조건에서 {entry['eligible_nodes']:,} node를 찾았다. "
            f"이 조건에서 q-path 오차가 가장 큰 사례의 token `{e['token']!r}` (문항 {e['uid']}, P{e['phase']}, depth{e['depth']}): "
            f"q={e['q']:.6f}, p={e['p']:.6f}, alpha={e['alpha']:.6f}, "
            f"진짜 reach={e['reach']:.6f}, q-path={e['q_path']:.6f}, frozen reach={e['frozen_reach']:.6f}.", '']
    body += ['q≥.9인 첫 형제만 제한했을 때의 오류도 확인했다. 아래 비율은 exact attempt mass 가중이다.', '',
        '| Phase | Nodes | alpha<.5 비율 | alpha<.1 비율 |', '|---|---:|---:|---:|']
    for p,r in x['high_confidence_errors'].items():
        body.append(f"| P{p} | {r['nodes']:,} | {100*r['alpha_below_half']:.2f}% | {100*r['alpha_below_tenth']:.2f}% |")
    body += ["Phase/depth/q/entropy/qmax의 좁은 bin 안에서도 alpha<.05와 alpha>.95 사례가 함께 존재하는 "
        f"bin을 {examples['feature_collisions']['count']:,}개 찾았다. 이는 현재 feature만으로 개별 정답을 "
        '구별하기 어렵다는 실제 사례다. 더 풍부한 context나 token identity를 써도 불가능하다는 증명은 아니다.', '',
        'q=(.9,.1), 후보 A가 같아도 p=q면 alpha=1, p=(0,1)이면 alpha=0이다. '
        '따라서 q와 그 entropy만으로 모든 문맥의 정확한 수락을 보장할 수 없다. '
        '이것과 실제 workload에서 평균 예측을 개선할 수 있는지는 별개의 문제다.', '',
        '## 5. 원래 맞출 수 없는 coin 변동', '',
        f"전체 event에서 exact conditional coin variance 평균은 **{allm['coin_variance']:.4f} token²**, "
        f"실제 AL−exact expected AL의 제곱오차는 **{allm['oracle_observed_mse']:.4f}**다. "
        f"두 평균의 차이가 아닌 signed AL 잔차는 **{ci(x['scores']['all_events:all']['observed_minus_exact'])}**다.", '',
        'p/q를 완전히 알아도 새 수락 coin의 결과까지 맞힐 수는 없다. '
        '이번 점수 평가는 실제 AL 한 번 대신 exact expected reach/AL을 정답으로 삼아 그 변동을 분리했다. '
        '기대 AL의 점수 오류는 이 coin 변동으로 설명하거나 면책할 수 없다.', '',
        '## 6. 점수와 선택 알고리즘 중 어느 쪽에 여지가 있는가', '',
        '같은 실현 pool에서 ancestor+sibling-prefix 제약을 지키며 예산4 subset을 선택했다. '
        '대안 노드를 새로 생성하지 않은 **진단용** 비교다.', '',
        '| 방법 | 선택된 subset의 exact expected AL |', '|---|---:|',
        f"| q-path greedy | {b['means']['q_greedy']:.5f} |",
        f"| q-path DP | {b['means']['q_path']:.5f} |",
        f"| Frozen reach DP | {b['means']['frozen']:.5f} |",
        f"| Rich cross-fit DP | {b['means']['draft_rich']:.5f} |",
        f"| Exact reach oracle DP | {b['means']['oracle']:.5f} |", '',
        f"Rich−frozen: **{ci(b['differences_vs_frozen']['draft_rich'])}**. "
        f"Oracle−frozen: **{ci(b['differences_vs_frozen']['oracle'])}**.", '',
        f"Rich 보정으로 subset 자체가 바뀐 비율은 {100*b['choice_rates']['draft_rich_changed_vs_frozen']:.2f}%, "
        f"기대 AL이 늘어난 비율은 {100*b['choice_rates']['draft_rich_win_vs_frozen']:.2f}%, "
        f"줄어든 비율은 {100*b['choice_rates']['draft_rich_loss_vs_frozen']:.2f}%다.", '',
        '현재 selector는 같은 depth==round 안에서 비교한다. 공통 scale 보정은 MSE를 줄여도 '
        '그 round의 순위를 바꾸지 않는다. 보정 table이 모두 공통 배율이라는 뜻은 아니지만, '
        '확률값을 더 잘 맞추는 것과 tree를 바꾸는 것이 다른 이유다.', '',
        '예산2/6 및 phase별 결과는 analysis.json에 함께 있다. 생성 예산 G=M인 실제 tree에서 '
        '모든 node를 유지하면 이 subset 최적화로 추가 AL을 얻지 않는다. '
        '남은 차이를 실제 이득으로 바꾸려면 **샘플을 뽑기 전** 확장·fanout 배분을 개선해야 한다. '
        'Token 값을 본 뒤 사후 pruning하는 정책은 별도 분포 보존 문제가 있어 이번에 적용하지 않았다.', '',
        '## 7. Reach만으로 충분한가: 다음 한 후보의 기대 이득', '',
        '한 토큰을 추가하기 전의 정확한 이득은 rho(u)×g1(u), '
        'g1(u)=Σmin(p_u,q_u)=1−TV(p_u,q_u)다. '
        '같은 depth에서 이미 확장되어 다음 p/q를 아는 위치들끼리 비교했다. '
        '실현된 자식 token의 운을 적분한 one-child 진단이다.', '',
        '| 선택 우선순위 | 선택 위치의 exact one-child gain |', '|---|---:|']
    for key,label in [('q_path_choice_gain','q-path'),('frozen_choice_gain','Frozen reach'),('draft_rich_choice_gain','Rich reach'),
        ('frozen_times_pred_g_choice_gain','Frozen reach × cross-fit g'),('rich_times_pred_g_choice_gain','Rich reach × cross-fit g'),
        ('rho_choice_gain','Exact rho만 사용'),('oracle_gain','Exact rho×exact g')]:
        body.append(f"| {label} | {g['means'][key]:.5f} |")
    body += ['', f"비교 집합 수 {g['comparisons']:,}. 진짜 rho만으로도 선택이 달라지는 비율 "
        f"{100*g['means']['rho_misrank']:.2f}%, 그 평균 gain 차이는 {g['means']['rho_regret']:.5f}다.",
        f"Frozen×예측 g − frozen: **{ci(g['differences_vs_frozen']['frozen_times_pred_g_choice_gain'])}**.",
        f"Rich×예측 g − frozen: **{ci(g['differences_vs_frozen']['rich_times_pred_g_choice_gain'])}**.", '',
        f"Rich 보정에 따른 위치 선택 변경률 {100*g['means']['rich_choice_changed']:.2f}%, "
        f"예측 g를 추가했을 때 변경률 {100*g['means']['g_choice_changed']:.2f}%다. "
        f"Depth1의 대안 집합에 첫 형제가 실제 포함된 {g['root_branch_comparisons']:,}개 비교에서 "
        f"frozen이 첫 형제를 고르는 비율은 "
        f"{100*g['root_branch']['frozen_selects_first_sibling']:.2f}%, "
        f"oracle의 최선이 뒤 형제인 비율은 {100*g['root_branch']['oracle_selects_later_sibling']:.2f}%다.", '',
        'G 보정은 node 생성 때 이미 아는 phase/sibling/q/depth/부모 q entropy만 입력으로 썼다. '
        '아직 forward하지 않은 node의 다음 q entropy를 입력으로 쓰지 않았다. '
        '단, training label은 실제 확장된 node에만 있어서 selection bias/coverage 한계가 있다. '
        '확장하지 않은 모든 leaf나 미적중 root에 이 결과를 일반화할 수 없다. '
        '여기의 대안 집합도 실제 scheduler의 전체 frontier를 재현한 것이 아니다.', '',
        '## 8. 형제 수 배분: 첫 후보부터 하나씩 주면 최적인가', '',
        '그렇지 않다. p=(0,1), q=(.9,.1)이면 첫 후보까지의 기대 수락 g(1)=.1, '
        '두 후보까지는 g(2)=1이다. 둘째 후보의 추가 이득 .9가 첫째 .1보다 크다. '
        'Residual/WOR에서는 추가 후보의 이득이 항상 감소하지 않는다.', '',
        '부모 A/B의 reach=.8/.2, g_A=(.1,1), g_B=(.9,1)이면 '
        '새 node 예산2를 하나씩 주면 .26, A에 둘 다 주면 .8이다. '
        '이는 정확한 반례이며 아래 표는 실제 분포에서 추가 검증한 결과다.', '',
        f"매32번째 원본 snapshot의 **{fanout['contexts']:,} expanded contexts / "
        f"{fanout['questions']}문항**에서 g(1), g(2)를 vocabulary 전체에 대해 정확히 적분했다. "
        'Monte Carlo가 아니다. O(V log V) 계산을 작은 vocab의 350개 완전 열거 사례와 대조했다. '
        '이 정밀 fanout 진단의 표본 범위는 전체 tree를 쓴 앞 절의 범위와 다르다.', '',
        '| Phase | Contexts | 평균 g(1) | 평균 g(2) | 둘째 추가 이득>첫째인 비율 | Reach 가중 비율 |',
        '|---|---:|---:|---:|---:|---:|']
    for phase in [1,2]:
        f=fanout['summary'][str(phase)];v=f['means']
        body.append(f"| P{phase} | {f['contexts']:,} | {v['g1']:.4f} | {v['g2']:.4f} | {100*v['nonconcave']:.2f}% | {100*f['pooled_reach_weighted_nonconcave']:.2f}% |")
    body += ['', '이미 확장된 동일-depth 부모 두 개에 새 node 예산2를 줄 때, '
        '1+1과 2+0/0+2를 비교했다. 모두 정확한 rho/p/q를 아는 oracle 진단이다. '
        '새 자식 token을 뽑기 전의 기대값을 비교하며, 한 단계 뒤 손자의 추가 가치는 포함하지 않는다.', '',
        '| Phase | 쌍 비교 수 | 하나씩 배분 | 최선 배분 | 차이 | 배분이 바뀌는 비율 |',
        '|---|---:|---:|---:|---:|---:|']
    for phase in [1,2]:
        f=fanout['summary'][str(phase)];v=f['allocation_means']
        body.append(f"| P{phase} | {f['pair_comparisons']:,} | {v['breadth']:.4f} | {v['oracle']:.4f} | {v['gap']:.4f} | {100*v['changed']:.2f}% |")
    body += ['', f"전체 차이 CI: **{ci(fanout['summary']['all']['gap_ci'])}**. "
        '이 값은 해당 두 후보의 한 번 배분에서 얻는 oracle gain이며, 전체 검증 AL에 그대로 더할 수 없다. '
        'q/p가 알려진 expanded subset 및 systematic raw sample에 조건부이고, '
        '실제 모든 frontier의 대안·forward/deadline 경쟁을 포함하지 않는다.', '',
        '**미래 target 정보를 쓰지 않는 배분 점수도 시험했다.** 기존8문항만으로 phase별 '
        'gamma(1)/gamma(2)를 보정하고, frozen reach×gamma(c)의 합이 가장 큰 '
        '(1,1)/(2,0)/(0,2)를 고른다. 추가 neural head나 새 full-corpus fitting은 없다.', '',
        '| Phase | Calibration gamma(1) | gamma(2) |', '|---|---:|---:|']
    for p,curve in fanout['original_8_prompt_gain_curves'].items():
        body.append(f"| P{p} | {curve[0]:.5f} | {curve[1]:.5f} |")
    body += ['', '| 같은 두-node 배분 문제의 선택 기준 | 실제 exact expected gain | 기존 하나씩 배분 대비 차이 [95% CI] |',
        '|---|---:|---|']
    for key,label in [('q_curve','q-path × calibration curve'),('frozen_curve','Frozen reach × calibration curve'),
        ('rho_curve','Exact reach × calibration curve'),('frozen_oracle_g','Frozen reach × exact g')]:
        f=fanout['summary']['all']
        body.append(f"| {label} | {f['allocation_means'][key]:.5f} | {ci(f['curve_vs_breadth'][key])} |")
    body += ['', '이 작은 배분 검사는 첫 번째 정책의 점수 오차 분석 뒤 추가한 탐색적 follow-up이다. '
        'Curve를 맞추는 자료는 기존8문항뿐이고 새 corpus outcome으로 계수를 조정하지 않았다. '
        '실제 생성/검증은 수행하지 않은 국소적인 한 단계 의사결정 검사다. '
        'C=3, 여러 round, root 간 배분까지 포함한 end-to-end AL 개선으로 주장하지 않는다.', '',
        '## 9. 깊이·예산 때문에 멈춘 부분', '',
        '| Phase | Depth cap에서 끝나는 질량 | Cap 전에 leaf에서 끝나는 질량 | 내부에서 모든 형제 거절 | Early-leaf 추가 길이 상한 |',
        '|---|---:|---:|---:|---:|']
    for p in [1,2]:
        v=x['scores'][f'nonfinal:{p}']['means']
        body.append(f"| P{p} | {v['depth_cap_terminal']:.4f} | {v['early_leaf_terminal']:.4f} | {v['internal_rejection_terminal']:.4f} | {v['early_leaf_extra_upper']:.4f} |")
    body += ['', '| Phase | Node cap 미달 tree 비율 | 평균 미사용 node 예산 | Early leaf이면서 raw-q threshold 아래인 terminal mass |',
        '|---|---:|---:|---:|']
    for p in [1,2]:
        v=x['scores'][f'nonfinal:{p}']['means']
        body.append(f"| P{p} | {v['underfilled_tree']:.4f} | {v['unused_node_budget']:.4f} | {v['early_leaf_below_q_threshold_mass']:.4f} |")
    body += ['', 'P2의 raw-q threshold=.03 아래 node는 이후 확장의 eligibility에서 제외된다. '
        '위 질량은 그 조건을 충족하는 early leaf의 질량이다. '
        'Threshold를 없애면 모두 확장할 수 있다는 뜻은 아니며 global width/root quota/root-prior threshold도 작용한다. '
        'P1 raw-q threshold는 0이다.']
    body += ['', '첫 번째 종료 구조 표의 마지막 열은 early leaf를 모두 남은 depth까지 공짜로 확장하고 전부 수락시킨다는 '
        '낙관적 상한이다. 현재 node/forward 예산에서 얻을 수 있는 개선량이 아니다. '
        f"진짜 reach가 0인 node 비율은 {100*m['zero_reach_node_fraction']:.2f}%이고, "
        f".01 미만은 {100*m['low_reach_node_fraction']:.2f}%다. "
        '그렇다고 이 node들을 사전에 알 수 있거나 안전하게 사후 삭제할 수 있는 것은 아니다.', '',
        '## 10. 결론의 범위와 재개 지점', '',
        '- Reach의 합이라는 목적함수는 정확하다. alpha를 추정하는 입력/보정, 그 곱의 의존성, '
        '확장 gain, 예산 알고리즘은 각각 추가 검증 대상이다.',
        '- MSE 개선, 고정-pool 선택 개선, 실제 tree 생성 AL 개선을 서로 구별한다. '
        '새 보정식의 online AL 개선은 이번 사후 분석만으로 확정하지 않는다.',
        '- 다음 구현의 우선순위는 기존 frozen reach와 후보 수별 gain curve를 사용하는 fanout 배분이다. '
        '동일8문항에서 depth/entropy feature만 늘린 개선은 확인하지 못했다. '
        'C=3/여러 round/전체 frontier로 확장한 뒤 실제 AL을 검증해야 한다.',
        '- 전체 frontier·미적중 root·확장되지 않은 node에 대한 탐색/감사 기록이 있어야 '
        '점수 때문에 놓친 후보와 eligibility/quota 때문에 제외된 후보를 완전히 구별할 수 있다. '
        '현재 저장 자료만으로 그 부분의 전역 최적성을 선언하지 않는다.',
        '- 사용자와의 수식 설명 TODO는 여전히 미완료로 남긴다.', '',
        '수식의 전체 유도: [THEORY.md](THEORY.md). 구체적 해석과 개발 우선순위: [FINDINGS.md](FINDINGS.md). '
        '원본 실패 사례: [examples.json](examples.json), subset 사례: [failure_examples.json](failure_examples.json). '
        '전체 통계: [analysis.json](analysis.json), 검사: [audit.json](audit.json), '
        '[math_audit.json](math_audit.json), 설계: [PLAN.md](PLAN.md).', '',
        '```bash', 'ssd/.venv/bin/python results/duet_tree_posthoc/validate.py',
        'ssd/.venv/bin/python results/duet_tree_posthoc/analyze.py',
        'ssd/.venv/bin/python results/duet_tree_posthoc/fanout.py',
        'ssd/.venv/bin/python results/duet_tree_posthoc/examples.py',
        'ssd/.venv/bin/python results/duet_tree_posthoc/sibling_bound.py',
        'ssd/.venv/bin/python results/duet_tree_posthoc/make_report.py', '```', '']
    (HERE/'REPORT.md').write_text('\n'.join(body))
    # Compact figure: error, fixed-pool choice, terminal structure, next-step gain.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names=['q_path','frozen','refit_q','depth_q','draft_rich','direct_beta']
    labels=['q-path','Frozen','Refit q','+Depth','+Entropy','Direct beta']
    vals=[m[n+'_reach_mse'] for n in names]
    bars=ax[0,0].bar(labels,vals,color=['#888888','#4378a8','#85a8c8','#69a991','#348169','#c49a58'])
    ax[0,0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax[0,0].set_ylim(0,max(vals)*1.2)
    ax[0,0].set_title('A  Node reach prediction error');ax[0,0].set_ylabel('MSE (lower is better)')
    vals=[b['means'][k] for k in ['q_greedy','q_path','frozen','draft_rich','oracle']]
    bars=ax[0,1].bar(['q greedy','q DP','Frozen DP','Rich DP','Oracle DP'],vals,color=['#888888','#aaa','#4378a8','#348169','#c49a58'])
    ax[0,1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax[0,1].set_ylim(0,max(vals)*1.18)
    ax[0,1].set_title('B  Fixed sampled pool, 4-node subset');ax[0,1].set_ylabel('Exact expected accepted descendants')
    bottom=np.zeros(2)
    for key,label,color in [('internal_rejection_terminal','All siblings rejected','#bd6a64'),('early_leaf_terminal','Leaf before depth cap','#c8aa61'),('depth_cap_terminal','Depth cap reached','#348169')]:
        vals=[x['scores'][f'nonfinal:{p}']['means'][key] for p in [1,2]]
        ax[1,0].bar(['P1 (depth 4)','P2 (depth 2)'],vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax[1,0].set_ylim(0,1);ax[1,0].set_ylabel('Exact terminal probability');ax[1,0].set_title('C  Why verification stops')
    ax[1,0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=1)
    keys=['q_path_choice_gain','frozen_choice_gain','draft_rich_choice_gain','rich_times_pred_g_choice_gain','rho_choice_gain','oracle_gain']
    vals=[g['means'][k] for k in keys]
    bars=ax[1,1].bar(['q-path','Frozen','Rich','Rich x g','True rho','Oracle'],vals,color=['#888','#4378a8','#348169','#69a991','#cfb77f','#c49a58'])
    ax[1,1].bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax[1,1].set_ylim(0,max(vals)*1.2)
    ax[1,1].set_title('D  One-child gain at observed expanded nodes');ax[1,1].set_ylabel('Expected added accepted token')
    fig.suptitle('DUET full-corpus posthoc diagnostics\nPrediction and conditional choices; not online policy AL gains',fontsize=14)
    fig.savefig(HERE/'diagnostics.png',dpi=180);fig.savefig(HERE/'diagnostics.pdf')
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    f=fanout['summary']['all'];keys=['breadth','q_curve','frozen_curve','rho_curve','oracle']
    vals=[f['allocation_means'][k] for k in keys]
    bars=ax[0].bar(['One each','q x curve','Frozen x curve','True rho x curve','Oracle'],vals,
        color=['#888','#aaa','#4378a8','#cfb77f','#c49a58'])
    ax[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9);ax[0].set_ylim(0,max(vals)*1.18)
    ax[0].tick_params(axis='x',labelsize=9,rotation=15)
    ax[0].set_ylabel('Exact expected added accepted tokens')
    ax[0].set_title('Same two-parent / two-new-node decision')
    for y,p in enumerate(['all','1','2']):
        stat=fanout['summary'][p]['curve_vs_breadth']['frozen_curve'];v=stat['mean'];lo,hi=stat['ci95']
        ax[1].errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#4378a8',capsize=5)
        ax[1].annotate(f'{v:+.4f}',(v,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
    ax[1].set_yticks([0,1,2],['All','P1','P2']);ax[1].invert_yaxis();ax[1].set_ylim(2.5,-.5)
    ax[1].axvline(0,color='#999',linestyle='--');ax[1].set_xlabel('Gain difference vs one-each allocation (95% CI)')
    ax[1].set_title('Frozen reach x curve from original 8 prompts')
    fig.suptitle(f"Prospective local allocation diagnostic: {f['pair_comparisons']:,} comparisons / {f['gap_ci']['questions']} questions\n"
        'This is not an end-to-end generation AL improvement',fontsize=12)
    fig.savefig(HERE/'allocation.png',dpi=180);fig.savefig(HERE/'allocation.pdf')
    numbers=dict(scope='Posthoc diagnostics; local allocation gain is not end-to-end AL improvement.',
        corpus_questions=480,executed_turns=1120,trees=total,nodes=nodes,
        q_path_reach_mse=m['q_path_reach_mse'],frozen_reach_mse=m['frozen_reach_mse'],
        frozen_mse_reduction_percent=100*(1-m['frozen_reach_mse']/m['q_path_reach_mse']),
        rich_crossfit_reach_mse=m['draft_rich_reach_mse'],
        budget4_q_path_dp=b['means']['q_path'],budget4_frozen_dp=b['means']['frozen'],
        budget4_rich_dp=b['means']['draft_rich'],budget4_oracle_dp=b['means']['oracle'],
        frozen_first_sibling_choice=g['root_branch']['frozen_selects_first_sibling'],
        oracle_later_sibling_choice=g['root_branch']['oracle_selects_later_sibling'],
        fanout_breadth=f['allocation_means']['breadth'],fanout_frozen_curve=f['allocation_means']['frozen_curve'],
        fanout_delta=f['curve_vs_breadth']['frozen_curve'],
        fanout_gain_percent=100*(f['allocation_means']['frozen_curve']/f['allocation_means']['breadth']-1),
        original_untraced_al=dict(q_path=2.0651326883,frozen=2.1003084298,delta_ci95=[-.00248884,.07205639]))
    (HERE/'NUMBERS.txt').write_text(json.dumps(numbers,indent=2)+'\n')
    print('report and figures written')


if __name__=='__main__':main()
