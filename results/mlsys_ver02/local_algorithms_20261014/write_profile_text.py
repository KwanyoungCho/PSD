"""Render the separate diagnostic phase table without treating it as primary TPS."""
import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'PROFILE_RESULTS.json').read_text());assert len(r)==4
lines=['48문항(각 Spec-Bench group에서 8개)을 따로 측정했다. Profile 모드는 독립 capture audit를 비활성화하므로 무계측 primary와 같은 preparation-free 인증으로 취급하지 않는다. 모든 arm이 8개 generate epoch(2 warmup + 6 측정)를 기록했고 각 유효 step의 target/draft 구간이 빠짐없이 대응했다. 이벤트 상한에 의한 누락은 없었다. [구간 그림 PNG](phases.png) / [PDF](phases.pdf), [원시 집계](PROFILE_RESULTS.json).','','| 변경 | 유효 step | Target pre | Proxy | Target post | Accept | P1 replay / total | P2 replay / total |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,label in [('base','기존'),('root_only','Root만'),('tree_only','Tree만'),('combined','Root+tree')]:
 a=r[name];t=a['processes']['target_rank0'];d=a['processes']['draft'];v=lambda x,k:x[k]['mean_ms']
 lines.append(f"| {label} | {a['eligible_steps']} | {v(t,'batch_target_pre'):.3f} | {v(t,'batch_target_proxy'):.3f} | {v(t,'batch_target_post'):.3f} | {v(t,'batch_tree_accept'):.3f} | {v(d,'batch_p1_replay'):.3f} / {v(d,'batch_p1_total'):.3f} | {v(d,'batch_p2_replay'):.3f} / {v(d,'batch_p2_total'):.3f} |")
lines+=['','단위 ms/호출. 서로 겹치거나 포함되는 구간이므로 모두 더해 한 step의 직렬 latency로 사용하지 않는다. Total에는 CPU launch gap·준비가 포함되고, P2 total에는 P1/P2의 공통 bulk cache export도 포함된다.','', '**이 측정은 tree 점수가 공짜인 최적화라고 보여주지 않는다.** Tree-only의 replay는 P1 20.208→22.796ms, P2 8.500→9.561ms로 늘었다. AL 향상과 비용 사이 tradeoff가 있다. 방문 문맥·shape가 다른 online profile이므로 이 전체 차이를 순수 점수 kernel 비용이라고 단정하지 않는다. Combined에서 늘어난 host gap/대기와 계측 영향도 겹쳐, 단일 원인으로 확정하지 않았다.','','| 변경 | P1 완료→proxy 도착 여유 중앙값 (ms) | P2 완료→target 후처리 완료 여유 중앙값 (ms) |','|---|---:|---:|']
for name,label in [('base','기존'),('root_only','Root만'),('tree_only','Tree만'),('combined','Root+tree')]:
 d=r[name]['deadlines'];lines.append(f"| {label} | {statistics.median(x['p1_slack_ms'] for x in d):.3f} | {statistics.median(x['p2_ready_slack_ms'] for x in d):.3f} |")
lines+=['','음수는 draft 구간이 해당 target 기준시점보다 늦게 끝났다는 뜻이다. Proxy 도착은 P1 뒤의 blocking wait가 아니라 별도 관측 stream의 수신 완료 event로 잡았다. 따라서 P1 지연을 실제로 구분할 수 있다. GPU별 CUDA event를 host 기준시각에 맞춘 근사이므로 sub-ms 차이는 과해석하지 않는다.','', '**현재 K1=4/K2=2를 완전히 숨겨지는 최적 설정이라고 볼 근거는 없다.** 적어도 이 profile에서는 P1/P2가 늦게 끝나는 구간이 많다. 단, 계측이 추가된 별도 표본이므로 실제 무계측 deadline 비율이나 최적 K를 이 숫자만으로 확정하지 않는다. 정확한 시간 기반 재튜닝은 선택할 최종 root/tree 정책을 고정한 후 필요하다. 이번에는 품질을 구분하려고 같은 K와 예산을 유지했다.','']
p=HERE/'REPORT.md';s=p.read_text();start=s.index('## 3-1. Phase 비용과 시간 예산')+len('## 3-1. Phase 비용과 시간 예산');end=s.index('## 4. 기능·수식 검증',start);s=s[:start]+'\n\n'+'\n'.join(lines)+'\n\n'+s[end:];p.write_text(s)
