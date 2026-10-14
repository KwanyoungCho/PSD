"""Render matched algorithm comparisons; measured latency is shape-dependent."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
LABEL={'base':'Existing ver0.2','root_only':'Root only','tree_only':'Tree only',
       'combined':'Root + tree','root_norm':'Full norm',
       'root_position':'Full norm + position','root_proxy':'Proxy + same position'}
ORDER=['base','root_only','tree_only','combined','root_norm','root_position','root_proxy']
NAMES={'l2':'LayerSkip Llama2-7B / AMD-135M','l3':'LayerSkip Llama3-8B / Qwama-0.5B'}

def main():
 r=json.loads((HERE/'RESULTS.json').read_text())
 assert r['status']=='complete','Wait for all prespecified arms'
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
 colors=['#777777','#368bb8','#d69a2d','#30946c']
 for row,model in enumerate(('l2','l3')):
  arms=r['models'][model]
  for col,(key,mult,title) in enumerate((('al',1,'AL (tokens / request-step)'),('hit',100,'Cache hit (%)'),('tps',1,'Aggregate decode TPS (secondary)'))):
   ax=axs[row,col];ys=[arms[n]['summary'][key]*mult for n in ORDER[:4]]
   ax.bar(range(4),ys,color=colors)
   for i,y in enumerate(ys):ax.text(i,y,f'{y:.3f}' if key=='al' else f'{y:.2f}',ha='center',va='bottom',fontsize=9)
   ax.set_xticks(range(4),['Existing','Root','Tree','Both']);ax.set_ylim(0,max(ys)*1.18)
   ax.set_title(title);ax.set_ylabel(NAMES[model] if col==0 else '')
 fig.suptitle('Matched 2 x 2: B8, T=0.7, K1/K2=4/2, exit=21, nodes=8/4\n480 first-turn prompts x 2 passes; input cap 512, output cap 64')
 fig.savefig(HERE/'comparison.png',dpi=180);fig.savefig(HERE/'comparison.pdf');plt.close(fig)

 pairs=[('base -> root_norm','Full normalization'),('root_norm -> root_position','Position mixture'),
        ('root_position -> root_only','Complement score'),('root_proxy -> root_only','Complement vs proxy'),
        ('root_only -> combined','Tree with local root')]
 fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
 for col,(metric,title) in enumerate((('hit_pp','Change in cache hit (percentage points)'),('al','Change in AL (tokens / request-step)'))):
  ax=axs[col]
  for m,model in enumerate(('l2','l3')):
   ds=[r['comparisons'][model][k]['quality'][metric] for k,_ in pairs]
   y=np.arange(len(pairs))+(m-.5)*.18;delta=np.array([d['delta'] for d in ds]);ci=np.array([d['ci95'] for d in ds])
   ax.hlines(y,ci[:,0],ci[:,1],color=['#368bb8','#d69a2d'][m]);ax.plot(delta,y,'o',label=model.upper(),color=['#368bb8','#d69a2d'][m])
  ax.axvline(0,color='grey',ls='--',lw=1);ax.set_yticks(range(len(pairs)),[s for _,s in pairs]);ax.invert_yaxis();ax.set_title(title);ax.legend(loc='lower right',fontsize=9)
 fig.suptitle('Paired prompt-group bootstrap 95% intervals (60 clusters)\nRepeated passes share clusters; process-to-process variance is not estimated')
 fig.savefig(HERE/'ablations.png',dpi=180);fig.savefig(HERE/'ablations.pdf');plt.close(fig)

 lines=['# 동일 조건 알고리즘 비교','','주 지표는 cache hit와 AL. TPS는 보조 지표이며 모델별 최적 파라미터를 탐색한 결과가 아니다. 같은 소스·동일 960개 출력인 L3 root-only의 제외 실행과 정상 재실행에서도 TPS가 420.57→369.72로 달랐다. 문항 bootstrap은 process 사이 변동을 추정하지 않으므로 TPS 차이를 확정적인 알고리즘 가속으로 인용하지 않는다.','']
 for model in ('l2','l3'):
  lines+=['## '+NAMES[model],'','| 설정 | Batch | AL | Hit (%) | TPS | Verify (ms) | P1-hit AL | P2-hit AL |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for name in ORDER:
   a=r['models'][model][name];s=a['summary'];ph=a['source']
   lines.append(f"| {LABEL[name]} | 8 | {s['al']:.4f} | {s['hit']*100:.3f} | {s['tps']:.2f} | {s['verify_ms']:.3f} | {ph['1']['al']:.4f} | {ph['2']['al']:.4f} |")
  lines+=['','| 변화 | ΔAL [95% CI] | Δhit (%p) [95% CI] |','|---|---:|---:|']
  for name,x in r['comparisons'][model].items():
   q=x['quality'];a=q['al'];h=q['hit_pp']
   lines.append(f"| {name} | {a['delta']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | {h['delta']:+.3f} [{h['ci95'][0]:+.3f}, {h['ci95'][1]:+.3f}] |")
  lines+=['','| 설정 | P1 실제 node 평균 | P1 최대 깊이 평균 | P2 실제 node 평균 | P2 최대 깊이 평균 |','|---|---:|---:|---:|---:|']
  for name in ORDER[:4]:
   shape=r['models'][model][name]['shape'];a,b=shape['1'],shape['2']
   lines.append(f"| {LABEL[name]} | {a['valid_k']:.3f} | {a['tree_max_depth']:.3f} | {b['valid_k']:.3f} | {b['tree_max_depth']:.3f} |")
  lines+=['','| 설정 | 전원 hit step (%) | 하나 이상 miss (%) | 실제 JIT step (%) | 포함 step 수 |','|---|---:|---:|---:|---:|']
  for name in ORDER[:4]:
   b=r['models'][model][name]['batch_metrics']
   lines.append(f"| {LABEL[name]} | {100*b['all_hit']:.3f} | {100*b['any_miss']:.3f} | {100*b['actual_jit_step']:.3f} | {b['steps']} |")
  lines+=['']
 lines+=['## 같은 480문항의 요청별 보조 지표','','주 표와 달리 tail batch와 마지막 step을 포함한다. 각 pass를 문항 내 평균한 뒤 문항 사이 평균한다. AL은 clipping 전 verifier 이벤트 기준이므로 실제 반환 토큰/step과 다를 수 있다. Cache hit의 집계 가중치도 주 표와 다르다.','','| 모델 | 설정 | 문항 수 | AL (문항 평균) | Hit (%) |','|---|---|---:|---:|---:|']
 for model in ('l2','l3'):
  for name in ORDER[:4]:
   a=r['common_requests'][model]['arms'][name];m=a['metrics']
   lines.append(f"| {model} | {LABEL[name]} | {a['questions']} | {m['al']:.4f} | {100*m['hit']:.3f} |")
 lines+=['','## 집계 조건','','- Batch가 실제 B=8이고 cap/clipping/graph 준비의 영향을 받지 않은 step을 분자·분모에 함께 포함한다.','- AL은 verifier가 내보내는 토큰 수 기준이며 correction/bonus를 포함한다. P1/P2-hit AL은 해당 source를 실제 사용한 요청에 대한 조건부 값이다.','- Full 480은 480개 첫 턴이며 두 pass를 수행한다. 960개의 독립 문항이나 560턴 평가라는 뜻이 아니다.','- CI는 동일 8문항 group 60개를 재표집한다. 여러 ablation을 탐색적으로 비교한 구간이며 다중 비교 보정은 하지 않았다.','- Prefix cache OFF. 동일 조건 상대 비교이며 논문 원본의 절대 TPS가 아니다.','- 전체 요청·문항 분모와 cap 이후 tail을 포함한 별도 요청별 평균은 RESULTS.json의 common_requests에 있다. 그 request_decode_tps는 aggregate TPS와 정의가 다르다.','']
 (HERE/'TABLES.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
