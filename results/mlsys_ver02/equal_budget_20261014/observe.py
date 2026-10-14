"""Read-only GPU/job observer; mirror completed remote evidence and analyze it.

Never kills processes, changes runtime, launches an experimental arm or treats
partial runs as completed. Uses the existing SSH control connection.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
LOCAL=Path('/data/chokwans99/equal-tree-paper-full-20261014')
MIRROR=LOCAL/'remote_mirror';MIRROR.mkdir(exist_ok=True)
SSH=['ssh','-S','/tmp/duet-equal-eslab18-20261014.sock','-o','BatchMode=yes','-o','ConnectTimeout=10','-p','1100','chokwans99@115.145.135.238']
PAIRS={'local_45_paper':'4,5','local_67_paper':'6,7','nova_23_paper':'2,3','nova_45_paper':'4,5','nova_67_paper':'6,7','nova_23_retry_disk':'2,3','nova_67_retry_disk':'6,7','local_45_retry_external':'4,5'}
probe=r'''
from pathlib import Path
import json,subprocess,shutil
root=Path(BASE);out=[]
for name in NAMES:
 d=root/name;plan=json.loads((d/'plan.json').read_text()) if (d/'plan.json').exists() else []
 manifest_error=None
 try:manifest=json.loads((d/'campaign.json').read_text()) if (d/'campaign.json').exists() else []
 except (OSError,json.JSONDecodeError) as exc:manifest=[];manifest_error=str(exc)
 done={x['name']:x for x in manifest};jobs=[]
 for p in plan:
  n=p['name'];r=done.get(n);status=r['status'] if r else 'queued';groups=None;last=None
  if r and status=='complete' and r.get('external_pids'):status='complete_external_excluded'
  log=d/(n+'.log')
  if log.exists():
   with log.open('rb') as f:
    f.seek(0,2);size=f.tell();f.seek(max(0,size-100000));text=f.read().decode(errors='replace')
   progress=[x for x in text.splitlines() if x.startswith('PAPER_GROUP')]
   if progress:last=progress[-1]
   if not r:status='running' if 'PAPER_COMPLETE' not in text else 'finishing'
  jobs.append(dict(name=n,status=status,last_group=last,external_pids=(r or {}).get('external_pids',[])))
 out.append(dict(name=name,directory=str(d),jobs=jobs,manifest=manifest,manifest_error=manifest_error))
print(json.dumps(dict(queues=out,disk_free_bytes=shutil.disk_usage(root).free,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True))))
'''

def run(argv,**kw):
 return subprocess.check_output(argv,text=True,timeout=kw.pop('timeout',120),**kw)

def snapshot(remote=False):
 base='/home/chokwans99/equal_budget_results' if remote else str(LOCAL)
 names=[n for n in PAIRS if n.startswith('nova_' if remote else 'local_')]
 code='BASE='+repr(base)+'\nNAMES='+repr(names)+'\n'+probe
 return json.loads(run([*SSH,'python3 -c '+shlex.quote(code)] if remote else [sys.executable,'-c',code]))

def sync(remote):
 for queue in remote['queues']:
  dest=MIRROR/queue['name'];dest.mkdir(exist_ok=True)
  for name in ('plan.json','campaign.json'):
   src=queue['directory']+'/'+name
   # rsync can fail for not-yet-created campaign.json; do not hide other errors.
   if name=='campaign.json' and (not queue['manifest'] or queue.get('manifest_error')):continue
   run(['rsync','-az','-e',shlex.join(SSH[:-1]),SSH[-1]+':'+src,str(dest)+'/'])
  for job in queue['manifest']:
   if job.get('status')!='complete':continue
   mark=dest/(job['name']+'.mirrored.json')
   if mark.exists():continue
   names=[job['name']+s for s in ('.json','.log','.telemetry.jsonl','_profile/')]
   for name in names:
    # Preserve directory nesting and completed, immutable raw data only.
    target=dest/name
    if name.endswith('/'):target.mkdir(exist_ok=True)
    run(['rsync','-az','--checksum','-e',shlex.join(SSH[:-1]),SSH[-1]+':'+queue['directory']+'/'+name,str(target)],timeout=300)
   mark.write_text(json.dumps(dict(mirrored_at=time.time(),source=queue['directory'],run=job),indent=2)+'\n')

def render(status,analysis=None):
 now=datetime.now().astimezone().isoformat(timespec='seconds')
 finalized=(HERE/'FINAL_REPORT.md').exists() and status['state']=='runs_validated'
 lines=['# 동일 draft 입력 예산의 tree AL 비교', '',f'최종 확인: {now}', '',
 '**상태: '+('14/14 실행·검증·최종 분석 완료. [최종 분석 보고서](FINAL_REPORT.md)와 [그림 PDF](figs/ALL_FIGURES.pdf) 참조.' if finalized else '전체 실행 및 자동 검증 완료. 결과 해석·최종 검토 필요.' if status['state']=='runs_validated' else '전체 데이터 실험 진행 중. 부분 결과로 우열을 확정하지 않음.')+'**','',
 '- 모델: LayerSkip-Llama2-7B + AMD-Llama-135m, full FP16.',
 '- B=1, T=0.7, raw Spec-Bench 480문항/560턴, 입력 절단 없음, 출력 상한1024, 문맥 상한2048.',
 '- MT-Bench 두 턴은 기존 논문 실행처럼 독립 raw prompt. 문항별 턴 평균 후 문항 평균.',
 '- K1=K2=8, P1/P2 root 각각8개, forward 폭8개. Phase당 물리 입력64행.',
 '- Calibration 수락률 없이 root 점수 × draft 경로 확률. P1/P2 root 점수는 모든 비교군에서 동일 규칙.',
 '- Chain 형태(C1), 마지막 깊이 경쟁(C3/C8), 전체 미확장 말단 경쟁(C3/C8); 검증상한32. C8은 상한64도 진단.',
 '- 모든 비교군은 같은 unified tree 실행기를 실제 B1로 사용. C1은 chain 모양의 대조군이며 chain 전용 fast path 성능 비교가 아님.',
 '- 로컬 seed16101, eslab18 seed26101. 각 서버/seed 안에서 모든 비교군을 비교. 서버와 seed 효과를 별도로 추정하는 설계는 아님.',
 '- 주 비교 5개 + 검증상한 진단2개 × 2seed = 총14개 full 실행. B>1은 후속 검증.','',
 '아래 표는 최초 실패·외부 간섭 기록도 보존한 대기열 이력이다. 재실행 완료분으로 대체한 최종 유효 결과는 양 서버 각각7개다.' if finalized else '',
 '| 서버 | GPU | 실행 대기열 | 완료/계획 | 실행 기록 |','|---|---|---|---:|---|']
 for server in ('local','remote'):
  for q in status.get(server,{}).get('queues',[]):
   complete=sum(x['status']=='complete' and not x['external_pids'] for x in q['jobs'])
   active=[x['name']+': '+x['status']+(' / '+x['last_group'] if x['last_group'] else '') for x in q['jobs'] if x['status'] not in ('complete','queued')]
   lines.append(f"| {'로컬4090' if server=='local' else 'eslab18 3090'} | {PAIRS[q['name']]} | {q['name']} | {complete}/{len(q['jobs'])} | {'; '.join(active) or '대기/완료'} |")
 lines += ['', '## 검증 및 해석 기준','',
 '- 최종 비교는 560턴 UID 순서·입력token ID·출력상한·종료상태·동일 source hash를 검증한 full 결과만 포함.',
 '- 외부 GPU 프로세스가 관측되거나 실행이 실패한 결과는 제외. 사용 중인 GPU의 외부 작업을 종료하지 않음.',
 '- AL은 확정 root/target token 1개 포함. 상한 도달·잘린 step은 제외하며 graph 준비 step은 AL에는 포함(TPS에서는 제외).',
 '- 전체/P1-hit/P2-hit/miss AL 및 분모, 6개 하위과제, 실제 hit tree 깊이·폭, 생성 대비 검증상한 손실을 함께 분석.',
 '- 신뢰구간은 문항 단위 paired bootstrap. P1/P2 조건부 표본이 정책별로 달라짐을 명시. 같은 방문 문맥을 고정한 인과 실험은 아님.',
 '- 생성 입력 행수가 같아도 샘플링 자식 수·호스트 작업·verification 노드 수·실측 latency까지 같다는 뜻은 아님.',
 '- eslab18 최초 full 7개 중5개는 완료. Chain 및 last C8 M64는 저장공간 소진으로 각각545/529턴 raw만 남아 실패. 실패 원본을 보존하고 같은 runtime/seed로 전체560턴 '+('재실행 완료.' if finalized else '재실행 중.')+' 잘린 campaign manifest도 원본을 남기고 실패 상태로 복구.',
 '- 검증상한64 첫 smoke는 NCCL 메모리 할당 실패. 동일 runtime 재실행은 6턴/262 forests 모두 통과(phase당 active64행). 최초 실패 원인은 미확정이며 실패 raw를 보존.',
 '- 기존 회귀283개, 입력 프로토콜22개, 입력검사4개 통과 기록은 각각 소스시점이 다름. 최종 압축기록 변경은 위 M64 smoke와 분석검사3개로 추가 검증.','',
 '## 완료된 비교 결과','']
 if not analysis or not any(x.get('arms') for x in analysis.values()):
  lines += ['아직 full arm이 완료되지 않아 AL 우열 표를 작성하지 않았다.']
 else:
  for server,c in analysis.items():
   lines += ['',f'### {server} — {c["status"]}', '', '| 방법 | 전체 AL | P1-hit AL | P2-hit AL | miss AL | hit(%) | AL 문항 수 |','|---|---:|---:|---:|---:|---:|---:|']
   def fmt(v):return '—' if v is None else f'{v:.4f}'
   for name,arm in c['arms'].items():
    q=arm['question_quality']['all']; vals=[fmt(q[k]['mean']) for k in ('al','p1_al','p2_al','miss_al','hit_percent')]
    lines.append('| '+name+' | '+' | '.join(vals)+' | '+str(q['al']['questions'])+' |')
   lines += ['', '방법별로 방문한 문맥과 hit 조건부 모집단이 다르다. '+('통계·구조·한계 해석은 FINAL_REPORT.md 참조.' if finalized else '전체 arm이 완료되기 전에는 최종 순위를 판단하지 않는다.')]
 lines += ['', '## 파일 및 재현','',
 '- `PLAN.md`, `PAPER_PROTOCOL.json`, `SPEC_BENCH_FULL.json`: 설계·논문 입력조건 근거.',
 '- `SOURCE_SHA256.json`, `MODEL_SHA256.json`: 양 서버에 같은 소스/모델을 복제했는지 대조.',
 '- `deployment_plan.py`, `local_*_paper.json`, `nova_*_paper.json`: GPU쌍별 순차실행 계획.',
 '- `status.json`: 자동 갱신 진행상태. `RESULTS.json`: 완료 full arm의 질문별/phase별/구조별 집계 및 CI.',
 '- `observe.py`: 진행 확인, 완료된 원격 raw의 로컬 복제, 집계·본 문서 갱신. Runtime 수정이나 외부 프로세스 종료 없음.',
 f'- 로컬 raw: `{LOCAL}`. 원격 raw: `/home/chokwans99/equal_budget_results`.',
 '- Runtime/bench는 실행 중 고정. 이번 실험 변경은 아직 로컬 연구 브랜치에 commit/push하지 않았음. 원격 c91a1f4는 복제 소스의 snapshot commit.','']
 (HERE/'REPORT.md').write_text('\n'.join(lines))

def once():
 state=dict(updated=time.time(),state='running',results_complete=False)
 state['local']=snapshot();state['remote']=snapshot(True)
 # Save current job status before transfer/analysis, so a collection failure
 # cannot leave old jobs advertised as still running.
 (HERE/'status.json').write_text(json.dumps(state,indent=2)+'\n');render(state)
 sync(state['remote'])
 signature=[(s,q['name'],j['name'],j['status'],j['external_pids']) for s in ('local','remote') for q in state[s]['queues'] for j in q['jobs']]
 fingerprint=json.dumps(signature,sort_keys=True)
 if not (HERE/'observation_signature.json').exists() or (HERE/'observation_signature.json').read_text()!=fingerprint:
  cmd=[sys.executable,'-B',str(HERE/'analyze.py'),'--local',*[str(LOCAL/n) for n in PAIRS if n.startswith('local_')],'--remote',*[str(MIRROR/n) for n in PAIRS if n.startswith('nova_')]]
  state['analysis_output']=run(cmd,timeout=900)
  (HERE/'observation_signature.json').write_text(fingerprint)
 analysis=json.loads((HERE/'RESULTS.json').read_text()) if (HERE/'RESULTS.json').exists() else None
 if analysis and all(x['status']=='complete' for x in analysis.values()):
  state.update(state='runs_validated',results_complete=True)
 (HERE/'status.json').write_text(json.dumps(state,indent=2)+'\n');render(state,analysis)
 print(datetime.now().isoformat(timespec='seconds'),state['state'],[(s,q['name'],[(j['name'],j['status']) for j in q['jobs']]) for s in ('local','remote') for q in state[s]['queues']],flush=True)
 return state['results_complete']

def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
 while True:
  try:done=once()
  except Exception as e:
   print(datetime.now().isoformat(),type(e).__name__,str(e),flush=True)
   (HERE/'OBSERVER_ERROR.json').write_text(json.dumps(dict(time=time.time(),error=repr(e)),indent=2)+'\n');done=False
   if a.once:raise
  if done or a.once:return
  time.sleep(60)
if __name__=='__main__':main()
