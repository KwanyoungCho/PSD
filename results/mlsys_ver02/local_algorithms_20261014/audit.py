"""Audit final evidence, source provenance and the prespecified measurement population."""
import hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/bench'))
from mlsys_tree_results import summarize

BASE=Path('/data/chokwans99/PSD-ver02-algorithm-base-20261014')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def audit_run(folder,job,primary=False):
 p=folder/(job['name']+'.json');r=json.loads(p.read_text())
 assert job['status']=='complete' and job['returncode']==0 and not job['external_pids'],job
 assert r['status']=='complete'
 assert not any(line[3:].startswith('ssd/') for line in r['paper_dirty'].splitlines()),(p,'dirty runtime or bench at launch')
 checkout=BASE if r['paper_commit'].startswith('18e912f') else ROOT
 assert checkout==BASE or r['paper_commit'].startswith('992b8b8'),r['paper_commit']
 expected={str(f.relative_to(checkout)):sha(f) for f in (checkout/'ssd/ssd').rglob('*.py')}
 assert r['source_hashes']==expected,(p,'runtime source differs')
 for name,h in r['adapter_hashes'].items():assert sha(ROOT/'ssd/bench'/name)==h,(p,name)
 assert r['summary']['actual_output_tokens']==sum(map(len,r['outputs']))==r['summary']['emitted_decode_tokens']
 assert all(r['args'][x]==v for x,v in [('temperature',.7),('batch',8),('k1',4),('k2',2),('exit_layer',21)]) if primary else True
 if primary:
  assert len(r['outputs'])==960 and len(r['question_indexes'])==480
  assert len(set(r['question_indexes']))==480
  assert r['env']['CUDA_VISIBLE_DEVICES'] in ('4,5','6,7')
  assert r['env']['PAPER_COLD_PREFIX']=='1'
 plan=next(j for j in json.loads((folder/'plan.json').read_text()) if j['name']==job['name'])
 root_mode=plan.get('env',{}).get('DUET_LOCAL_REVIEW_ROOT')
 if root_mode is not None:
  assert '[research ablation] '+root_mode+' ' in (folder/(job['name']+'.log')).read_text(),(p,'ablation startup log missing')
 s=summarize(r)
 if s['preparation'] is None:
  assert r['env'].get('SSD_PROFILE_DUET')=='1' and not primary,(p,'missing preparation audit')
 else:
  assert s['preparation']['affected_steps']==0,(p,s['preparation'])
 return dict(file=str(p),sha256=sha(p),commit=r['paper_commit'],runtime_source_match=True,runtime_clean_at_launch=True,
             adapters_match=True,root_ablation=root_mode,tree_policy=plan.get('env',{}).get('SSD_TREE_EXPANSION_POLICY','default'),outputs=len(r['outputs']),questions=len(r['question_indexes']),
             steps=r['summary']['steps'],all_steps=r['summary']['all_steps'],
             actual_output_tokens=r['summary']['actual_output_tokens'],external_pids=[],
             preparation=s['preparation'],temperature=r['args']['temperature'],batch=r['args']['batch'],
             runtime_file_count=len(expected))

def main():
 frozen=json.loads((HERE/'FROZEN_SOURCE.json').read_text())
 assert all(sha(ROOT/n)==h for n,h in frozen.items()),'Final source changed'
 calibration=json.loads((HERE/'VALIDATION.json').read_text())['calibration']
 assert all(sha(ROOT/'ssd/ssd/engine/helpers/tree_calibration'/n)==h for n,h in calibration.items())
 out=dict(status='complete',frozen_files=len(frozen),source_unchanged=True,runs={},excluded=[])
 base=Path('/data/chokwans99')
 selected={}
 for name in ['ver02-local-algorithms-full-l2-final-20261014','ver02-local-algorithms-full-l3-20261014','ver02-local-algorithms-l3-recheck3-20261014']:
  folder=base/name;model='l2' if '-l2-' in name else 'l3'
  for j in json.loads((folder/'campaign.json').read_text()):
   if j['status']=='complete' and not j['returncode'] and not j['external_pids']:
    selected[(model,j['name'])]=(folder,j)
   else:out['excluded'].append(dict(directory=str(folder),job=j))
 assert len(selected)==14
 for (model,arm),(folder,j) in selected.items():out['runs'][model+'/'+arm]=audit_run(folder,j,True)
 for suffix in ['final-greedy','profiles']:
  folder=base/f'ver02-local-algorithms-{suffix}-20261014'
  jobs=json.loads((folder/'campaign.json').read_text())
  assert len(jobs)==(3 if suffix=='final-greedy' else 4)
  for j in jobs:out['runs'][suffix+'/'+j['name']]=audit_run(folder,j)
 out['committed_runtime_clean']=not subprocess.check_output(['git','diff','HEAD','--','ssd'],cwd=ROOT,text=True)
 assert out['committed_runtime_clean']
 (HERE/'FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
 print('PASS',len(out['runs']),'final runs;',out['frozen_files'],'unchanged files')

if __name__=='__main__':main()
