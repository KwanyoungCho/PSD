"""Within-server complete five-method comparisons, two independent seeds."""
import importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('p',HERE/'make_plan.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
def select(jobs,names,cap=32):
 out=[]
 for j in jobs:
  if j['name'] not in names:continue
  j['name']+='_m'+str(cap);j['env']['PAPER_ARM']=j['name'];out.append(j)
 return out
# Same architecture/server contains all five methods at each main seed.
# Cap64 is an explicit C8 diagnosis, never pooled with the cap32 main table.
plans={
 'local_45_paper':select(p.plan(),['chain','last_c3','frontier_c3']),
 'local_67_paper':select(p.plan(cpus='24-31,56-63'),['last_c8','frontier_c8'])+select(p.plan(cap=64,cpus='24-31,56-63'),['last_c8','frontier_c8'],64),
 'nova_23_paper':select(p.plan(remote=True,seed=26101,cpus='0-5,32-37'),['chain','last_c3','frontier_c3']),
 'nova_45_paper':select(p.plan(remote=True,seed=26101,cpus='6-10,38-42'),['last_c8','frontier_c8']),
 'nova_67_paper':select(p.plan(remote=True,seed=26101,cap=64,cpus='11-15,43-47'),['last_c8','frontier_c8'],64),
}
for name,jobs in plans.items():(HERE/(name+'.json')).write_text(json.dumps(jobs,indent=2)+'\n')
