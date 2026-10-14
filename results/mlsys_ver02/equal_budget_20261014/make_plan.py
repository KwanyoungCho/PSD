"""Frozen same-model, within-server comparisons. No TPS-based parameter search."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
REMOTE='/home/chokwans99/duet_equal_budget_20261014'
ARMS=[('chain',1,'last'),('last_c3',3,'last'),('frontier_c3',3,'frontier'),('last_c8',8,'last'),('frontier_c8',8,'frontier')]
def plan(remote=False,seed=16101,cap=32,smoke=False,cpus='16-23,48-55',batch=1):
 checkout=REMOTE if remote else str(ROOT)
 models='/home/chokwans99/equal_budget_models' if remote else '/home/chokwans99/models_mlsys'
 arms=[('last_c3_smoke',3,'last'),('frontier_c8_smoke',8,'frontier')] if smoke else ARMS
 if seed==26101 and not smoke: arms=list(reversed(arms))
 jobs=[]
 for name,c,scope in arms:
  args=['--mode','tree','--target',models+'/layerskip-llama2-7b','--draft',models+'/AMD-Llama-135m',
   '--prompts',checkout+'/results/mlsys_ver02/equal_budget_20261014/SPEC_BENCH_FULL.json',
   '--limit',str(16 if smoke else 560),'--warm-prompts',str(2 if smoke else 2),
   '--output-tokens',str(32 if smoke else 1024),'--input-cap','0','--passes','1',
   '--batch',str(batch),'--temperature','.7','--k1','8','--k2','8','--exit-layer','21',
   '--p1-nodes',str(64*c),'--p2-nodes',str(64*c),
   '--p1-verify',str(cap),'--p2-verify',str(cap),'--roots','1','--siblings',str(c),'--seed',str(seed)]
  env={'PAPER_CHECKOUT':checkout,'PAPER_CPUSET':cpus,'PAPER_COLD_PREFIX':'1',
       'PAPER_ARM':name,'DUET_EQUAL_ROOTS':'8','DUET_EQUAL_SELECTION':scope,
       'SSD_TREE_EXPANSION_POLICY':'q_path','SSD_TREE_SHAPE_METRICS':'1',
       'SSD_TREE_EXEC_WARMUP':'1,2,4,8','SSD_PROFILE_DUET':'1' if smoke else '0',
       'SSD_PROFILE_DUET_MAX_EVENTS':'12000','SSD_DUET_EXIT_REPLICA':'0',
       'SSD_BATCH_TREE_PROXY_STREAM':'0'}
  jobs.append(dict(name=name,bench='mlsys_equal_budget.py',args=args,env=env,draft_seed=231+(seed-16101)))
 return jobs
PLANS={
 'paper_l2_SMOKE':plan(smoke=True),
 'paper_local_45_m32_s16101':plan(),
 'paper_local_67_m32_s26101':plan(seed=26101,cpus='24-31,56-63'),
 'paper_nova_23_SMOKE':plan(remote=True,smoke=True,cpus='0-5,32-37'),
 'paper_nova_23_m32_s16101':plan(remote=True,cpus='0-5,32-37'),
 'paper_nova_45_m64_s16101':plan(remote=True,cap=64,cpus='6-10,38-42'),
 'paper_nova_67_m64_s26101':plan(remote=True,cap=64,seed=26101,cpus='11-15,43-47'),
}
for name,jobs in PLANS.items():
 (HERE/(name+'.json')).write_text(json.dumps(jobs,indent=2)+'\n')
