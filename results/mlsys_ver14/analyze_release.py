"""Direct ver1.3/ver1.4 comparison: exact parity gate, matched steps, UID CI."""
from pathlib import Path
import json
import analysis_common as a
HERE=Path(__file__).resolve().parent
ROOT=HERE/'raw/runs' if (HERE/'raw/runs').exists() else Path('/data/chokwans99/duet-ver14-release-20261017/runs')
a.RAW=ROOT

def main():
    pairs=[('v13_pair0_a','v14_pair0_b'),('v13_pair1_a','v14_pair1_b')]
    result={}
    ready=[p for p in pairs if all(a.load(t) for t in p)]
    if ready:result['tree_T07']=a.compare(ready)
    for name,pair in [('chain_smoke',('v13_chain','v14_chain')),('tree_greedy_smoke',('v13_greedy','v14_greedy'))]:
        if all(a.load(t) for t in pair):
            b,c=map(a.load,pair);result[name]=a.parity(b,c)
            assert result[name]['outputs_equal'] and result[name]['events_equal'],name
    (HERE/'DIRECT_COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,r in result.items():print(name,r.get('status',r),r.get('reduction_pct'),r.get('ci95_pct'))

if __name__=='__main__':main()
