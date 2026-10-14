"""Reaggregate preserved capture audit with the newly requested population."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ssd/bench'))
from mlsys_measurement import capture_intervals,annotate,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('MEASUREMENT_REVIEW.json'))
    args=p.parse_args();folder=args.raw_root/'nova_latest/review_followup'
    result=dict(scope='Existing Llama3 B8 captured runs; no new GPU performance experiment.',
        policy='Drop complete steps overlapping observed preparation; never subtract overlap duration from total time while retaining tokens.',
        caveat='Old capture coverage is the labels present in the preserved runs; new harness additionally observes graph contexts and input/proxy/accept constructors.',
        cells={},packed_components={},model_configs={})
    for arm in ('wait','ar'):
        data=json.loads((folder/f'{arm}.json').read_text());captures=capture_intervals(folder/f'{arm}_captures')
        result['cells'][arm]={}
        for name in ('measured_first','measured_second'):
            groups=[copy.deepcopy(g) for g in data['groups'] if g['phase']==name]
            steps=[s for g in groups for s in g['steps']];outputs=[o for g in groups for o in g['outputs']]
            annotate(steps,captures);result['cells'][arm][name]=summarize(steps,8,outputs)
    for model,relative,cfg in [('llama2','target_components/llama2_target.json','/home/chokwans99/models_mlsys/layerskip-llama2-7b/config.json'),
                              ('llama3','component_llama3_4090_v2.json','/data/chokwans99/models/layerskip-llama3-8B/config.json')]:
        data=json.loads((args.raw_root/relative).read_text())
        result['packed_components'][model]=[c for c in data['cells'] if c['batch']==8 and c['shape'] in ('half_short','one_hit') and c['implementation'] in ('padded','fixed') and not c['stream']]
        config=json.loads(Path(cfg).read_text())
        result['model_configs'][model]={k:config[k] for k in ('hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','torch_dtype','vocab_size')}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for arm,cells in result['cells'].items():
        for phase,c in cells.items():
            print(arm,phase,'TPS',c['summary']['tps'],'n',c['summary']['steps'])
            print('batch',json.dumps(c['batch_metrics']))
            print('post-startup',json.dumps(c['batch_metrics_excluding_group_startup']))
            print('prep',json.dumps(c['preparation_full_batch']),json.dumps(c['preparation_all_decode']))


if __name__=='__main__':main()
