"""Per-request distribution files for a verifier recreated by each generate()."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'))
sys.path.insert(0,str(ROOT/'ssd/bench'))


def main():
    import bench
    original=bench.run_benchmark
    manifest=Path(os.environ['SSD_DUET_PROBE_OUT'])
    folder=manifest.with_suffix('');folder.mkdir(exist_ok=True)

    def run(args,llm,prompts,sampling_params):
        records=[];outputs=[];metas=[]
        def one(index,prompt,params):
            name=folder/('warmup.json' if index<0 else f'prompt_{index:03d}.json')
            os.environ['SSD_DUET_PROBE_OUT']=str(name)
            result=original(args,llm,[prompt],[params])
            probe=getattr(getattr(llm,'verifier',None),'_exit_probe',None)
            if probe is None:raise ValueError('No distribution observer after generate')
            # Engine exit flushes only the latest verifier; flush before the
            # next request replaces it, including requests below flush_steps.
            probe.dump()
            meta=json.loads(name.read_text())
            # The low-level observer is fresh per request, so global step IDs
            # need not be unique across different chunk files.
            for chunk in meta['chunks']:
                chunk['file']=str((name.parent/chunk['file']).relative_to(manifest.parent))
            metas.append(meta)
            return result
        start=time.perf_counter();one(-1,prompts[0],sampling_params[0]);warm=time.perf_counter()-start
        for index,(prompt,params) in enumerate(zip(prompts,sampling_params)):
            bench.reset_metrics();out,elapsed,metrics=one(index,prompt,params)
            rec=dict(prompt_index=index,prompt_tokens=len(prompt),prompt_token_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                     output_tokens=sum(len(x['token_ids']) for x in out),wall_s=elapsed,metrics=json.loads(json.dumps(metrics)))
            if rec['output_tokens']!=args.output_len:raise ValueError('Unexpected output length')
            records.append(rec);outputs.extend(out)
            payload=dict(schema='duet_calibration_run_v1',args=vars(args),candidate_policy='legacy',
                         warmup_wall_s=warm,profile=os.environ.get('SSD_PROFILE_DUET','0'),records=records)
            dst=Path(os.environ['DUET_CAL_OUT']);tmp=dst.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload));tmp.replace(dst)
        combined={k:metas[0][k] for k in ['schema','layers','exit_layer','temperature','stride','sampling','dtype']}
        combined.update(n_steps=sum(m['n_steps'] for m in metas),n_samples=sum(m['n_samples'] for m in metas),
                        n_rows=sum(m['n_rows'] for m in metas),seq_steps={k:v for m in metas for k,v in m['seq_steps'].items()},
                        chunks=[chunk for m in metas for chunk in m['chunks']],
                        max_prob_sum_error=max(m['max_prob_sum_error'] for m in metas),
                        collection='Separate output filename per generate(); aggregated after all requests. Warmup retained for explicit exclusion.')
        if len(combined['seq_steps'])!=len(prompts)+1:raise ValueError('Missing or repeated request snapshots')
        manifest.write_text(json.dumps(combined,indent=2))
        return outputs,sum(r['wall_s'] for r in records),records[-1]['metrics']
    bench.run_benchmark=run
    bench.main()


if __name__=='__main__':main()
