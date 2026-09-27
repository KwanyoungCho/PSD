"""Experiment adapter: per-prompt records, unchanged production verifier/sampler."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'ssd'))


def main():
    policy = os.environ.get('DUET_CAL_CANDIDATE', 'legacy')
    if policy != 'legacy':
        sys.path.insert(0, str(ROOT / 'results/residial_dist/training_free'))
        from runtime_policy import make_policy
        import ssd.engine.helpers.p2_tree as p2
        p2.chain_proxy_candidates_fixed = make_policy(
            policy, float(os.environ.get('DUET_CAL_TEMP', '.7')), p2.pack_piv)
    sys.path.insert(0, str(ROOT / 'ssd/bench'))
    import bench
    original = bench.run_benchmark

    def run(args, llm, prompts, sampling_params):
        records, outputs = [], []
        # One excluded warmup request; every measured request retains raw metrics.
        warm_start = time.perf_counter()
        original(args, llm, prompts[:1], sampling_params[:1])
        warm_s = time.perf_counter() - warm_start
        for index, (prompt, params) in enumerate(zip(prompts, sampling_params)):
            bench.reset_metrics()
            out, elapsed, metrics = original(args, llm, [prompt], [params])
            snapshot = json.loads(json.dumps(metrics))
            record = dict(prompt_index=index, prompt_tokens=len(prompt),
                          prompt_token_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                          output_tokens=sum(len(x['token_ids']) for x in out),
                          wall_s=elapsed, metrics=snapshot)
            if record['output_tokens'] != args.output_len:
                raise RuntimeError('Unexpected output length')
            records.append(record)
            outputs.extend(out)
            payload = dict(schema='duet_calibration_run_v1', args=vars(args),
                           candidate_policy=policy, warmup_wall_s=warm_s,
                           profile=os.environ.get('SSD_PROFILE_DUET', '0'),
                           records=records)
            dst = Path(os.environ['DUET_CAL_OUT'])
            tmp = dst.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload))
            tmp.replace(dst)
        return outputs, sum(x['wall_s'] for x in records), records[-1]['metrics']
    bench.run_benchmark = run
    bench.main()


if __name__ == '__main__':
    main()
