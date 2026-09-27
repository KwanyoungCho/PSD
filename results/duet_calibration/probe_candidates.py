"""Cost of replaying candidate hyperparameters, on fresh full-DUET paths.

The score is pre-P1-dedup root coverage, not end-to-end P2 hit probability.
Do not use these timings as inference throughput: distribution taps are on.
"""
import json
import time
from campaign import HERE, config, run_one


def main():
    while not (HERE/'refinement_frozen.json').exists():time.sleep(10)
    ref=json.loads((HERE/'refinement_frozen.json').read_text())
    expected=[HERE/'runs/confirmation'/f'{c["tag"]}_s{s}/complete.json' for s in ref['seeds'] for c in ref['confirmation']]
    while not all(p.exists() for p in expected):time.sleep(10)
    plan=json.loads((HERE/'plan.json').read_text())
    cfg=config(k1=4,k2=2)
    for st in ['calibration','validation']:
        run_one(cfg,'probe_'+st,profile=False,seed=4913 if st=='calibration' else 5913,
                dataset_info=plan['datasets'][st],probe_layers=[24,40,48,56,64,72,79])
    from replay_candidates import evaluate_campaign
    evaluate_campaign()


if __name__=='__main__':main()
