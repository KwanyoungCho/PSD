"""Freeze phase-only prospective gain curves using the original 8 prompts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from gain import third_gain

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'duet_tree_analysis'
sys.path.insert(0, str(HERE.parent / 'duet_tree_posthoc'))
from features import extract, groups


def main():
    dest = HERE / 'gain_calibration.json'
    if dest.exists():
        raise FileExistsError('Frozen calibration already exists; do not overwrite')
    manifest = json.loads((OLD / 'probe/manifest.json').read_text())
    rows = []; hashes = {}; start = time.time()
    for meta in manifest['snapshots']:
        if manifest['plan']['mapping'][meta['prompt']]['split'] != 'calibration':
            continue
        path = OLD / 'probe' / meta['file']
        hashes[meta['file']] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            t = extract(**dict(z))
            for parent, children in groups(t['par'], t['sib']).items():
                if parent < 0:
                    continue
                result = third_gain(z['p'][parent+1], z['q'][children[0]],
                                    seed=9022000+len(rows), head=32, samples=512)
                rows.append(dict(file=meta['file'], prompt=meta['prompt'],
                    phase=meta['phase'], parent=parent, weight=t['reach'][parent], **result))
    curves = {}; integration = {}; uncertainty = {}
    for phase in (1, 2):
        rr = [r for r in rows if r['phase'] == phase]
        w = np.array([r['weight'] for r in rr]); w /= w.sum()
        raw = w @ np.array([r['g'] for r in rr])
        deployed = np.minimum(1., np.maximum.accumulate(raw))
        curves[str(phase)] = deployed.tolist()
        integration[str(phase)] = dict(
            raw=raw.tolist(), adjustment=(deployed-raw).tolist(),
            se=float(np.sqrt(np.sum((w*np.array([r['se'] for r in rr]))**2))),
            deterministic_bound=(w @ np.array([r['bound'] for r in rr])).tolist(),
            contexts=len(rr), reach_weight=float(sum(r['weight'] for r in rr)))
        # Leave-one-prompt-out stability, NOT a large-data generalization CI.
        leave = []
        for prompt in sorted({r['prompt'] for r in rows}):
            keep = [r for r in rr if r['prompt'] != prompt]
            ww = np.array([r['weight'] for r in keep]); ww /= ww.sum()
            leave.append((ww @ np.array([r['g'] for r in keep])).tolist())
        uncertainty[str(phase)] = dict(leave_one_prompt_out=leave,
            min=np.min(leave, axis=0).tolist(), max=np.max(leave, axis=0).tolist())
    old = json.loads((HERE.parent / 'duet_tree_posthoc/fanout_analysis.json').read_text())
    for phase, curve in curves.items():
        if not np.allclose(curve[1:3], old['original_8_prompt_gain_curves'][phase], atol=1e-12):
            raise ValueError('Original exact g1/g2 calibration does not reproduce')
    result = dict(schema='duet_gain_c3_v1', source_prompts=8, source_trees=len(hashes),
        contexts=len(rows), curves=curves, numerical_integration=integration,
        calibration_stability=uncertainty, source_sha256=hashes,
        source_manifest_sha256=hashlib.sha256((OLD/'probe/manifest.json').read_bytes()).hexdigest(),
        method='Exact g1/g2; exact first-rejection head32 plus IID tail512, conditional analytic g2.',
        root_contexts_excluded=True, neural_training=False, full_dataset_fit=False,
        wall_s=time.time()-start)
    (HERE/'gain_calibration_contexts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
