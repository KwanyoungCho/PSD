"""Serialized supplementary experiments after the frozen validation grid."""
import json
from pathlib import Path
import time
import traceback
from campaign import HERE, config, run_one


def main():
    plan=json.loads((HERE/'plan.json').read_text())
    # Explicitly separate supplementary domains from the immutable primary plan.
    extensions=plan['extensions']+[config(exit=40,k1=2,k2=2),config(exit=56,k1=2,k2=2),config(fanout=1),config(fanout=2)]
    extplan=HERE/'extension_plan.json'
    if not extplan.exists():
        extplan.write_text(json.dumps(dict(configs=extensions,note='Separate shape/source and low-K boundary feasibility. Not part of primary 19-config regret benchmark.'),indent=2))
    extensions=json.loads(extplan.read_text())['configs']
    expected=[HERE/'runs/validation'/f'{c["tag"]}_s1913/complete.json' for c in plan['validation']]
    while not all(p.exists() for p in expected):time.sleep(10)
    from assess import summary
    summary()
    for cfg in extensions:
        try:run_one(cfg,'extensions',profile=True)
        except Exception:
            failure=HERE/'runs/extensions'/f'{cfg["tag"]}_s913/failure.txt'
            failure.parent.mkdir(parents=True,exist_ok=True)
            failure.write_text(traceback.format_exc());print(failure.read_text(),flush=True)
    print('Extensions completed; failures retained separately.',flush=True)


if __name__=='__main__':main()
