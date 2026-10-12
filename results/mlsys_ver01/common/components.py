import importlib.util,json
from pathlib import Path
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paper_components',R.parent/'analyze_components.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.R=R
arms={}
for raw in sorted((R/'profiles').glob('profile_*.json')):
    if json.loads(raw.read_text()).get('status')=='complete':arms[raw.stem]=mod.analyze(raw)
(R/'COMPONENTS.json').write_text(json.dumps(dict(scope='Separate 24-input profiles. Exclude first19, cap/clip, capture+next; CUDA spans include launch gaps; phase totals include preparation. Do not sum overlapping medians.',arms=arms),indent=2))
for name,a in arms.items():print(name,{k:round(v['median'],4) for k,v in a['summary'].items() if '/' not in k})
