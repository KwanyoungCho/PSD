"""Finite-state negative control using current helper bodies (CPU only)."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parents[3]
from proposal_audit_math import production_functions, bias_audit
prod=production_functions()
r=bias_audit(prod['rerank_tree_indices'],prod['tree_verify_walk_tensor'])
r['old_negative_control_tv']=r.pop('tv')
r['generation_order_prefix_tv']=sum(abs(a-b) for a,b in zip(r['target'],r['first_output']['fixed_prefix']))/2
r['production_files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'ssd/ssd/engine/draft_runner.py',ROOT/'ssd/ssd/engine/helpers/p2_tree.py']}
Path(__file__).with_suffix('.json').write_text(json.dumps(r,indent=2))
print({k:r[k] for k in ['states','target','first_output','old_negative_control_tv','generation_order_prefix_tv']})
