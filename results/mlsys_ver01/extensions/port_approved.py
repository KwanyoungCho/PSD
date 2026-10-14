"""Reproduce the reviewed systems port without replacing ver01 fixes.

Run once on f12d0f1 runtime. Hunk identities are recorded in PORT.json.
The source is a git object, never the other worktree's uncommitted files.
"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = 'feat/duet-mlsys-coverage'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True)


selected = {
    'config.py': [3, 4, 7],
    'engine/draft_runner.py': [2, 3, 4, 5, 6, 7, 8, 19, 21],
    'engine/model_runner.py': [0, 1, 3, 5, 6, 10],
    'engine/verifier.py': [2, 12],
    'engine/llm_engine.py': [0, 2, 3],
    'engine/speculator_async.py': None,
    'engine/helpers/p1_tree.py': None,
    'layers/attention.py': None,
    'layers/layernorm.py': None,
    'utils/misc.py': None,
}

manifest = dict(base=git('rev-parse', 'HEAD').strip(),
                source=git('rev-parse', SOURCE).strip(), hunks={})
for name, indices in selected.items():
    path = 'ssd/ssd/' + name
    chunks = re.split(r'(?m)(?=^@@ )', git('diff', '-U3', 'HEAD', SOURCE, '--', path))[1:]
    text = (ROOT / path).read_text()
    for i in range(len(chunks)) if indices is None else indices:
        lines = chunks[i].splitlines(keepends=True)[1:]
        old = ''.join(l[1:] for l in lines if l.startswith((' ', '-')))
        new = ''.join(l[1:] for l in lines if l.startswith((' ', '+')))
        if name == 'engine/model_runner.py' and i == 6:
            new = new.replace('        from ssd.engine.helpers.root_policy import options as root_options\n', '')
        if name == 'engine/model_runner.py' and i == 10:
            # Packed execution is a later isolated experiment, not approved default.
            start = new.index("        if os.environ.get('SSD_PACKED_TREE_VERIFY'")
            end = new.index('        from ssd.engine.helpers.batch_tree_forward', start)
            new = new[:start] + new[end:]
        assert text.count(old) == 1, (name, i, text.count(old))
        text = text.replace(old, new, 1)
    (ROOT / path).write_text(text)
    manifest['hunks'][name] = indices

helpers = ['batch_tree_common', 'batch_tree_draft', 'batch_tree_forward',
           'batch_tree_inputs', 'batch_tree_roots', 'batch_tree_sampling',
           'batch_tree_verify', 'batched_tree_executor', 'tree_fused_math',
           'tree_host_topology', 'tree_proxy_stream']
tests = ['test_model_pair_contract', 'test_batched_tree_executor',
         'test_batch_tree_serving', 'test_tree_fused_math']
for folder, names in [('ssd/ssd/engine/helpers', helpers), ('ssd/tests', tests)]:
    for name in names:
        path = f'{folder}/{name}.py'
        assert not (ROOT / path).exists(), path
        (ROOT / path).write_text(git('show', f'{SOURCE}:{path}'))

# Generator protocol and fused bookkeeping only: keep original q-path selection.
path = ROOT / 'ssd/ssd/engine/helpers/p2_tree_executor.py'
text = git('show', f'{SOURCE}:ssd/ssd/engine/helpers/p2_tree_executor.py')
text = text.replace('        from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy\n        self.expansion_policy = TreeExpansionPolicy(self)\n', '')
text = text.replace('conf_threshold=conf_threshold,\n                    all_frontier=self.expansion_policy.frontier)', 'conf_threshold=conf_threshold)')
text = text.replace("            if _global and self.expansion_policy.gain is not None:\n                fan = self.expansion_policy.fanout(ar, sel, sel_valid, remaining, F-f-1)\n            elif _global and os.getenv('SSD_TREE_FUSED_MATH','0')=='1':", "            if _global and os.getenv('SSD_TREE_FUSED_MATH','0')=='1':")
text = text.replace('lp = self.expansion_policy.priority(ar.logpri, par, raws)', 'lp = ar.logpri.gather(0, par) + rq.clamp_min(1e-9).log()')
text = text.replace('allow_greedy=True,', 'allow_greedy=getattr(self.cfg, "greedy_only", False),')
path.write_text(text)

# Keep the established miss-chain policy. Shallow/dynamic miss research is deferred.
path = ROOT / 'ssd/ssd/engine/helpers/batch_tree_draft.py'
text = path.read_text()
text = text.replace("miss_depth=int(os.getenv('SSD_DUET_MISS_K',str(default_k)))", 'miss_depth=default_k')
text = text.replace("miss_width=int(os.getenv('SSD_DUET_MISS_WIDTH','1'))", 'miss_width=1')
text = text.replace('bt[idx],None,\n                            k_override=miss_depth,branch_width=miss_width)', 'bt[idx],None)')
path.write_text(text)

# Do not remove ver01's admission contracts or emitted-token metrics.
# Replace only the unsafe historical process teardown.
path = ROOT / 'ssd/ssd/engine/llm_engine.py'
text = path.read_text()
start = text.index('        # 5) Kill resource tracker')
end = text.index('        # 6) Force-exit', start)
text = text[:start] + '        # Multiprocessing owns its own semaphores; preserve other engines.\n' + text[end:]
path.write_text(text)

(Path(__file__).parent / 'PORT.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
