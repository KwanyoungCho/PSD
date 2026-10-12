"""Generate separately reviewable patches against the actual paper commit.

These are artifacts, not an automatic merge or production-default change.
"""
import difflib,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'patches'
OUT.mkdir(exist_ok=True)
BASE='a82f7d24fb36827a9a81a3567f344dccb71f193e'

def original(path):
    return subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT,text=True)

def patch(name, files):
    chunks=[]
    for path,(before,after) in files.items():
        if before==after:continue
        chunks.append('diff --git a/'+path+' b/'+path+'\n')
        if not before:chunks.append('new file mode 100644\n')
        chunks.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
            fromfile='a/'+path if before else '/dev/null',tofile='b/'+path))
    (OUT/name).write_text(''.join(chunks))

# Direct version of the chain arm: CPU-known hints and the same two routines.
v='ssd/ssd/engine/verifier.py';before=original(v);after=before
anchor='''                valid_k=speculate_result.valid_k,
            )
            _mc("chain_accept", _mev_chain_accept)'''
replacement='''                valid_k=speculate_result.valid_k,
                all_stochastic=all(t > 0 for t in temps_target + temps_draft),
            )
            _mc("chain_accept", _mev_chain_accept)'''
if after.count(anchor)!=1:raise ValueError('Chain call-site drift')
after=after.replace(anchor,replacement)
utils_path='ssd/ssd/utils/verify.py'
old_utils=original(utils_path)
helper='''
def _materialize_verified(speculations, counts, recovery):
    rows = torch.cat((counts[:, None], recovery[:, None], speculations), dim=1).tolist()
    return [r[2:3 + r[0]] for r in rows], [r[1] for r in rows]

'''
new_utils=old_utils.replace('\ndef verify(',helper+'\ndef verify(',1)
new_utils=new_utils.replace('    valid_k: torch.Tensor | None = None,\n',
    '    valid_k: torch.Tensor | None = None,\n    all_stochastic: bool = False,\n',1)
new_utils=new_utils.replace('    device = logits_p.device\n','''    if all_stochastic and jit_speculate:
        from ssd.utils.verify_fast import verify_stochastic
        return verify_stochastic(logits_p, logits_q, speculations,
            temperatures_target, temperatures_draft, valid_k,
            sampler_x, async_fan_out)

    device = logits_p.device
''',1)
patch('01_chain_fast.patch',{
    v:(before,after),
    utils_path:(old_utils,new_utils),
    'ssd/ssd/utils/verify_fast.py':('',(ROOT/'ssd/ssd/utils/verify_fast.py').read_text()),
})

# Original B1 execution/commit remains; this optional candidate is isolated.
before=original(v);after=before
anchor='''        path, terminal = tree_verify_walk_tensor(
            ti, p_rows, q_probs, tt, _coin, _mult)'''
replacement='''        walk_fn = tree_verify_walk_tensor
        if os.environ.get("SSD_PAPER_TREE_ACCEPT_GRAPH", "0") == "1":
            from ssd.engine.helpers.paper_tree_accept import tree_verify_walk_captured
            walk_fn = tree_verify_walk_captured
        path, terminal = walk_fn(ti, p_rows, q_probs, tt, _coin, _mult)'''
if after.count(anchor)!=1:raise ValueError('Tree call-site drift')
after=after.replace(anchor,replacement)
source=(ROOT/'ssd/bench/mlsys_paper_tree.py').read_text().split('\ndef install():')[0]
source+='''
_graphs = {}

def tree_verify_walk_captured(tree, p, q, temp, coin, multinomial):
    from ssd.engine.helpers.p2_tree import tree_verify_walk_tensor
    n = int(tree['valid'])
    if n == 0:
        return tree_verify_walk_tensor(tree, p, q, temp, coin, multinomial)
    c = max(int(x) for x in tree['sib_order'][:n]) + 1
    key = n, p.shape[-1], c, p.device
    if key not in _graphs:
        _graphs[key] = PaperTreeLadder(*key[:3], key[3])
    return _graphs[key].walk(tree, p, q, temp, coin, multinomial)
'''
patch('02_tree_graph_candidate.patch',{v:(before,after),
    'ssd/ssd/engine/helpers/paper_tree_accept.py':('',source)})

# Four correctness repairs, no change to attention tree dispatch or scheduler B.
files={}
for path in ('ssd/ssd/engine/block_manager.py','ssd/ssd/engine/sequence.py'):
    files[path]=(original(path),(ROOT/path).read_text())
path='ssd/ssd/layers/attention.py';before=original(path);after=(ROOT/path).read_text()
after=after.replace('if context.active_wrappers is None and self.only_prefill_wrapper is not None:',
                    'if self.only_prefill_wrapper is not None:')
files[path]=(before,after)
path='ssd/ssd/engine/scheduler.py';before=original(path)
after=before.replace('prefix = block_manager.blocks[block_table[-2]].hash if len(block_table) > 1 else -1',
                    'prefix = block_manager.blocks[block_table[block_index - 1]].hash if block_index > 0 else -1')
after=after.replace('last_block = block_manager.blocks[block_table[-1]]','last_block = block_manager.blocks[block_table[block_index]]')
files[path]=(before,after)
patch('03_kv_correctness.patch',files)
print('\n'.join(str(p.relative_to(ROOT)) for p in OUT.glob('*.patch')))
