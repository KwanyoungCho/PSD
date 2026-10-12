"""Explicit benchmark adapters for an actual a82f7d2 engine checkout."""
import importlib.util
import os
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def load(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_common():
    """No acceptance, forward, draft or KV algorithm changes."""
    import inspect
    import ssd.engine.model_runner as mr
    import ssd.engine.llm_engine as eng
    # Old code uses a global SHM name and deletes unrelated semaphores.
    # Isolate only resource naming and shutdown; apply identically in all arms.
    original_shm = mr.SharedMemory
    def shared_memory(name, **kwargs):
        return original_shm(name=os.environ['SSD_SHM_NAME'] if name == 'ssd' else name, **kwargs)
    mr.SharedMemory = shared_memory
    source = textwrap.dedent(inspect.getsource(eng.LLMEngine.exit))
    start, end = source.index('    # 5) Kill resource tracker'), source.index('    # 6) Force-exit')
    source = source[:start] + source[end:]
    scope = {}
    exec(compile(source, '<paper-benchmark-safe-shutdown>', 'exec'), eng.__dict__, scope)
    eng.LLMEngine.exit = scope['exit']
    from ssd.config import Config
    old_init = Config.__post_init__
    def post_init(self):
        old_init(self)
        if self.speculate:
            # Parent and child must allocate the same wire logits dtype.
            self.draft_hf_config.torch_dtype = self.hf_config.torch_dtype
    Config.__post_init__ = post_init
    from ssd.engine.draft_runner import DraftRunner
    old_draft_config = DraftRunner.create_draft_config
    def draft_config(cls, cfg):
        result = old_draft_config(cfg)
        result.hf_config.torch_dtype = cfg.hf_config.torch_dtype
        result.draft_hf_config.torch_dtype = cfg.hf_config.torch_dtype
        return result
    DraftRunner.create_draft_config = classmethod(draft_config)

    from ssd.engine.block_manager import BlockManager
    original_allocate = BlockManager.allocate
    def allocate(self, seq):
        if os.getenv('PAPER_COLD_PREFIX', '1') == '1':
            self.hash_to_block_id.clear()
        return original_allocate(self, seq)
    BlockManager.allocate = allocate


def install_chain():
    import ssd.engine.verifier as vm
    import ssd.utils.verify as original_module
    current = load('paper_current_verify', HERE.parent/'ssd/utils/verify.py')
    load('ssd.utils.verify_fast', HERE.parent/'ssd/utils/verify_fast.py')
    original_module._materialize_verified = current._materialize_verified
    original_outer = vm.Verifier.verify
    metadata = {}
    def outer(self, seqs, *args, **kwargs):
        tt = [s.temperature for s in seqs]
        dt = [s.temperature if s.draft_temperature is None else s.draft_temperature for s in seqs]
        metadata['greedy'] = all(t == 0 for t in tt+dt)
        metadata['stochastic'] = all(t > 0 for t in tt+dt)
        return original_outer(self, seqs, *args, **kwargs)
    def inner(*args, **kwargs):
        return current.verify(*args, **kwargs, all_greedy=metadata['greedy'],
                              all_stochastic=metadata['stochastic'])
    vm.Verifier.verify = outer
    vm.verify = inner


def install_kv():
    """Only four correctness changes, not glue/promotion or new batching."""
    import ssd.engine.block_manager as bm
    import ssd.engine.sequence as seq
    import ssd.engine.scheduler as sched
    import ssd.layers.attention as attn
    # Load revised method definitions while retaining original class identity.
    new_bm = load('paper_new_blocks', HERE.parent/'ssd/engine/block_manager.py')
    new_seq = load('paper_new_sequence', HERE.parent/'ssd/engine/sequence.py')
    new_seq.SequenceStatus = seq.SequenceStatus
    new_sched = load('paper_new_scheduler', HERE.parent/'ssd/engine/scheduler.py')
    new_attn = load('paper_new_attention', HERE.parent/'ssd/layers/attention.py')
    old_allocate = bm.BlockManager.allocate
    def allocate(self, s):
        old_allocate(self, s)
        attr = 'num_draft_cached_tokens' if self.is_draft else 'num_cached_tokens'
        setattr(s, attr, min(getattr(s, attr), s.num_tokens-1))
    bm.BlockManager.allocate = allocate
    # Sequence has a normal __dict__; serialization list is an explicit contract.
    seq.Sequence._ATTRIBUTES = new_seq.Sequence._ATTRIBUTES
    seq.Sequence.__init__ = new_seq.Sequence.__init__
    seq.Sequence.__setstate__ = new_seq.Sequence.__setstate__
    seq.Sequence.num_completion_tokens = new_seq.Sequence.num_completion_tokens
    seq.Sequence.completion_token_ids = new_seq.Sequence.completion_token_ids
    sched.Scheduler._finalize_block = new_sched.Scheduler._finalize_block
    # Only the prefix-cache branch is replaced in an otherwise original forward.
    import inspect
    newer = textwrap.dedent(inspect.getsource(new_attn.Attention.forward))
    newer = newer.replace('if context.active_wrappers is None and self.only_prefill_wrapper is not None:',
                          'if self.only_prefill_wrapper is not None:')
    scope = {}
    exec(compile(newer, '<paper-paged-prefix-fix>', 'exec'), attn.__dict__, scope)
    attn.Attention.forward = scope['forward']


def install(arm):
    install_common()
    if arm in ('chain', 'both'): install_chain()
    if arm in ('tree', 'both'):
        from mlsys_paper_tree import install as tree_install
        tree_install()
    if arm == 'kv': install_kv()
