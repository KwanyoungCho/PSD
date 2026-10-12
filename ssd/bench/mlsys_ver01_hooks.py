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


def install(arm):
    if arm != 'base':
        raise ValueError('This review uses real checkout code, no algorithm hooks')
    install_common()
