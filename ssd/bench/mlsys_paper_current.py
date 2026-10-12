"""Same benchmark protocol, but explicitly select the current unified engine.

The current engine already has process isolation and AMD dtype compatibility;
only common cold-prefix lookup control is required here. No old-engine function
or experimental tree ladder is installed.
"""
import os,sys
from pathlib import Path
checkout=Path(__file__).resolve().parents[2]
os.environ['PAPER_CHECKOUT']=str(checkout)
sys.path.insert(0,str(checkout/'ssd'))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import mlsys_paper_hooks

def current_adapter(arm):
    from ssd.engine.block_manager import BlockManager
    old=BlockManager.allocate
    def allocate(self,seq):
        if os.getenv('PAPER_COLD_PREFIX','1')=='1':self.hash_to_block_id.clear()
        return old(self,seq)
    BlockManager.allocate=allocate

mlsys_paper_hooks.install=current_adapter
exec(compile(Path(__file__).with_name('mlsys_paper_ab.py').read_text(),
             str(Path(__file__).with_name('mlsys_paper_ab.py')),'exec'))
