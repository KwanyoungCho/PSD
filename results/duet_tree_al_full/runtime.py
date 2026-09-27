"""Research process hooks: reserve graph memory and optionally change tree score."""
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'));sys.path.insert(0,str(ROOT/'results/duet_tree_analysis'))


def install():
    from ssd.engine.draft_runner import DraftRunner
    if not getattr(DraftRunner,'_full_al_memory_hook',False):
        original=DraftRunner.create_draft_config.__func__
        def create(cls,cfg):
            draft=original(cls,cfg)
            draft.gpu_memory_utilization=float(os.environ['DUET_FULL_DRAFT_KV_FRACTION'])
            return draft
        DraftRunner.create_draft_config=classmethod(create)
        DraftRunner._full_al_memory_hook=True
    if os.environ['DUET_TREE_SCORE_MODE']!='q_path':
        from score_hook import install as score_install
        score_install()
