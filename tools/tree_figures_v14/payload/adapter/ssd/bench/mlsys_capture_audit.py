"""Record CPU intervals of rare graph construction, without CUDA profiling.

No GPU events, synchronization or extra random draws are introduced. Existing
capture labels are enabled by SSD_PROFILE_DUET_DETAIL=1. Per-step timestamps
in the review harness let the parent correlate draft-process captures too.
"""
import json
import os
from pathlib import Path
import sys
import time
import functools
import importlib


def install():
    folder=os.getenv('DUET_REVIEW_CAPTURE_AUDIT')
    if not folder:return
    from ssd.engine.helpers import cudagraph_helpers as h
    if h.PROFILE_DUET:return
    if getattr(h,'_measurement_capture_audit_installed',False):return
    h._measurement_capture_audit_installed=True
    old_record,old_close=h.duet_record,h.duet_close
    path=Path(folder);path.mkdir(parents=True,exist_ok=True)
    def write(row):
        with (path/f'capture_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def record(label,parent=None):
        if 'capture' in label:
            row={'capture_audit':True,'label':label,'start_ns':time.monotonic_ns(),'pid':os.getpid()}
            write(row)
            return row
        return old_record(label,parent)
    def close(label,handle):
        if isinstance(handle,dict) and handle.get('capture_audit'):
            row=dict(handle,end_ns=time.monotonic_ns())
            write(row)
            return
        return old_close(label,handle)
    # Some helper modules were loaded by package import before this hook.
    for name,module in list(sys.modules.items()):
        if name.startswith('ssd.') and module is not None:
            if vars(module).get('duet_record') is old_record:setattr(module,'duet_record',record)
            if vars(module).get('duet_close') is old_close:setattr(module,'duet_close',close)
    h.duet_record,h.duet_close=record,close
    # These constructors also warm kernels/build input graphs. The existing
    # target/P1/P2 capture labels do not cover every constructor's setup.
    for mod,name in [('batch_tree_inputs','BatchedArenaInputs'),
                     ('batch_tree_roots','BatchedP1Roots'),
                     ('batch_tree_sampling','BatchedTreeProxy'),
                     ('batch_tree_sampling','BatchedTreeAccept')]:
        cls=getattr(importlib.import_module('ssd.engine.helpers.'+mod),name)
        original=cls.__init__
        def wrap(fn,label):
            @functools.wraps(fn)
            def init(self,*args,**kwargs):
                handle=record(label)
                try:return fn(self,*args,**kwargs)
                finally:close(label,handle)
            return init
        cls.__init__=wrap(original,name+'_setup_capture')
    forest_cls=importlib.import_module('ssd.engine.helpers.batch_tree_draft').BatchedDuetDraft
    original_forest=forest_cls._forest
    @functools.wraps(original_forest)
    def forest(self,phase,contexts,batch,pages):
        ex=self.executors.get((phase,contexts,batch))
        cold=(phase,contexts) not in self.forests or ex is None or (
            pages is not None and pages not in ex.wrappers)
        if not cold:return original_forest(self,phase,contexts,batch,pages)
        handle=record('draft_forest_setup_capture')
        try:return original_forest(self,phase,contexts,batch,pages)
        finally:close('draft_forest_setup_capture',handle)
    forest_cls._forest=forest
    # Catch other graph captures without synchronizing or changing RNG.
    import torch
    original_enter,original_exit=torch.cuda.graph.__enter__,torch.cuda.graph.__exit__
    def enter(self):
        self._measurement_capture_handle=record('cuda_graph_capture')
        try:return original_enter(self)
        except BaseException:
            close('cuda_graph_capture',self._measurement_capture_handle)
            raise
    def leave(self,*args):
        try:return original_exit(self,*args)
        finally:close('cuda_graph_capture',self._measurement_capture_handle)
    torch.cuda.graph.__enter__,torch.cuda.graph.__exit__=enter,leave
