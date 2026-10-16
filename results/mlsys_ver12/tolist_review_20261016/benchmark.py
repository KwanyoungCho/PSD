"""Shared before/after harness with CPU-only metadata timing audits."""
import json
import os
from pathlib import Path
import sys
import time

CHECKOUT=Path(os.environ['PAPER_CHECKOUT'])
sys.path.insert(0,str(CHECKOUT/'ssd/bench'))
from mlsys_ps_preparation import main
from mlsys_phase_compilation import install
install()
from ssd.engine.draft_runner import DraftRunner

update=DraftRunner._update_phase2_layout_inplace
state=DraftRunner._p2_step_state
prefill=DraftRunner.draft_async_prefill
exit=DraftRunner.exit


def log(r,kind,start,extra):
    if not hasattr(r,'_fanout_audit'):r._fanout_audit=[]
    r._fanout_audit.append(dict(kind=kind,step_id=r._request_step_id,
        cpu_ms=(time.perf_counter_ns()-start)/1e6,**extra))


def update_timed(r,*args,**kwargs):
    start=time.perf_counter_ns();out=update(r,*args,**kwargs)
    log(r,'fanout_layout',start,dict(host_list=out.fan_out_list is not None,
        position_count=out.position_count,width=out.MQ_LEN))
    return out


def state_timed(r,*args,**kwargs):
    start=time.perf_counter_ns();out=state(r,*args,**kwargs)
    log(r,'state_readback',start,{})
    return out


def flush(r):
    rows=getattr(r,'_fanout_audit',[])
    if rows:
        folder=Path(os.environ['SSD_PROFILE_DIR']);epoch=r._fanout_epoch
        (folder/f'fanout_audit_request_{epoch:04d}.json').write_text(json.dumps(dict(epoch=epoch,rows=rows))+'\n')
        r._fanout_audit=[]


def prefill_flush(r,*args,**kwargs):
    flush(r);r._fanout_epoch=getattr(r,'_fanout_epoch',-1)+1
    return prefill(r,*args,**kwargs)


def exit_flush(r,*args,**kwargs):
    flush(r);return exit(r,*args,**kwargs)


DraftRunner._update_phase2_layout_inplace=update_timed
DraftRunner._p2_step_state=state_timed
DraftRunner.draft_async_prefill=prefill_flush
DraftRunner.exit=exit_flush

if __name__=='__main__':main()
