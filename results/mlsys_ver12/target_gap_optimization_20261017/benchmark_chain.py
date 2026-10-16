"""Same harness for all arms; a signal can dump startup stacks safely."""
import faulthandler
import os
from pathlib import Path
import signal
import sys
faulthandler.register(signal.SIGUSR1,all_threads=True)
sys.path.insert(0,str(Path(os.environ['PAPER_CHECKOUT'])/'ssd/bench'))
from mlsys_ps_preparation import main
import threading
import traceback
from ssd.engine.llm_engine import LLMEngine

_initialize=LLMEngine.__init__
def initialize(self,*args,**kwargs):
    stop=threading.Event()
    def observe():
        path=Path(os.environ['SSD_PROFILE_DIR'])/'startup_stacks.log'
        while not stop.wait(15):
            with path.open('a') as stream:
                for ident,frame in sys._current_frames().items():
                    stream.write(f'\nTHREAD {ident}\n'+''.join(traceback.format_stack(frame)))
    thread=threading.Thread(target=observe,daemon=True);thread.start()
    try:return _initialize(self,*args,**kwargs)
    finally:stop.set();thread.join()
LLMEngine.__init__=initialize

import json
import ssd
_LLM=ssd.LLM

def exact_chain(*args,**kwargs):
    if (kwargs.get('max_num_seqs')!=1 or kwargs.get('duet_phase1_k')!=10
            or kwargs.get('duet_phase2_k')!=10
            or kwargs.get('duet_p1_tree_policy')!='off'
            or kwargs.get('duet_p2_tree_policy')!='off'):
        raise ValueError('Expected native B1 K10 chain')
    kwargs.update(duet_split_phase1_fan_out_list=[4]*7+[3]*4,
                  duet_p2_forward_width=40,duet_p2_root_count=40)
    llm=_LLM(*args,**kwargs);c=llm.config
    assert sum(c.duet_split_phase1_fan_out_list)==40
    assert c.duet_p2_seed_count==c.duet_proxy_total_budget==40
    assert not c.duet_tree_enabled
    (Path(os.environ['SSD_PROFILE_DIR'])/'RESOLVED_CONFIG.json').write_text(
        json.dumps(dict(ds_roots=40,ps_roots=40,k1=10,k2=10,ds_fan_out=[4]*7+[3]*4,kwargs=kwargs),indent=2)+'\n')
    return llm
ssd.LLM=exact_chain

if __name__=='__main__':main()
