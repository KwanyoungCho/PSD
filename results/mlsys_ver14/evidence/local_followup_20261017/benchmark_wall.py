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


# Independent CPU wall time around the complete engine call. No CUDA event,
# synchronization or runtime algorithm patch. Same observer for all final arms.
import json
from time import perf_counter
from ssd.engine.llm_engine import METRICS
_step = LLMEngine.step
_wall_steps = []
def measured_step(self, *args, **kwargs):
    count = len(METRICS.get('decode_steps', []))
    start = perf_counter()
    result = _step(self, *args, **kwargs)
    elapsed = perf_counter() - start
    steps = METRICS.get('decode_steps', [])
    if len(steps) == count + 1:
        row = steps[-1]
        events = METRICS.get('phase_events', [])[row['event_start']:row['event_end']]
        _wall_steps.append(dict(wall_seconds=elapsed, native_seconds=row['seconds'],
            source=[e.get('source') for e in events],
            accepted_len=[e.get('accepted_len') for e in events]))
    return result
LLMEngine.step = measured_step

if __name__=='__main__':
    try:main()
    finally:
        (Path(os.environ['SSD_PROFILE_DIR'])/'WALL_STEPS.json').write_text(json.dumps(_wall_steps)+'\n')
