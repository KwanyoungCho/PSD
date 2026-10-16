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

if __name__=='__main__':main()
