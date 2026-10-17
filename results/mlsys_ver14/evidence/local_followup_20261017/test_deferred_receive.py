"""Two-GPU dependency test with a deliberately delayed q sender.

Run alone on an idle authorized GPU pair, never alongside timing runs.
The target must execute independent work before q arrives, then read q
correctly on both a proxy side stream and its default acceptance stream.
"""
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from types import SimpleNamespace


def worker(rank,output):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method='tcp://127.0.0.1:29887',rank=rank,world_size=2)
    from ssd.engine.speculator_async import SpeculatorAsync
    q=torch.zeros(4096,device=f'cuda:{rank}')
    x=torch.ones((128,128),device=f'cuda:{rank}')
    y=torch.empty_like(x)
    torch.mm(x,x,out=y)
    compute_done=torch.cuda.Event()
    # Initialize P2P communication before the delay/readiness assertion.
    if rank==1:dist.send(q,dst=0)
    else:dist.recv(q,src=1)
    dist.barrier();torch.cuda.synchronize()
    if rank==1:
        time.sleep(.4)
        q.fill_(7)
        dist.send(q,dst=0)
    else:
        start=time.perf_counter()
        work=dist.irecv(q,src=1)
        owner=SimpleNamespace(_pending_parent_q=work)
        torch.mm(x,x,out=y)
        compute_done.record()
        # Device-wide synchronize would also wait for the intentionally
        # pending receive. Wait only for independent default-stream work.
        compute_done.synchronize()
        independent_ms=1000*(time.perf_counter()-start)
        assert independent_ms<300,independent_ms
        side=torch.cuda.Stream()
        with torch.cuda.stream(side):
            SpeculatorAsync._wait_parent_q_ready(owner)
            side_value=q.sum()
        # Retaining Work permits a second consuming stream to establish its
        # own dependency; no use of pending q is allowed before either wait.
        SpeculatorAsync._wait_parent_q_ready(owner)
        default_value=q.sum()
        side.synchronize()
        assert float(side_value)==float(default_value)==4096*7
        assert float(y[0,0])==128
        Path(output).write_text(json.dumps(dict(status='pass',independent_before_receive_ms=independent_ms,
            delayed_sender_seconds=.4,side_stream_sum=float(side_value),default_stream_sum=float(default_value)),indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    mp.spawn(worker,args=(os.environ['DEFERRED_TEST_OUTPUT'],),nprocs=2,join=True)
