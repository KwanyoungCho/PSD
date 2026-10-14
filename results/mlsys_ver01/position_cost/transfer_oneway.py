"""NCCL one-way payload microbench on the same pair, ready receiver.

Scalar-q is a transport-only hypothetical, not a complete lossless SD path.
Uses the same dist.send/dist.recv API as the paper. Excludes initialization,
warmup, CPU control barrier, captures and tensor construction.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,port,args):
    torch.set_num_threads(1);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank,
                            timeout=timedelta(seconds=180),device_id=torch.device('cuda',rank))
    control=dist.new_group([0,1],backend='gloo')
    result=dict(rank=rank,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        peer_access=torch.cuda.can_device_access_peer(0,1),env={k:v for k,v in os.environ.items() if k.startswith('NCCL_') or k=='CUDA_VISIBLE_DEVICES'},cells=[])
    for vocab in (32000,128256):
        for b,k in ((1,4),(1,12),(8,4),(8,12)):
            rows=b*k;dev=torch.device('cuda',rank)
            shapes={'full_q':(rows*vocab,torch.float16),'scalar_q':(rows,torch.float32),
                    'one_q_row':(b*vocab,torch.float16),'root_wire':(2*b*15,torch.int64)}
            tensors={name:torch.full((n,),.5 if dt!=torch.int64 else 7,device=dev,dtype=dt) for name,(n,dt) in shapes.items()}
            samples={name:dict(gpu_us=[],wall_us=[]) for name in tensors}
            for warm in range(8):
                for tensor in tensors.values():
                    if rank:dist.send(tensor,dst=0)
                    else:dist.recv(tensor,src=1)
                torch.cuda.synchronize()
            order=list(tensors);rng=random.Random(155+rows)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            for rep in range(args['repeats']):
                rng.shuffle(order)
                for name in order:
                    torch.cuda.synchronize();dist.barrier(group=control)
                    t0=time.perf_counter_ns();start.record()
                    if rank:dist.send(tensors[name],dst=0)
                    else:dist.recv(tensors[name],src=1)
                    end.record();end.synchronize()
                    elapsed=(time.perf_counter_ns()-t0)/1000
                    samples[name]['gpu_us'].append(start.elapsed_time(end)*1000)
                    samples[name]['wall_us'].append(elapsed)
            for name,tensor in tensors.items():
                assert bool((tensor==(.5 if tensor.dtype!=torch.int64 else 7)).all())
                stats={kind:dict(median=statistics.median(x),p10=sorted(x)[len(x)//10],p90=sorted(x)[len(x)*9//10],samples=x) for kind,x in samples[name].items()}
                result['cells'].append(dict(vocab=vocab,batch=b,positions=k,payload=name,bytes=tensor.numel()*tensor.element_size(),timing=stats))
            if rank==0:print('transfer',vocab,b,k,'full/scalar median us',*[round(statistics.median(samples[n]['gpu_us']),2) for n in ('full_q','scalar_q')],flush=True)
    out=Path(args['output']);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_name(out.stem+f'.rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier(group=control);dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--repeats',type=int,default=80);args=ap.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,vars(args)),nprocs=2,join=True)
    result=dict(scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),args=vars(args),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        ranks=[json.loads(out.with_name(out.stem+f'.rank{r}.json').read_text()) for r in (0,1)],status='complete')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
