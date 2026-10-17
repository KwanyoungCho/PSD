"""CPU-only mechanism check, separate from full-model latency claims."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from ssd.engine.helpers.p2_tree import parse_tree_ints, validate_tree_ints


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    rows=[];capacity=40
    for count in (3,12,39):
        packet=torch.zeros(3+4*capacity,dtype=torch.int64)
        packet[0]=packet[1]=count;packet[2]=1
        packet[3:3+count]=torch.arange(count)
        packet[3+capacity:3+capacity+count]=torch.arange(count)-1
        packet[3+3*capacity:3+3*capacity+count]=torch.arange(count)
        times={'tensor':[],'hostlist':[]}
        for _ in range(5):
            for mode in ('tensor','hostlist'):
                for i in range(20):validate_tree_ints(parse_tree_ints(packet if mode=='tensor' else packet.tolist(),capacity),capacity,32000)
                start=perf_counter()
                for i in range(1000):validate_tree_ints(parse_tree_ints(packet if mode=='tensor' else packet.tolist(),capacity),capacity,32000)
                times[mode].append((perf_counter()-start)*1000/1000)
        rows.append(dict(valid=count,tensor_ms=median(times['tensor']),hostlist_ms=median(times['hostlist']),repeats=times))
    args.output.write_text(json.dumps(dict(mechanism_only=True,rows=rows),indent=2)+'\n')
    print(rows)


if __name__=='__main__':main()
