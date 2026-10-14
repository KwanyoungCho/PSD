"""Matched input-preparation wall cost and byte-for-byte buffer parity."""
import argparse,json,os,sys,time,statistics
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers.batch_tree_inputs import BatchedArenaInputs
from mlsys_tree_systems import install

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    before=BatchedArenaInputs.replay;os.environ['DUET_REVIEW_NUMPY_INPUT']='1';install();after=BatchedArenaInputs.replay
    report=dict(status='running',gpu=torch.cuda.get_device_name(),cells=[])
    for b in (1,8):
      for g in (5,9,17):
        r=g*2;m=8;dev=torch.device('cuda');pages=4
        def arena():
            def z(*s,dtype=torch.int64):return torch.zeros(*s,device=dev,dtype=dtype)
            return NS(dev=dev,bs=256,R=r,max_blocks=m,canvas_extra_pages=1,total_cells=64,
                round_widths=[32,16,8,8],round_offsets=[0,32,48,56],in_root_tok=z(r),in_root_piv=z(r,dtype=torch.float32),
                in_glue=z(r,g,dtype=torch.uint8),in_rope_base=z(r),in_glue_w=z(1),in_prefix_len=z(1),
                in_temps=z(r,dtype=torch.float32),in_slot=[z(w,dtype=torch.int32) for w in (32,16,8,8)],
                in_ctx_len=[z(1,dtype=torch.int32) for _ in range(4)],in_block_tables=z(1,m,dtype=torch.int32),
                wrappers={pages:[NS(_paged_kv_indices_buf=z(pages+1,dtype=torch.int32)) for _ in range(4)]})
        arenas=[arena() for _ in range(b)];ex=NS(first=arenas[0],executors=arenas)
        graph=BatchedArenaInputs(ex,pages,2048)
        rows=[];params=[]
        for i in range(b):
            n=g if i%2==0 else max(1,g//2)
            rows.append(dict(tokens=list(range(n)),parents=list(range(-1,n-2)),prefix=100+i*24,blocks=list(range(i*m,(i+1)*m))))
            params.append((torch.arange(n*2,device=dev),torch.ones(n*2,device=dev),torch.arange(n,device=dev).repeat_interleave(2)))
        temps=[.7]*b
        def snapshot():
            return [x.clone() for ar in arenas for x in [ar.in_root_tok,ar.in_root_piv,ar.in_glue,ar.in_rope_base,ar.in_block_tables,*ar.in_slot]]
        s1=before(graph,rows,params,temps);x=snapshot();s2=after(graph,rows,params,temps);y=snapshot()
        if s1!=s2 or not all(torch.equal(a,z) for a,z in zip(x,y)):raise AssertionError('Input preparation mismatch')
        funcs={'before':before,'after':after};samples={k:[] for k in funcs}
        for _ in range(5):
            for f in funcs.values():f(graph,rows,params,temps)
        for repeat in range(80):
            names=list(funcs)
            if repeat%2:names.reverse()
            for key in names:
                torch.cuda.synchronize();start=time.perf_counter();funcs[key](graph,rows,params,temps);torch.cuda.synchronize()
                samples[key].append((time.perf_counter()-start)*1000)
        cell=dict(batch=b,contexts=g,parity=True,samples_ms=samples,median_ms={k:statistics.median(v) for k,v in samples.items()})
        report['cells'].append(cell);print(b,g,cell['median_ms'],flush=True)
    report['status']='complete';args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
