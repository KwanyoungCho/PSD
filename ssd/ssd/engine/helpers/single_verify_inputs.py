"""Persistent B1 verify inputs without rebuilding tokens through host lists."""
import torch
import triton
import triton.language as tl


@triton.jit
def _geometry(pages,positions,slots,length,cu, prefix,context_length,
              rows:tl.constexpr, block_size:tl.constexpr, BLOCK:tl.constexpr):
    i=tl.arange(0,BLOCK)
    pos=prefix+i
    page=tl.load(pages+pos//block_size,mask=i<rows,other=0)
    tl.store(positions+i,pos,mask=i<rows)
    tl.store(slots+i,page*block_size+pos%block_size,mask=i<rows)
    tl.store(length,context_length)
    tl.store(cu,0)
    tl.store(cu+1,rows)


class SingleVerifyInputs:
    def __init__(self,device,max_rows,max_pages,block_size):
        self.device=device
        self.max_rows=int(max_rows);self.max_pages=int(max_pages);self.block_size=int(block_size)
        self.ids=torch.empty(max_rows,dtype=torch.int64,device=device)
        self.positions=torch.empty_like(self.ids)
        self.slots=torch.empty(max_rows,dtype=torch.int32,device=device)
        self.length=torch.empty(1,dtype=torch.int32,device=device)
        self.pages=torch.empty((1,max_pages),dtype=torch.int32,device=device)
        self.cu=torch.empty(2,dtype=torch.int32,device=device)
        # Host slots may be reused only after their tiny H2D copies finish.
        self.host=[(torch.empty(max_rows,dtype=torch.int64,pin_memory=True),
                    torch.empty(max_pages,dtype=torch.int32,pin_memory=True)) for _ in range(2)]
        self.done=[torch.cuda.Event() for _ in range(2)]
        self.pending=[False,False];self.next_slot=0;self.page_key=None

    def prepare(self,seq,k,source_ids=None):
        rows=int(k)+1
        if not 1<int(rows)<=self.max_rows:
            raise ValueError('Verify rows exceed input capacity')
        total=int(seq.num_tokens);prefix=total-rows
        if prefix<0 or seq.num_cached_tokens!=prefix:
            raise AssertionError('Verify window misaligned')
        page_key=tuple(seq.block_table)
        if not page_key or len(page_key)>self.max_pages or (total-1)//self.block_size>=len(page_key):
            raise ValueError('Verify window lacks allocated pages')
        if any(p<0 for p in page_key[:(total+self.block_size-1)//self.block_size]):
            raise ValueError('Verify window references negative pages')
        page_changed=page_key!=self.page_key
        if source_ids is not None:
            if source_ids.device!=self.device or source_ids.dtype!=torch.int64 or source_ids.numel()!=rows:
                raise ValueError('GPU response token input mismatch')
            ids=source_ids.reshape(-1)
        else:
            ids=self.ids[:rows]
        if source_ids is None or page_changed:
            slot=self.next_slot;self.next_slot=(slot+1)%2
            if self.pending[slot] and not self.done[slot].query():
                self.done[slot].synchronize()
            host_ids,host_pages=self.host[slot]
            if source_ids is None:
                host_ids.numpy()[:rows]=seq.token_ids[prefix:total]
                ids.copy_(host_ids[:rows],non_blocking=True)
            if page_changed:
                hp=host_pages.numpy();hp.fill(-1);hp[:len(page_key)]=page_key
                self.pages[0].copy_(host_pages,non_blocking=True)
                self.page_key=page_key
            self.done[slot].record();self.pending[slot]=True
        _geometry[(1,)](self.pages,self.positions,self.slots,self.length,self.cu,
            prefix,total,rows,self.block_size,triton.next_power_of_2(rows))
        return ids,self.positions[:rows],self.slots[:rows],self.length,self.pages,self.cu
