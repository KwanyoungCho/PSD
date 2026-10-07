"""Passive diagnostic snapshots; never use these runs for throughput."""
from pathlib import Path
import json
import numpy as np
import torch


class TreeObserver:
    def __init__(self, directory):
        from ssd.engine.verifier import Verifier
        from ssd.engine.helpers.p2_tree import parse_tree_ints, q_probs_from_logits
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        if list(self.directory.glob('tree*.npz')):raise FileExistsError(directory)
        self.active=False;self.count=0;self.records=[]
        original=Verifier._tree_verify_walk
        observer=self
        def observe(verifier,result,logits,tt,td):
            answer=original(verifier,result,logits,tt,td)
            if not observer.active:return answer
            index=observer.count;observer.count+=1
            if index%4 or len(observer.records)>=128:return answer
            if float(tt[0])<=0:raise ValueError('Calibration requires T>0')
            nv=verifier.target_model_runner.config.duet_tree_wire_nodes
            ti=parse_tree_ints(result.tree_ints[0].cpu(),nv);n=int(ti['valid'])
            pq=result.parent_q_logits[0].float()
            probs=q_probs_from_logits(pq,torch.full((len(pq),),float(td[0]),device=pq.device),
                                      verifier.sampler_x,verifier.async_fan_out)
            p=torch.softmax(logits[0,:n+1].float()/float(tt[0]),-1).cpu().numpy()
            q=probs.index_select(0,ti['parent_q_ref'][:n].to(pq.device)).cpu().numpy()
            name=f'tree{index:05d}.npz'
            np.savez_compressed(observer.directory/name,p=p,q=q,
                par=ti['parent_local'][:n].numpy(),sib=ti['sib_order'][:n].numpy(),
                tok=ti['tok'][:n].numpy(),path=np.asarray(answer[3],dtype=np.int64))
            observer.records.append(dict(file=name,step_id=result.step_id,
                                         phase=int(result.phase_source[0]),valid=n))
            (observer.directory/'manifest.json').write_text(json.dumps(observer.records,indent=2))
            return answer
        Verifier._tree_verify_walk=observe
