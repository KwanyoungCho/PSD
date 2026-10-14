"""Post-hoc conditional AL for the exact tree proposals already verified.

Diagnostic only: readbacks affect overlap; never use this run's TPS. Capture
the acceptance probabilities already computed by the verifier, without new
random draws or a second vocabulary-sized residual calculation.
"""
import inspect
import textwrap
import numpy as np


def install():
    import ssd.engine.helpers.batch_tree_sampling as module
    source=textwrap.dedent(inspect.getsource(module.verify_batch))
    needle='        alpha,_,residual=ladder(tokens,p,q,topology,N,exact=True)\n'
    assert needle in source
    source=source.replace(needle,needle+'        globals()["_review_alpha"] = alpha\n')
    scope={}
    exec(compile(source,'<conditional-tree-AL>','exec'),module.__dict__,scope)
    module.verify_batch=scope['verify_batch']
    original=module.BatchedTreeAccept.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self._review_alpha=module._review_alpha
    module.BatchedTreeAccept.__init__=init


def annotate(llm,events):
    if not events:return
    from ssd.engine.helpers.batch_tree_common import capacity
    r=llm.model_runner
    n=events[0]['physical_verify_width']
    graph=r._batch_tree_accept[capacity(len(events)),n,False,r.config.stochastic_only]
    alpha=graph._review_alpha[:len(events)].cpu().numpy()
    topo=graph.proxy.topology
    parents=topo['par'][:len(events)].cpu().numpy()
    siblings=topo['sib'][:len(events)].cpu().numpy()
    valid=topo['node_valid'][:len(events)].cpu().numpy()
    for b,event in enumerate(events):
        reach=np.zeros(n,dtype=float)
        for j in range(n):
            if not valid[b,j]:continue
            parent=int(parents[b,j]); sibling=int(siblings[b,j])
            assert parent<j
            preceding=np.prod(1-alpha[b,parent+1,:sibling])
            reach[j]=(1 if parent<0 else reach[parent])*preceding*alpha[b,parent+1,sibling]
        expected=1+float(reach.sum())
        event['conditional_expected_len']=expected
        event['conditional_expected_len_if_miss_ar']=expected if event['cache_hit'] else 1.
