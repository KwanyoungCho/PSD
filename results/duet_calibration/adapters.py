"""Config/shape boundary for a replaceable tree selector.

These descriptors do not pretend that inactive/padded nodes save GPU work.
Measured cost and quality must be attached for the exact policy/shape domain.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    phase1_rounds: int
    phase2_rounds: int
    phase1_roots: int
    phase2_root_capacity: int
    phase1_generated_per_root: int
    phase2_generated_per_root: int
    phase1_verify_queries: int
    phase2_verify_queries: int
    selector: str


class ChainAdapter:
    name='chain-v1'

    def validate(self,cfg):
        k1,k2=cfg['k1'],cfg['k2']
        if not 1<=k2<=k1:raise ValueError('Current DUET engine requires K1>=K2>=1')
        if cfg['budget']<1 or cfg.get('fanout',3)<1:raise ValueError('Positive root budgets required')
        if not 0<cfg['exit']<80:raise ValueError('This 80-layer campaign cannot transfer to another target without calibration')

    def shape(self,cfg,current_verify_nodes):
        self.validate(cfg)
        return Shape(cfg['k1'],cfg['k2'],cfg.get('fanout',3)*(current_verify_nodes+1),cfg['budget'],
                     cfg['k1'],cfg['k2'],cfg['k1']+1,cfg['k2']+1,self.name)

    def prediction_supported(self,cfg):
        self.validate(cfg)
        return (cfg['mode']=='chain' and cfg['budget']==15 and cfg.get('fanout',3)==3
                and cfg['candidate']=='legacy' and 40<=cfg['exit']<=72
                and 4<=cfg['k1']<=10 and 2<=cfg['k2']<=4)


class CurrentTreeAdapter(ChainAdapter):
    name='current-dynamic-tree-v1'

    def validate(self,cfg):
        super().validate(cfg)
        if cfg['mode'] not in ['p2tree','fulltree']:raise ValueError('Tree mode required')
        if not cfg['k2']<=cfg['n2']<=3*cfg['k2']:raise ValueError('N2 generation outside C=3 round capacity')
        for n,v in [(cfg['n2'],cfg['v2'])]+([(cfg['n1'],cfg['v1'])] if cfg['mode']=='fulltree' else []):
            if not 1<=v<=n:raise ValueError('Verify node count must be within generated nodes')

    def shape(self,cfg,current_verify_nodes):
        self.validate(cfg)
        p1tree=cfg['mode']=='fulltree'
        return Shape(cfg['k1'],cfg['k2'],3*(current_verify_nodes+1),cfg['budget'],
                     cfg['n1'] if p1tree else cfg['k1'],cfg['n2'],
                     cfg['v1']+1 if p1tree else cfg['k1']+1,cfg['v2']+1,self.name)

    def prediction_supported(self,cfg):
        self.validate(cfg)
        return False  # Chain cost/reward coefficients must not silently transfer.


def cli_arguments(cfg):
    adapter=ChainAdapter() if cfg['mode']=='chain' else CurrentTreeAdapter()
    adapter.validate(cfg)
    args=['--duet','--duet_exit_layer',str(cfg['exit']),'--duet_k1',str(cfg['k1']),
          '--duet_k2',str(cfg['k2']),'--duet_p1_fanout',str(cfg.get('fanout',3)),
          '--duet_p2_budget',str(cfg['budget']),'--duet_p1_tree_policy','on' if cfg['mode']=='fulltree' else 'off',
          '--duet_p2_tree_policy','off' if cfg['mode']=='chain' else 'on']
    if cfg['mode']!='chain':
        args+=['--duet_tree_c_tensor','3','--duet_tree_root_count',str(cfg['budget']),
               '--duet_p1_tree_max_nodes',str(cfg['n1']),'--duet_p1_tree_verify_nodes',str(cfg['v1']),
               '--duet_p2_tree_max_nodes',str(cfg['n2']),'--duet_p2_tree_verify_nodes',str(cfg['v2'])]
    return args
