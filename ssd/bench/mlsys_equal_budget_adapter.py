"""Isolated experimental forest scheduler with equal physical forward rows.

Only imported by mlsys_equal_budget. Last-layer and all-frontier modes share
roots, widths, sampling and the native ordered-WOR verifier. C=1 is the
root-protected chain-shape control on the SAME GPU executor. There is no
post-sampling score pruning. Serving retains a generation-order prefix.
"""
import json
import gzip
import os
from copy import copy
from pathlib import Path
import torch


def specification():
    return dict(roots=int(os.environ['DUET_EQUAL_ROOTS']),
                selection=os.getenv('DUET_EQUAL_SELECTION', 'last'),
                root_weighting=os.getenv('DUET_EQUAL_WEIGHTING', 'weighted'),
                score='root_prior * product(raw_draft_q)', calibration=False,
                thresholds=False, rerank=False,
                generation_cap='R * K_phase * C per root, nonbinding',
                control='C=1 chain shape, same tree executor',
                generation_cost='equal physical query rows; measured latency may differ')


def configure(kwargs):
    s = specification();r = s['roots']
    if kwargs.get('max_num_seqs', 1) < 1:
        raise ValueError('Positive batch required')
    if kwargs['duet_phase1_k'] != kwargs['duet_phase2_k']:
        raise ValueError('Matched phases require K1=K2')
    kwargs.update(duet_p2_budget=r, duet_tree_root_count=r,
                  duet_p1_roots_per_position=r,
                  duet_tree_proxy_threshold=0., duet_tree_conf_threshold=0.,
                  duet_p1_tree_start_threshold=0., duet_p1_tree_conf_threshold=0.)
    for phase in (1, 2):
        kwargs[f'duet_p{phase}_tree_max_nodes'] = r * kwargs[f'duet_phase{phase}_k'] * kwargs['duet_tree_c_tensor']
    print('[equal budget]', json.dumps(s), flush=True)


def select_roots(tokens, scores, contexts, count):
    """Stable score top-R; return fixed width with zero-score padding."""
    if tokens.ndim != 2 or tokens.shape != scores.shape:
        raise ValueError('Expected matching [B,candidate] tensors')
    contexts = contexts.expand_as(tokens)
    n = min(count, tokens.shape[1])
    ix = scores.argsort(dim=1, descending=True, stable=True)[:, :n]
    values = [x.gather(1, ix) for x in (tokens, scores, contexts)]
    if n < count:
        values = [torch.cat((x, x.new_zeros(x.shape[0], count-n)), 1) for x in values]
    return tuple(values)


def install():
    if getattr(install, '_installed', False):
        return
    s = specification()
    if s['selection'] not in ('last', 'frontier') or s['root_weighting'] not in ('weighted', 'uniform'):
        raise ValueError(s)
    os.environ['SSD_TREE_EXPANSION_POLICY'] = 'q_path'
    # Existing unified service also supports actual B=1. Use it for every
    # arm so the experiment changes forest selection, not verifier backend.
    os.environ['SSD_BATCHED_TREE'] = '1'
    from ssd.engine.helpers import batch_tree_draft as bd
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
    from ssd.engine.helpers.batched_tree_executor import BatchedTreeExecutor
    from ssd.engine.helpers.batch_tree_roots import BatchedP1Roots
    from ssd.engine.helpers.batch_tree_common import capacity
    from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
    from ssd.config import Config
    from ssd.engine.scheduler import Scheduler
    # At most R P1 keys can remove entries from the ranked proxy list.
    # Keep sufficient alternatives without using the old contexts*R bound.
    Config.duet_proxy_wire_N = property(lambda cfg: 2*s['roots']+2)
    original_scheduler_init = Scheduler.__init__
    def scheduler_init(self, *args, **kwargs):
        original_scheduler_init(self, *args, **kwargs)
        if self._split_k1k2:
            self._sk_mq_p1 = self._sk_mq_p2 = s['roots']
            self._sk_cells_p1 = self._sk_K1 * s['roots']
            self._sk_cells_p2 = self._sk_K2 * s['roots']
    Scheduler.__init__ = scheduler_init

    def save_entries(self, records):
        # Export only the SAME generation-order prefix that native serving
        # would retain. Large search arenas must not inflate host export to
        # R times the entire search capacity. No score-based selection.
        pieces=[];shapes=[]
        for ar,seq,ci,rt,phase,safe in records:
            if not safe:continue
            r=len(rt);m=getattr(self.cfg,f'duet_p{phase}_tree_verify_nodes')
            depth=torch.zeros(ar.R,dtype=torch.long,device=ar.dev)
            root=ar.arena.root.clamp(0,ar.R-1)
            depth.scatter_reduce_(0,root,torch.where(ar.arena.valid,ar.arena.depth,0),reduce='amax')
            processed=torch.zeros_like(depth)
            processed.scatter_add_(0,root,(ar.arena.valid & (ar.arena.state==1)).long())
            n=ar.out_valid[:r].clamp_max(m)
            value=torch.cat([ci[:,None],rt[:,None],n[:,None],ar.out_valid[:r,None],
                depth[:r,None],processed[:r,None],ar.view_tok[:r,:m],ar.view_par[:r,:m],
                ar.view_sib[:r,:m],ar.view_pcell[:r,:m]],1)
            pieces.append(value.reshape(-1));shapes.append((ar,seq,phase,r,m))
        if not pieces:return
        host=torch.cat(pieces).cpu().tolist();offset=0;stats=[]
        for ar,seq,phase,r,m in shapes:
            for root in range(r):
                a=host[offset:offset+6+4*m];offset+=6+4*m
                ctx,token,n,generated,depth,processed=a[:6]
                stats.append(dict(step=getattr(self.r,'_request_step_id',None),seq=seq,
                    phase=phase,root=root,context=ctx,token=token,generated=generated,
                    generated_depth=depth,processed=processed,served=n,
                    physical_rows=ar.total_cells,roots=ar.R,C=ar.C))
                if n<=0:continue
                self.entries.setdefault((seq,int(ctx),int(token)),dict(
                    tokens=a[6:6+m][:n],parents=a[6+m:6+2*m][:n],
                    siblings=a[6+2*m:6+3*m][:n],cells=a[6+3*m:6+4*m][:n],
                    logits=ar.cell_logits,phase=phase,tree=True))
        path=Path(os.environ['SSD_PROFILE_DIR'])/'equal_forest.jsonl.gz'
        # Close each gzip member so even an interrupted engineering attempt
        # retains every previously completed record; full-corpus traces can
        # otherwise consume hundreds of MB per arm.
        with gzip.open(path,'at',compresslevel=1) as f:
            for row in stats:f.write(json.dumps(row,separators=(',',':'))+'\n')

    def forest(self, phase, contexts, batch, pages):
        key = phase, contexts
        if key not in self.forests:
            cfg = copy(self.cfg);hf = self.r.hf_config;r = s['roots']
            cfg.duet_p1_tree_allocation_policy = 'dynamic'
            cfg.duet_tree_policy = 'dynamic'
            arenas = []
            for b in range(capacity(cfg.max_num_seqs)):
                ex = P2TreeExecutor(self.r.model, self.r.model.compute_logits, cfg,
                    self.r.device, self.r.block_size, cfg.max_blocks, hf.vocab_size,
                    hf.num_attention_heads, hf.num_key_value_heads, hf.head_dim,
                    dtype=hf.torch_dtype, phase=f'p{phase}', width=r, root_count=r,
                    depth=getattr(cfg, f'duet_phase{phase}_k'),
                    max_nodes=getattr(cfg, f'duet_p{phase}_tree_max_nodes'),
                    glue_width=contexts, materialize_backbone_logits=False)
                ex.expansion_policy.frontier = s['selection'] == 'frontier'
                # This service consumes view_{tok,par,sib,pcell} directly.
                # The O(NV^2) unique-parent/backbone projection is unused.
                ex._finalize_outputs = lambda: None
                if ex.expansion_policy.alpha is not None or ex.expansion_policy.gain is not None:
                    raise AssertionError('Calibration leaked into q-only experiment')
                ex.gen.manual_seed((torch.initial_seed()+100003*phase+1009*contexts+b) % (2**63-1))
                ex.roots_per_position = r
                if ex.R != r or any(w != r for w in ex.round_widths):
                    raise AssertionError('Physical forward width differs from root count')
                arenas.append(ex)
            self.forests[key] = arenas
            print('[equal forest]', dict(phase=phase, contexts=contexts, R=r,
                round_widths=arenas[0].round_widths, C=arenas[0].C,
                generation_cap=arenas[0].NV, selection=s['selection']), flush=True)
        arenas = self.forests[key][:batch]
        graph_key = phase, contexts, batch
        if graph_key not in self.executors:
            self.executors[graph_key] = BatchedTreeExecutor(arenas, staging_only=True,
                                                         workspace=self.glue.workspace())
        executor = self.executors[graph_key]
        if pages is not None:
            executor.prepare(pages)
        return executor, arenas

    def build(self, rows, glue, keys, temps, proxy_work, proxy_buf, B, K):
        cfg, runner = self.cfg, self.r;r = s['roots'];bc = capacity(B)
        contexts = max(len(row['tokens']) for row in rows)
        cb = next(x for x in sorted({cfg.duet_phase1_k+1, cfg.duet_phase2_k+1,
                                    cfg.duet_response_token_width+1}) if x >= contexts)
        exports = [];selected = None

        def execute(phase, roots):
            rt, sc, ci = roots
            if s['root_weighting'] == 'uniform':
                sc = (sc > 0).to(sc.dtype)
            executor, arenas = self._forest(phase, cb, bc, None)
            pages = capacity(max((row['prefix']+len(row['tokens'])+r+runner.block_size-1)
                                 // runner.block_size for row in rows))
            executor, arenas = self._forest(phase, cb, bc, pages)
            safe = self._prepare_batch(executor, pages, rows,
                [(rt[b], sc[b], ci[b]) for b in range(B)], temps)
            self._execute(executor, pages, phase)
            for b in range(B):
                exports.append((arenas[b], keys[b][0], ci[b], rt[b], phase, safe[b]))

        event = duet_record('batch_p1_total')
        if not cfg.duet_only_proxy:
            key = bc, cb
            if key not in self.root_graphs:
                graph = BatchedP1Roots(bc, cb, runner.hf_config.vocab_size,
                    runner.hf_config.torch_dtype, runner.device, r,
                    cfg.sampler_x, cfg.async_fan_out, pool=self.root_graph_pool)
                self.root_graphs[key] = graph
                if self.root_graph_pool is None:self.root_graph_pool = graph.graph.pool()
            graph = self.root_graphs[key]
            rt, sc = graph.replay(rows, glue, temps)
            selected = select_roots(rt[:B], sc[:B], graph.context_ids[None], r)
            execute(1, selected)
        duet_close('batch_p1_total', event)
        event = duet_record('batch_proxy_wait');proxy_work.wait();duet_close('batch_proxy_wait', event)
        event = duet_record('batch_p2_total')
        proxy = runner._unpack_duet_proxy(proxy_buf, B, K)
        pos, tok, score = proxy['chosen_pos'], proxy['chosen_tok'], proxy['chosen_piv']
        valid = (pos >= 0) & (pos < torch.tensor([len(x['tokens']) for x in rows], device=runner.device)[:,None])
        if selected is not None:
            st, ss, sp = selected
            duplicate = ((tok[:,:,None] == st[:,None,:]) & (pos[:,:,None] == sp[:,None,:])
                         & (ss[:,None,:] > 0)).any(-1)
            valid &= ~duplicate
        sc = torch.where(valid, score, 0)
        picked = select_roots(tok, sc, pos.clamp(0, contexts-1), r)
        execute(2, picked)
        export = duet_record('batch_bulk_export')
        self._save_entries_bulk(exports)
        duet_close('batch_bulk_export', export);duet_close('batch_p2_total', event)

    bd.BatchedDuetDraft._forest = forest
    bd.BatchedDuetDraft._build = build
    bd.BatchedDuetDraft._save_entries_bulk = save_entries
    install._installed = True
