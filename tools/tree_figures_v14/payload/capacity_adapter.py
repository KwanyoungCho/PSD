"""Benchmark-only capacity reservation; no runtime file or decoding-policy edit."""
import json
import math
import os
from pathlib import Path

FRACTION = float(os.environ.get('P1_SWEEP_DRAFT_KV_FRACTION', '.35'))


def install():
    from ssd.engine.draft_runner import DraftRunner
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.scheduler import Scheduler
    import torch
    original_config = DraftRunner.create_draft_config.__func__
    original_allocate = ModelRunner.allocate_kv_cache
    original_warmup = DraftRunner._warmup_p1_tree_executors
    original_preempt = Scheduler.preempt

    def write(name, row):
        p = Path(os.environ['SSD_PROFILE_DIR']) / name
        p.write_text(json.dumps(row, indent=2) + '\n')

    def draft_config(cls, config):
        cfg = original_config(cls, config)
        cfg.gpu_memory_utilization = FRACTION
        return cfg

    def allocate(self):
        original_allocate(self)
        if not self.is_draft:
            return
        cfg = self.config
        root_cap = 4 * max(cfg.duet_p1_tree_verify_nodes + 1,
                           cfg.duet_p2_tree_verify_nodes + 1,
                           cfg.duet_phase1_k + 1, cfg.duet_phase2_k + 1)
        scratch = root_cap + (cfg.duet_phase1_k - 1) * cfg.duet_p1_forward_width
        # Four times the full context + worst live tree + two guard pages.
        # Batch is one; each completed request is released (cold prefix).
        minimum = 4 * math.ceil((cfg.max_model_len + scratch
                                + 2 * self.block_size) / self.block_size)
        blocks = int(cfg.num_kvcache_blocks)
        if blocks < minimum:
            raise RuntimeError(f'Insufficient draft capacity: {blocks} < {minimum}')
        write('DRAFT_CAPACITY.json', dict(fraction=FRACTION, allocated_blocks=blocks,
              block_size=self.block_size, full_context=cfg.max_model_len,
              worst_tree_cells=scratch, required_blocks_with_4x_margin=minimum,
              kv_bytes=self.kv_cache.numel() * self.kv_cache.element_size()))

    def warmup(self):
        result = original_warmup(self)
        free, total = torch.cuda.mem_get_info(self.device)
        rows = {str(c): dict(pages=sorted(ex.graphs), round_widths=ex.round_widths,
                             total_cells=ex.total_cells)
                for c, ex in getattr(self, '_p1_execs', {}).items()}
        write('PREWARM_CAPACITY.json', dict(free_bytes_after_p1_p2_warmup=free,
              total_bytes=total, p1=rows,
              note='Before final split-layout graph capture; telemetry records final usage.'))
        return result

    def preempt(self, seq):
        write('UNEXPECTED_PREEMPTION.json', dict(seq_id=seq.seq_id))
        raise RuntimeError('B1 sweep preempted: capacity comparability violated')

    DraftRunner.create_draft_config = classmethod(draft_config)
    ModelRunner.allocate_kv_cache = allocate
    DraftRunner._warmup_p1_tree_executors = warmup
    Scheduler.preempt = preempt
