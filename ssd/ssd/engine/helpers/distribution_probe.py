"""Lossless, periodically sampled chain distributions for offline policy replay.

Keeps float32 probabilities, including the bonus context and actual draft
tokens. Does not draw random numbers or feed any changes into the decoder.
"""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import torch


class DistributionProbe:
    def __init__(self, layers, out_path, exit_layer=None, stride=8, flush_steps=32):
        self.layers = list(layers)
        self.out_path = Path(out_path)
        self.chunk_dir = self.out_path.with_suffix("")
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.exit_layer = exit_layer
        self.stride = int(stride)
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.flush_steps = flush_steps
        self.n_steps = self.n_samples = self.n_rows = 0
        self.seq_steps = defaultdict(int)
        self.chunks, self.pending = [], []
        self.max_prob_sum_error = 0.0
        self.temperature = None

    @torch.inference_mode()
    def observe(self, p_E, p_D, p_T, y, seq_id=-1, prefix_len=-1, temperature=1.0):
        step = self.n_steps
        local_step = self.seq_steps[seq_id]
        self.n_steps += 1
        self.seq_steps[seq_id] += 1
        # Sample once every eight verification steps independently per prompt.
        if local_step % self.stride:
            return
        L, rows, V = p_E.shape
        K = rows - 1
        if L != len(self.layers) or p_T.shape != (rows, V) or p_D.shape != (K, V) or y.shape != (K,):
            raise ValueError("distribution probe input shape mismatch")
        if self.temperature is not None and self.temperature != temperature:
            raise ValueError("mixed temperatures")
        self.temperature = float(temperature)
        tensors = (p_E, p_T, p_D)
        error = max((p.sum(-1) - 1).abs().max().item() for p in tensors)
        if error > 1e-4 or any(not torch.isfinite(p).all() or (p < 0).any() for p in tensors):
            raise ValueError("invalid probabilities")
        self.max_prob_sum_error = max(error, self.max_prob_sum_error)
        self.pending.append(dict(
            p_E=p_E.transpose(0, 1).cpu().numpy(),
            p_T=p_T.cpu().numpy(),
            p_D=torch.cat([p_D, p_D.new_zeros(1, V)]).cpu().numpy(),
            y=torch.cat([y, y.new_full((1,), -1)]).cpu().numpy(),
            step=np.full(rows, step, dtype=np.int32),
            local_step=np.full(rows, local_step, dtype=np.int32),
            seq_id=np.full(rows, seq_id, dtype=np.int32),
            prefix_len=np.full(rows, prefix_len, dtype=np.int32),
            position=np.arange(rows, dtype=np.int16),
            is_bonus=np.arange(rows) == K,
        ))
        self.n_samples += 1
        self.n_rows += rows
        if self.n_samples % self.flush_steps == 0:
            self.dump()

    def dump(self):
        if not self.n_steps:
            return
        if self.pending:
            arrays = {k: np.concatenate([r[k] for r in self.pending]) for k in self.pending[0]}
            path = self.chunk_dir / f"chunk_{len(self.chunks):04d}.npz"
            with open(str(path) + ".tmp", "wb") as f:
                np.savez(f, **arrays)
            os.replace(str(path) + ".tmp", path)
            self.chunks.append(dict(file=str(path.relative_to(self.out_path.parent)),
                                    rows=len(arrays["step"])))
            self.pending.clear()
        meta = dict(schema="duet_full_distribution_v1", layers=self.layers,
                    exit_layer=self.exit_layer, temperature=self.temperature,
                    stride=self.stride, n_steps=self.n_steps, n_samples=self.n_samples,
                    n_rows=self.n_rows, seq_steps=dict(self.seq_steps), chunks=self.chunks,
                    max_prob_sum_error=self.max_prob_sum_error,
                    sampling="per-sequence local verification step 0, stride, 2*stride, ...",
                    dtype="float32; bonus draft row zero and y=-1")
        tmp = str(self.out_path) + ".tmp"
        Path(tmp).write_text(json.dumps(meta, indent=2))
        os.replace(tmp, self.out_path)
