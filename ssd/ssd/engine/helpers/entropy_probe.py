"""Per-context, full-vocabulary entropy measurements on DUET chain steps.

Stores compact scalar observations, not logits. No sampling, top-k truncation,
candidate-token exclusion, or policy feedback. Chunked NPZ files retain exact
per-row values so offline quantiles and alternative weighting are possible.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


class EntropyProbe:
    def __init__(self, layers, out_path, exit_layer=None, flush_steps=250):
        self.layers = list(layers)
        self.out_path = Path(out_path)
        self.chunk_dir = self.out_path.with_suffix("")
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.exit_layer = exit_layer
        self.flush_steps = flush_steps
        self.n_steps = 0
        self.n_rows = 0
        self.chunks = []
        self.pending = []
        self.max_prob_sum_error = 0.0
        self.temperature = None

    @torch.inference_mode()
    def observe(self, p_E, p_D, p_T, y, seq_id=-1, prefix_len=-1,
                temperature=1.0):
        L, rows, V = p_E.shape
        K = rows - 1
        if L != len(self.layers) or p_T.shape != (rows, V) or p_D.shape != (K, V):
            raise ValueError("entropy probe input shape mismatch")
        if K < 1 or y.shape != (K,):
            raise ValueError("entropy probe requires a nonempty chain")
        if self.temperature is not None and self.temperature != temperature:
            raise ValueError("mixed temperatures in one entropy probe")
        self.temperature = float(temperature)
        # Inputs are full-vocabulary float32 probabilities from the same
        # context. x*log(x) uses the exact zero-mass convention.
        def entropy(p):
            return -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)

        he, ht = entropy(p_E), entropy(p_T)
        peak_e, id_e = p_E.max(-1)
        peak_t, id_t = p_T.max(-1)
        at_t = p_E.gather(2, id_t.view(1, rows, 1).expand(L, -1, -1)).squeeze(-1)
        tvd = 0.5 * (p_E - p_T.unsqueeze(0)).abs().sum(-1)
        norm_error = torch.stack([(p_E.sum(-1) - 1).abs().max(),
                                  (p_T.sum(-1) - 1).abs().max(),
                                  (p_D.sum(-1) - 1).abs().max()]).max().item()
        if not torch.isfinite(he).all() or not torch.isfinite(ht).all() or norm_error > 1e-4:
            raise ValueError(f"invalid entropy probe probabilities: norm error {norm_error}")
        self.max_prob_sum_error = max(self.max_prob_sum_error, norm_error)
        idx = y.view(K, 1)
        alpha = (p_T[:K].gather(1, idx).squeeze(1)
                 / (p_D.gather(1, idx).squeeze(1) + 1e-10)).clamp(max=1)
        reach = torch.cat([alpha.new_ones(1), alpha.cumprod(0)])
        hazard = reach.clone()
        hazard[:K] *= 1 - alpha
        overlap = 1 - 0.5 * (p_T[:K] - p_D).abs().sum(-1)
        hd = torch.cat([entropy(p_D), ht.new_full((1,), float("nan"))])
        overlap = torch.cat([overlap, ht.new_full((1,), float("nan"))])
        # One transfer for all scalar float observations. Layout [rows, L]
        # for layer-dependent arrays, [rows] for target/context attributes.
        floats = torch.cat([
            he.T, peak_e.T, at_t.T, tvd.T,
            ht[:, None], peak_t[:, None], hd[:, None],
            hazard[:, None], reach[:, None], overlap[:, None],
        ], dim=1).cpu().numpy()
        item = {name: floats[:, j*L:(j+1)*L]
                for j, name in enumerate(("entropy_proxy", "peak_proxy",
                                          "proxy_at_target_top1", "tvd"))}
        for j, name in enumerate(("entropy_target", "peak_target", "entropy_draft",
                                  "hazard", "reach", "overlap")):
            item[name] = floats[:, 4*L+j]
        item["top1_match"] = (id_e == id_t.unsqueeze(0)).T.cpu().numpy()
        item.update(
            step=np.full(rows, self.n_steps, dtype=np.int32),
            seq_id=np.full(rows, seq_id, dtype=np.int32),
            prefix_len=np.full(rows, prefix_len, dtype=np.int32),
            position=np.arange(rows, dtype=np.int16),
            is_bonus=np.arange(rows) == K,
        )
        self.pending.append(item)
        self.n_steps += 1
        self.n_rows += rows
        if self.n_steps % self.flush_steps == 0:
            self.dump()

    def dump(self):
        if not self.n_steps:
            return
        if self.pending:
            arrays = {key: np.concatenate([r[key] for r in self.pending], axis=0)
                      for key in self.pending[0]}
            name = f"chunk_{len(self.chunks):04d}.npz"
            dest = self.chunk_dir / name
            with open(str(dest) + ".tmp", "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(str(dest) + ".tmp", dest)
            self.chunks.append({"file": str(dest.relative_to(self.out_path.parent)),
                                "rows": len(arrays["step"]),
                                "first_step": int(arrays["step"][0]),
                                "last_step": int(arrays["step"][-1])})
            self.pending.clear()
        metadata = dict(
            schema="duet_entropy_rows_v1", layers=self.layers,
            exit_layer=self.exit_layer, temperature=self.temperature,
            entropy_unit="nats", n_steps=self.n_steps, n_rows=self.n_rows,
            chunks=self.chunks, max_prob_sum_error=self.max_prob_sum_error,
            row_definition="one chain verification context; bonus row marked separately",
            weighting="raw rows; hazard is true first-reject/all-accept probability",
        )
        tmp = str(self.out_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(tmp, self.out_path)
