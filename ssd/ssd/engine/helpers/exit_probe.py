"""All-layer early-exit proxy probe (measurement only).

Scores counterfactual P2 candidate policies against the target's own
correction distribution, on the REAL step boundaries of a running DUET chain.

Objective.  The engine samples a rejected position's recovery token from
``normalize([p_T - p_D]_+)`` (ssd/utils/verify.py), so with budget R the
achievable value of a candidate set C is

    U(C) = sum over (i, v) in C of  h_true[i] * r_true[i][v]

which is a plain top-R selection on the (position, token) grid.  Ranking by
``h_true (x) r_true`` is therefore Bayes-optimal -- that is exactly DUET's
``P_iv = h_i * r_i(v)`` with exact inputs, and it upper-bounds every policy
below.  ``ceiling`` records it; if any policy beats it, this file is wrong.

Policies are a factorial of position budgeting x token ranking, so the loss
of DUET's two estimates can be separated:

    position:  hhat (alpha-hat hazard) | htrue (exact) | unif (round-robin)
    tokens:    rE = [p_E - p_D]_+ | pE | pD | rT = [p_T - p_D]_+ | pT

``hhat x rE`` is what the engine runs; ``* x pE`` is the target-only
baseline; oracle rows (htrue, rT, pT) isolate approximation error from the
formula itself.
"""

from __future__ import annotations

import json
import os

import torch

HPOL = ("hhat", "htrue", "unif")
RSRC = ("rE", "pE", "pD", "rT", "pT")
_EPS = 1e-10


def _hazard_batch(alpha: torch.Tensor) -> torch.Tensor:
    """alpha [n, K] -> h [n, K+1]; h[:, K] is the all-accept mass."""
    n, K = alpha.shape
    cum = torch.cumprod(alpha, dim=1)
    h = torch.zeros(n, K + 1, dtype=alpha.dtype, device=alpha.device)
    h[:, 0] = 1 - alpha[:, 0]
    if K > 1:
        h[:, 1:K] = cum[:, :-1] * (1 - alpha[:, 1:])
    h[:, K] = cum[:, -1]
    return h


def _topm(src, gi, M):
    """Per-position top-M of a [n, K, V] score, draft token excluded."""
    src = src.clone()
    src.scatter_(2, gi.expand(src.shape[0], -1, -1), 0.0)
    p, idx = src.topk(M, dim=-1)
    return p / p.sum(-1, keepdim=True).clamp(min=_EPS), idx


def _score(cand_p, cand_id, hw, gt, M, Rs, rr=None):
    """Global top-R over the (position, token) grid -> cumulative true value.

    cand_p/cand_id: [n, K+1, M]; hw: [n, K+1] or None for round-robin.
    Returns [n, len(Rs)].
    """
    n = cand_p.shape[0]
    flat_id = cand_id.reshape(n, -1)
    if hw is None:
        order = rr.unsqueeze(0).expand(n, -1)
    else:
        order = (hw.unsqueeze(2) * cand_p).reshape(n, -1).argsort(
            dim=1, descending=True)
    sel_pos = torch.div(order, M, rounding_mode="floor")
    gain = gt[sel_pos, flat_id.gather(1, order)]
    return gain.cumsum(1).double().index_select(1, Rs)


def _renorm(p):
    return p / p.sum(-1, keepdim=True).clamp(min=_EPS)


class ExitProbe:
    DELTAS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    MS = (16, 32, 64, 128)

    def __init__(self, layers, top_m=32, budgets=(1, 2, 4, 8, 15, 32, 64),
                 out_path="exit_probe.json", exit_layer=None):
        self.layers = list(layers)
        self.top_m = int(top_m)
        self.budgets = [int(r) for r in budgets if int(r) <= int(top_m) * 2]
        self.out_path = out_path
        self.n_steps = 0
        self.hit = None      # [L, |HPOL|, |RSRC|, |R|]
        self.ceil = None     # [|R|]
        self.diag = None     # [L, 5]
        self.glob = None     # [5]
        self.exit_layer = exit_layer
        # Controlled-noise control: p_noisy = normalize(p_T * exp(delta*xi)).
        # If the residual really is more fragile to ANY perturbation, rN must
        # decay faster than pN here too -- independently of whatever structure
        # the real early-exit error happens to have.
        self.noise = None    # [|DELTAS|, 2, |R|]
        self.ntvd = None     # [|DELTAS|]
        self.msw = None      # [|MS|, 4, |R|]  (htrue/hhat x rE/pE at exit layer)
        self.approx = None   # [L, 8] approximation quality + p_E character
        self.bins = None     # [4, 4] gap attribution by target/draft overlap
        self._gen = None
        # Rank profile: line up tokens by p_T rank and read p^E on the SAME
        # tokens.  Shows directly whether the early exit deflates the head and
        # inflates the tail, and how often it does so.
        self.rk_pT = None    # [M]      mean p_T at p_T-rank r
        self.rk_pE = None    # [L, M]   mean p^E at the same token
        self.rk_lo = None    # [L, M]   P(p^E < p_T) at that rank
        self.rk_top1 = None  # [L, 5]   where p_T's top-1 lands in p^E's order

    @torch.inference_mode()
    def observe(self, p_E_all, p_D, p_T, y):
        L, K1, V = p_E_all.shape
        K = K1 - 1
        dev = p_E_all.device
        gi = y.view(1, K, 1)
        M = min(self.top_m, V)

        if self.hit is None:
            self.hit = torch.zeros(L, len(HPOL), len(RSRC), len(self.budgets),
                                   dtype=torch.float64, device=dev)
            self.ceil = torch.zeros(len(self.budgets), dtype=torch.float64,
                                    device=dev)
            self.diag = torch.zeros(L, 5, dtype=torch.float64, device=dev)
            self.glob = torch.zeros(5, dtype=torch.float64, device=dev)
            self._Rs = torch.tensor([r - 1 for r in self.budgets],
                                    device=dev, dtype=torch.long)
            # round-robin visiting order over the flattened [K+1, M] grid
            self._rr = (torch.arange(K1 * M, device=dev)
                        .view(K1, M).t().reshape(-1))
            self.noise = torch.zeros(len(self.DELTAS), 2, len(self.budgets),
                                     dtype=torch.float64, device=dev)
            self.ntvd = torch.zeros(len(self.DELTAS), dtype=torch.float64,
                                    device=dev)
            self.msw = torch.zeros(len(self.MS), 4, len(self.budgets),
                                   dtype=torch.float64, device=dev)
            self._el = (self.layers.index(self.exit_layer)
                        if self.exit_layer in self.layers else None)
            self.approx = torch.zeros(L, 8, dtype=torch.float64, device=dev)
            self.rk_pT = torch.zeros(M, dtype=torch.float64, device=dev)
            self.rk_pE = torch.zeros(L, M, dtype=torch.float64, device=dev)
            self.rk_lo = torch.zeros(L, M, dtype=torch.float64, device=dev)
            self.rk_top1 = torch.zeros(L, 5, dtype=torch.float64, device=dev)
            self.bins = torch.zeros(4, 4, dtype=torch.float64, device=dev)

        # ---------- ground truth ----------
        pD_y = p_D.gather(1, gi[0]).squeeze(1)                      # [K]
        a_true = (p_T[:K].gather(1, gi[0]).squeeze(1)
                  / (pD_y + _EPS)).clamp(max=1.0)
        h_true = _hazard_batch(a_true.unsqueeze(0))[0]               # [K+1]
        r_true = torch.empty(K1, V, device=dev, dtype=torch.float32)
        _rt = (p_T[:K] - p_D).clamp(min=0)
        _rt.scatter_(1, gi[0], 0.0)
        r_true[:K] = _rt / _rt.sum(1, keepdim=True).clamp(min=_EPS)
        r_true[K] = p_T[K]
        gt = h_true.unsqueeze(1) * r_true                            # [K+1, V]

        # exact ceiling: top-R of the true objective
        cv, ci = gt.topk(M, dim=-1)
        cval = cv.reshape(-1).sort(descending=True).values
        self.ceil += cval.cumsum(0).double().index_select(0, self._Rs)

        # ---------- per-layer hazards ----------
        pE_y = p_E_all[:, :K].gather(2, gi.expand(L, -1, -1)).squeeze(2)
        a_hat = (pE_y / (pD_y + _EPS)).clamp(max=1.0)                # [L, K]
        h_hat = _hazard_batch(a_hat)                                 # [L, K+1]

        # ---------- candidate sets per token source ----------
        cand = {}
        pe_last = p_E_all[:, K]                                      # [L, V]
        lr_E, lid_E = pe_last.topk(M, dim=-1)
        lr_E = lr_E / lr_E.sum(-1, keepdim=True).clamp(min=_EPS)
        lr_T, lid_T = p_T[K].topk(M)
        lr_T = (lr_T / lr_T.sum().clamp(min=_EPS)).unsqueeze(0)   # [1, M]
        lid_T = lid_T.unsqueeze(0)                                # [1, M]

        pr, pid = _topm((p_E_all[:, :K] - p_D).clamp(min=0), gi, M)
        cand["rE"] = (pr, pid, lr_E, lid_E)
        pr, pid = _topm(p_E_all[:, :K], gi, M)
        cand["pE"] = (pr, pid, lr_E, lid_E)
        pr, pid = _topm(p_D.unsqueeze(0), gi, M)
        cand["pD"] = (pr, pid, lr_E, lid_E)          # engine keeps p_E at pos K
        pr, pid = _topm((p_T[:K] - p_D).clamp(min=0).unsqueeze(0), gi, M)
        cand["rT"] = (pr, pid, lr_T, lid_T)
        pr, pid = _topm(p_T[:K].unsqueeze(0), gi, M)
        cand["pT"] = (pr, pid, lr_T, lid_T)

        h_map = {"hhat": h_hat, "htrue": h_true.unsqueeze(0), "unif": None}

        for hi, hp in enumerate(HPOL):
            hw = h_map[hp]
            for ri, rs in enumerate(RSRC):
                pr, pid, lr, lid = cand[rs]
                n = max(pr.shape[0], lr.shape[0],
                        1 if hw is None else hw.shape[0])
                cp = torch.cat([pr.expand(n, K, M),
                                lr.unsqueeze(1).expand(n, 1, M)], 1)
                cid = torch.cat([pid.expand(n, K, M),
                                 lid.unsqueeze(1).expand(n, 1, M)], 1)
                acc = _score(cp, cid, None if hw is None else hw.expand(n, K1),
                             gt, M, self._Rs, self._rr)
                self.hit[:, hi, ri, :] += acc if n == L else acc.expand(L, -1)

        # ---------- control A: controlled noise on p_T ----------
        if self._gen is None:
            self._gen = torch.Generator(device=dev); self._gen.manual_seed(1234)
        xi = torch.randn(p_T.shape, device=dev, dtype=p_T.dtype,
                         generator=self._gen)
        for di, delta in enumerate(self.DELTAS):
            pn = _renorm(p_T * (delta * xi).exp())
            self.ntvd[di] += (0.5 * (pn - p_T).abs().sum(-1).mean()).double()
            lrn, lidn = pn[K].topk(M)
            lrn = _renorm(lrn).unsqueeze(0).unsqueeze(1)
            lidn = lidn.unsqueeze(0).unsqueeze(1)
            for ai, src in enumerate(((pn[:K] - p_D).clamp(min=0),
                                      pn[:K])):
                prn, pidn = _topm(src.unsqueeze(0), gi, M)
                cp = torch.cat([prn, lrn], 1)
                cid = torch.cat([pidn, lidn], 1)
                self.noise[di, ai] += _score(
                    cp, cid, h_true.unsqueeze(0), gt, M, self._Rs)[0]

        # ---------- control B: candidate-pool depth (top_m) at exit layer ----
        if self._el is not None:
            for mi, m in enumerate(self.MS):
                if m > M:
                    continue
                for ai, rs in enumerate(("rE", "pE")):
                    pr, pid, lr, lid = cand[rs]
                    pr_m = _renorm(pr[self._el:self._el + 1, :, :m])
                    pid_m = pid[self._el:self._el + 1, :, :m]
                    lr_m = _renorm(lr[self._el:self._el + 1, :m]).unsqueeze(1)
                    lid_m = lid[self._el:self._el + 1, :m].unsqueeze(1)
                    cp = torch.cat([pr_m, lr_m], 1)
                    cid = torch.cat([pid_m, lid_m], 1)
                    rr_m = (torch.arange(K1 * m, device=dev)
                            .view(K1, m).t().reshape(-1))
                    for hj, hw2 in enumerate((h_true.unsqueeze(0),
                                              h_hat[self._el:self._el + 1])):
                        self.msw[mi, hj * 2 + ai] += _score(
                            cp, cid, hw2, gt, m, self._Rs, rr_m)[0]

        # ---------- diagnostics ----------
        d0 = 0.5 * (p_E_all - p_T.unsqueeze(0)).abs().sum(-1).mean(1)   # TVD pE,pT
        rE_n, _ = _topm((p_E_all[:, :K] - p_D).clamp(min=0), gi, V if V < 4096 else M)
        d1 = 0.5 * (cand["rE"][0] - cand["rT"][0].expand(L, K, M)).abs().sum(-1).mean(1)
        d2 = 0.5 * (h_hat - h_true.unsqueeze(0)).abs().sum(-1)          # TVD h
        d3 = (a_hat - a_true.unsqueeze(0)).abs().mean(1)                # MAE alpha
        d4 = (h_hat[:, :K].argmax(1) == h_true[:K].argmax()).double()   # anchor top1
        self.diag += torch.stack([d0, d1, d2, d3, d4], 1).double()

        # ---------- how well does each FORM approximate r_true? ----------
        # Distances are to the distribution the engine actually samples from,
        # over the full vocabulary, weighted by the true reject probability.
        w = h_true[:K] / h_true[:K].sum().clamp(min=_EPS)                # [K]
        rt = r_true[:K]                                                  # [K, V]
        pE_p = p_E_all[:, :K].clone()
        pE_p.scatter_(2, gi.expand(L, -1, -1), 0.0)
        pE_p = _renorm(pE_p)                                             # p^E form
        rE_p = (p_E_all[:, :K] - p_D).clamp(min=0)
        rE_p.scatter_(2, gi.expand(L, -1, -1), 0.0)
        rE_p = _renorm(rE_p)                                             # residual form
        pT_p = p_T[:K].clone(); pT_p.scatter_(1, gi[0], 0.0); pT_p = _renorm(pT_p)

        def _wtvd(x, y):
            return (0.5 * (x - y).abs().sum(-1) * w).sum(-1)
        a0 = _wtvd(rE_p, rt.unsqueeze(0))          # residual form vs truth
        a1 = _wtvd(pE_p, rt.unsqueeze(0))          # proxy form vs truth
        a2 = _wtvd(pT_p.unsqueeze(0), rt.unsqueeze(0)).expand(L)  # oracle proxy bias
        a3 = _wtvd(p_E_all[:, :K], p_D.unsqueeze(0))   # is p^E pulled toward p_D?
        a4 = _wtvd(p_T[:K].unsqueeze(0), p_D.unsqueeze(0)).expand(L)
        ent = lambda x: -(x.clamp(min=1e-12).log() * x).sum(-1)
        a5 = (ent(p_E_all[:, :K]) * w).sum(-1)
        a6 = (ent(p_T[:K]).unsqueeze(0) * w).sum(-1).expand(L)
        a7 = (ent(p_D).unsqueeze(0) * w).sum(-1).expand(L)
        self.approx += torch.stack([a0, a1, a2, a3, a4, a5, a6, a7], 1).double()

        # ---------- rank profile: p^E read on p_T's own ranking ----------
        srt = p_T[:K].argsort(dim=-1, descending=True)[:, :M]            # [K, M]
        pT_at = p_T[:K].gather(1, srt)                                   # [K, M]
        pE_at = p_E_all[:, :K].gather(2, srt.unsqueeze(0).expand(L, K, M))
        self.rk_pT += (pT_at * w.unsqueeze(1)).sum(0).double()
        self.rk_pE += (pE_at * w.view(1, K, 1)).sum(1).double()
        self.rk_lo += ((pE_at < pT_at.unsqueeze(0)).to(pE_at.dtype)
                       * w.view(1, K, 1)).sum(1).double()
        # where does p_T's own top-1 sit in p^E's ordering?
        t1 = srt[:, 0].view(1, K, 1).expand(L, K, 1)
        rk = (p_E_all[:, :K] > p_E_all[:, :K].gather(2, t1)).sum(-1)      # [L, K]
        # rk is 0-based (0 = p^E also ranks it first).  Explicit comparisons
        # rather than bucketize: the boundary convention there is off by one
        # and silently produced an empty "rank 1" bin.
        masks = [rk == 0, (rk >= 1) & (rk <= 2), (rk >= 3) & (rk <= 9),
                 (rk >= 10) & (rk <= 99), rk >= 100]
        for j, mk in enumerate(masks):
            self.rk_top1[:, j] += (mk.to(w.dtype) * w.view(1, K)).sum(1).double()

        # ---------- where do we lose?  bucket positions by p_T/p_D overlap ----
        if self._el is not None:
            ov = 1 - 0.5 * (p_T[:K] - p_D).abs().sum(-1)                 # [K]
            b = torch.bucketize(ov, torch.tensor([.25, .5, .75], device=dev))
            le = self._el
            g_r = (rt * rE_p[le]).sum(-1)      # crude per-position agreement
            g_p = (rt * pE_p[le]).sum(-1)
            for k in range(4):
                m = (b == k)
                if m.any():
                    self.bins[k, 0] += (w * m).sum().double()
                    self.bins[k, 1] += (w * m * g_r).sum().double()
                    self.bins[k, 2] += (w * m * g_p).sum().double()
                    self.bins[k, 3] += (w * m * ov).sum().double()

        # champion policy position concentration at R=15 (or the largest <=15)
        rr = min([b for b in self.budgets if b <= 15], key=lambda b: abs(b - 15))
        k15 = self.budgets.index(rr)
        pr, pid, lr, lid = cand["rE"]
        cp = torch.cat([pr, lr.unsqueeze(1)], 1)   # lr is [L, M]
        piv = (h_hat.unsqueeze(2) * cp).reshape(L, -1)
        sel = piv.argsort(dim=1, descending=True)[:, :rr]
        pos = torch.div(sel, M, rounding_mode="floor")
        oh = torch.zeros(L, K1, device=dev)
        oh.scatter_add_(1, pos, torch.ones_like(pos, dtype=oh.dtype))
        frac = oh / rr
        ent = -(frac.clamp(min=1e-12).log() * frac).sum(1)
        self.glob += torch.stack([
            h_true[:K].sum().double(), h_true[K].double(),
            frac[:, K].mean().double(),          # share of budget at pos K
            ent.exp().mean().double(),           # effective #positions
            p_T[K].max().double(),
        ])
        self.n_steps += 1

    def dump(self):
        if self.hit is None:
            return
        n = max(self.n_steps, 1)
        g = (self.glob / n).tolist()
        out = {
            "n_steps": self.n_steps, "layers": self.layers,
            "hpol": list(HPOL), "rsrc": list(RSRC),
            "budgets": self.budgets, "top_m": self.top_m,
            "hit": (self.hit / n).tolist(),
            "ceiling": (self.ceil / n).tolist(),
            "diag_cols": ["tvd_pE_pT", "tvd_rE_rT", "tvd_h", "mae_alpha",
                          "anchor_top1"],
            "diag": (self.diag / n).tolist(),
            "approx_cols": ["tvd_residualform_vs_true", "tvd_proxyform_vs_true",
                            "tvd_oracleproxy_vs_true", "tvd_pE_pD", "tvd_pT_pD",
                            "H_pE", "H_pT", "H_pD"],
            "approx": (self.approx / n).tolist(),
            "rank_pT": (self.rk_pT / n).tolist(),
            "rank_pE": (self.rk_pE / n).tolist(),
            "rank_pE_below_pT": (self.rk_lo / n).tolist(),
            "rank_of_pT_top1": (self.rk_top1 / n).tolist(),
            "rank_top1_bins": ["1", "2-3", "4-10", "11-100", ">100"],
            "bins": (self.bins / n).tolist(),
            "bins_cols": ["mass", "resid_agree", "proxy_agree", "overlap"],
            "deltas": list(self.DELTAS),
            "noise_hit": (self.noise / n).tolist(),
            "noise_tvd": (self.ntvd / n).tolist(),
            "ms": list(self.MS),
            "msweep": (self.msw / n).tolist(),
            "msweep_cols": ["htrue_rE", "htrue_pE", "hhat_rE", "hhat_pE"],
            "exit_layer": self.exit_layer,
            "reject_mass": g[0], "all_accept_mass": g[1],
            "budget_share_posK": g[2], "eff_positions": g[3],
            "pT_K_top1": g[4],
        }
        tmp = self.out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, self.out_path)
