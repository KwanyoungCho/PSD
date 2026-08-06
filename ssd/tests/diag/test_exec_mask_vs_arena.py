"""실행기 _pack_row_mask vs arena _arena_mask_pack 순수 대조.
동일 (context_len, f, K_glue, glue, anc, sel_valid, prefix)에서
unpacked bool mask가 bit-동일한지. 다르면 그 열이 hit-drop 버그.
"""
import sys, torch
sys.path.insert(0, "/home/chokwans99/PSD/ssd")
from ssd.engine.helpers.p2_tree import _arena_mask_pack

dev = "cuda:0"
PAGE = 64


def exec_mask(canvas, plen, gW, gW_max, glue_row, anc, sel_valid, W, f):
    """_pack_row_mask 로직 재현 → unpacked [W, canvas] uint8."""
    col = torch.arange(canvas, device=dev)
    plen_t = torch.tensor([plen], device=dev)
    gW_t = torch.tensor([gW], device=dev)
    m = (col.unsqueeze(0) < plen_t).expand(W, canvas).to(torch.uint8).clone()
    g_off = col.unsqueeze(0) - plen_t
    in_glue = (g_off >= 0) & (g_off < gW_t)
    g_idx = g_off.clamp(min=0, max=gW_max - 1)
    g_bits = glue_row.gather(1, g_idx.expand(W, canvas)) \
        * sel_valid.unsqueeze(1).to(torch.uint8)
    m = torch.where(in_glue.expand(W, canvas), g_bits, m)
    spec_off = g_off - gW_t
    in_spec = (spec_off >= 0) & (spec_off < f * W) if f else \
        torch.zeros(1, canvas, dtype=torch.bool, device=dev)
    if f:
        a_bits = ((anc.unsqueeze(1) >> spec_off.clamp(
            min=0, max=max(f * W - 1, 0))) & 1).to(torch.uint8)
        m = torch.where(in_spec.expand(W, canvas), a_bits, m)
    lane = torch.arange(W, device=dev)
    self_col = plen_t + gW_t + f * W + lane.unsqueeze(1)
    is_self = col.unsqueeze(0) == self_col
    ones = torch.ones(W, dtype=torch.uint8, device=dev)
    m = torch.where(is_self, ones.unsqueeze(1).expand(W, canvas), m)
    return m


W = 10
mismatch = 0
for trial in range(200):
    g = torch.Generator(device=dev).manual_seed(trial)
    f = trial % 3
    K_glue = int(torch.randint(1, 5, (1,), generator=g, device=dev))
    context_len = int(torch.randint(80, 200, (1,), generator=g, device=dev))
    p0 = (context_len + PAGE - 1) // PAGE
    canvas = (p0 + 1) * PAGE
    gW = K_glue + 1
    plen = context_len - gW - W
    if plen < 0:
        continue
    # arena inputs
    glue_sel = (torch.rand(W, gW, generator=g, device=dev) > 0.3) \
        .to(torch.uint8)
    spec_w = (f + 1) * W
    anc = torch.randint(0, 1 << min(spec_w, 30), (W,), generator=g,
                        device=dev, dtype=torch.int64)
    # anc의 현재-라운드 비트([f·W, (f+1)·W))는 0 (조상은 과거만)
    if f < 1:
        pass
    mask_hi = ((1 << (f * W)) - 1) if f else 0
    anc = anc & mask_hi
    sel_valid = torch.ones(W, dtype=torch.bool, device=dev)
    packed, _ = _arena_mask_pack(f, W, K_glue, context_len, glue_sel,
                                 anc, sel_valid, dev)
    cols = context_len + f * W
    bits = torch.zeros(W * cols + ((-W * cols) % 8), dtype=torch.uint8,
                       device=dev)
    wbits = (1 << torch.arange(8, device=dev)).to(torch.uint8)
    unp = ((packed.unsqueeze(1) >> torch.arange(8, device=dev)) & 1) \
        .to(torch.uint8).reshape(-1)[:W * cols].view(W, cols)
    # exec mask (canvas ≥ cols); 앞 cols열만 비교, 나머지는 0이어야
    glue_row_full = torch.zeros(W, gW, device=dev, dtype=torch.uint8)
    em = exec_mask(canvas, plen, gW, gW, glue_sel, anc, sel_valid, W, f)
    em_cols = em[:, :cols]
    em_tail = em[:, cols:]
    d_core = (em_cols != unp).sum().item()
    d_tail = (em_tail != 0).sum().item()
    if d_core or d_tail:
        mismatch += 1
        if mismatch <= 3:
            diffcols = (em_cols != unp).any(0).nonzero().flatten()[:10]
            print(f"trial {trial} f={f} K_glue={K_glue} ctx={context_len}"
                  f" core_diff={d_core} tail_diff={d_tail} "
                  f"cols={diffcols.tolist()}")
print(f"\n{mismatch}/200 trials mismatch")
