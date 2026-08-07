"""단계0 — 실행기 mask 대조 (강화판, 고정 지침).

이전 판(복사 구현 대조)의 결함을 수정: **실제 executor의
`_pack_row_mask`를 호출**해 `wr._custom_mask_buf`에 기록된 packed
bit를 unpack, 같은 상태의 `_arena_mask_pack` 결과와 bit-대조한다.

커버: forward f=0..3 전부 / 완전-활성·부분-활성 lane(희소 트리) /
실제 block 크기 256과 미니 64 / page 마지막 길이 lpl ∈ {1,
PAGE−1, PAGE} / canvas 꼬리 전-0 검사.

실행: python -m unittest tests.diag.test_exec_mask_vs_arena
(성공 시 종료코드 0, 실패는 assertion)
"""
import unittest
import torch

try:
    import flashinfer                              # noqa: F401
    HAS_FI = True
except Exception:
    HAS_FI = False

from ssd.engine.helpers.p2_tree import _arena_mask_pack
from tests.test_p2_executor_parity import (
    TestExecutorModuleParity)


def unpack_mask_buf(wr, W, canvas, dev):
    """wr._custom_mask_buf(uint8 packbits little) → [W, canvas] uint8."""
    nbits = W * canvas
    nby = (nbits + 7) // 8
    pk = wr._custom_mask_buf[:nby]
    bits = ((pk.unsqueeze(1)
             >> torch.arange(8, device=dev)) & 1).to(torch.uint8)
    return bits.reshape(-1)[:nbits].view(W, canvas)


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestRealPackRowMaskVsArena(unittest.TestCase):
    """f=0..3 각각, run_once로 만든 실제 상태(_sel/arena/버퍼)에서
    ex._pack_row_mask 재호출 → arena _arena_mask_pack과 bit-대조."""

    def _harness(self):
        h = TestExecutorModuleParity()
        h.tearDown = lambda: None
        return h

    def _run_case(self, PAGE, ctx0, piv, gw=None, seed=7):
        dev = "cuda:0"
        h = self._harness()
        ex, cfg, _page, V = h._mk(dev, PAGE=PAGE)
        p0 = (ctx0 + PAGE - 1) // PAGE
        self.assertLessEqual(p0 + 1, ex.max_blocks)
        ex.prepare_bucket(p0)
        # 기본 입력 채움 (harness와 동일) 후 piv 덮어쓰기
        h._fill_inputs(ex, PAGE, ctx0)
        ex.in_root_piv.copy_(torch.tensor(piv, device=dev).float())
        if gw is not None:
            self.assertLessEqual(gw, ex.in_glue.shape[1])
            ex.in_glue.zero_()
            ex.in_glue[:, :gw] = 1
            ex.in_glue_w.fill_(gw)
            ex.in_prefix_len.fill_(ctx0 - gw - ex.W)
        gN = torch.Generator().manual_seed(33 + seed)
        ex.parity_noise = [
            (torch.empty(ex.W, V).exponential_(1, generator=gN)).to(dev)
            for _ in range(ex.F)]
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.model.cache.zero_()
        with torch.inference_mode():
            ex.run_once(p0)
        W = ex.W
        gw_real = int(ex.in_glue_w.item())
        plen = int(ex.in_prefix_len.item())
        K_glue = gw_real - 1
        context_len = plen + gw_real + W
        self.assertEqual(context_len, ctx0)
        partial_seen = False
        for f in range(ex.F):
            wr = ex.wrappers[p0][f]
            sel, sel_valid = ex._sel[f]
            if not bool(sel_valid.all()):
                partial_seen = True
            # ── 실제 메서드 재호출 (상태는 run 후에도 불변:
            # anc_bits/root는 삽입 시 고정, _sel[f]는 라운드 기록)
            with torch.inference_mode():
                ex._pack_row_mask(wr, f)
            canvas = wr._canvas_cols
            em = unpack_mask_buf(wr, W, canvas, dev)
            # ── arena 기준
            ar = ex.arena
            r_of = torch.where(
                sel_valid, ar.root.gather(0, sel.clamp(min=0)),
                torch.zeros_like(sel))
            glue_sel = ex.in_glue.index_select(0, r_of)[:, :gw_real]
            anc_sel = ar.anc_bits.index_select(0, sel.clamp(min=0)) \
                * sel_valid.long().unsqueeze(1)
            packed, _ = _arena_mask_pack(
                f, W, K_glue, context_len, glue_sel, anc_sel,
                sel_valid, dev)
            cols = context_len + f * W
            bits = ((packed.unsqueeze(1)
                     >> torch.arange(8, device=dev)) & 1) \
                .to(torch.uint8).reshape(-1)[:W * cols].view(W, cols)
            core_diff = int((em[:, :cols] != bits).sum())
            tail_diff = int((em[:, cols:] != 0).sum())
            self.assertEqual(
                core_diff, 0,
                f"PAGE={PAGE} ctx0={ctx0} f={f}: core {core_diff} "
                f"bit 불일치 (최초 열 "
                f"{(em[:, :cols] != bits).any(0).nonzero().flatten()[:5].tolist()})")
            self.assertEqual(
                tail_diff, 0,
                f"PAGE={PAGE} ctx0={ctx0} f={f}: canvas 꼬리 비-0")
        return partial_seen

    def test_full_lanes_mini_page64(self):
        piv = [.4, .2, .1, .06, .03, .01]
        self._run_case(64, 64 + 21, piv)

    def test_page_boundaries_mini(self):
        piv = [.4, .2, .1, .06, .03, .01]
        # 라운드 kv_len(lpl) 경계: ctx0=119→r0 lpl1 / 87→r2 lpl63 /
        # 88→r3 lpl64 (W=10, F=4, PAGE=64)
        for ctx0 in (119, 87, 88):
            self._run_case(64, ctx0, piv)

    def test_partial_and_inactive_lanes(self):
        # 뿌리 2개만 유효 → 노드 수 < F·W → 후반 라운드 부분-활성
        piv = [.4, .2, 0.0, 0.0, 0.0, 0.0]
        seen = self._run_case(64, 64 + 21, piv)
        self.assertTrue(seen, "부분-활성 lane 케이스가 발생하지 않음 — "
                              "piv 재조정 필요")

    def test_real_block_size_256(self):
        piv = [.4, .2, .1, .06, .03, .01]
        # 실 block 256: lpl 경계 {1, 255, 256} — W=10, F=4
        # r0 kv=ctx0+10: ctx0=247→lpl1(kv257) / 245→lpl255 / 246→lpl256
        for ctx0 in (256 + 21, 247, 245, 246):
            self._run_case(256, ctx0, piv)

    def test_glue_width_variants(self):
        piv = [.4, .2, .1, .06, .03, .01]
        for gw in (2, 3):
            self._run_case(64, 64 + 21, piv, gw=gw)


if __name__ == "__main__":
    unittest.main()
