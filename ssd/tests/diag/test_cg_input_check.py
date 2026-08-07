"""리뷰 2단계: replay-직전 범위검사의 인위-퇴화 통과 검증 (CPU,
엔진 불필요). 완전-퇴화(-1 position / 음수 page 유래 slot -256..-2 /
token OOB / ctx<=0)가 각각 반드시 RuntimeError를 내야 한다."""
import unittest
import torch
from ssd.engine.helpers.cudagraph_helpers import cg_input_range_check


def ok_inputs(W=10):
    return dict(input_ids=torch.zeros(W, dtype=torch.int64),
                positions=torch.arange(W),
                slot_mapping=torch.arange(W, dtype=torch.int32),
                context_lens=torch.tensor([100]),
                vocab_size=32000, rope_len=2048,
                n_kv_slots=100000, step=0)


class TestCgInputCheck(unittest.TestCase):
    def test_valid_passes(self):
        cg_input_range_check(**ok_inputs())

    def test_padding_slot_minus1_allowed(self):
        a = ok_inputs()
        a["slot_mapping"][3] = -1
        a["active_mask"] = torch.ones(10, dtype=torch.bool)
        a["active_mask"][3] = False
        cg_input_range_check(**a)

    def test_active_slot_minus1_fires(self):
        # 규약: active_mask가 '명시'된 active lane은 -1 불허.
        # (mask 미제공 기본은 -1=정상 padding — run_fi glue 경로의
        # 기존 pad 규약과 정합; test_padding_slot_minus1_allowed)
        a = ok_inputs()
        a["slot_mapping"][3] = -1
        a["active_mask"] = torch.ones(
            a["slot_mapping"].numel(), dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "slot\\[min=-1"):
            cg_input_range_check(**a)

    def test_negative_position_fires(self):
        a = ok_inputs()
        a["positions"] = a["positions"].clone()
        a["positions"][5] = -1
        with self.assertRaisesRegex(RuntimeError, "pos\\[min=-1"):
            cg_input_range_check(**a)

    def test_negative_page_slot_fires(self):
        # page=-1 유래 slot = -256..-2 (기존 == -1 가드가 못 막던 대역)
        a = ok_inputs()
        a["slot_mapping"][2] = -256 + 5
        with self.assertRaisesRegex(RuntimeError, "slot\\[min=-251"):
            cg_input_range_check(**a)

    def test_token_oob_fires(self):
        a = ok_inputs()
        a["input_ids"][0] = 32000
        with self.assertRaisesRegex(RuntimeError, "token\\["):
            cg_input_range_check(**a)

    def test_ctx_zero_fires(self):
        a = ok_inputs()
        a["context_lens"] = torch.tensor([-1])
        with self.assertRaisesRegex(RuntimeError, "ctx\\["):
            cg_input_range_check(**a)


if __name__ == "__main__":
    unittest.main()
