"""Packet parity against the legacy path, including padding and strided views."""
import unittest

import torch

from ssd.engine.helpers.fused_response_topology import prepare_response_topology_packet
from ssd.engine.helpers.fused_tree_response import attach_response_packet
from ssd.engine.helpers.response_topology import prepare_response_topology


def make_views(count, nodes, stride, dtype):
    views = {
        key: torch.randint(-1, 32000, (count, nodes * stride),
                           dtype=dtype, device='cuda')[:, ::stride]
        for key in ('tok', 'parent_local', 'sib_order', 'parent_q_ref')
    }
    for key in ('valid', 'u_valid'):
        views[key] = torch.randint(0, nodes + 1, (count * stride,),
                                  dtype=dtype, device='cuda')[::stride]
    return views


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FusedResponseTopologyTests(unittest.TestCase):
    @torch.inference_mode()
    def test_legacy_bitwise_parity_and_rng(self):
        for count in (0, 1, 16, 32, 128):
            for nodes, cap, width in ((3, 1, 1), (3, 3, 5), (3, 7, 7),
                                      (15, 15, 15), (31, 31, 31)):
                for stride in (1, 2):
                    for dtype in (torch.int32, torch.int64):
                        views = make_views(count, nodes, stride, dtype)
                        for phase in (1, 2):
                            with self.subTest(count=count, nodes=nodes, cap=cap,
                                              width=width, stride=stride,
                                              dtype=dtype, phase=phase):
                                rng = torch.cuda.get_rng_state().clone()
                                old = prepare_response_topology(views, cap)
                                old_packet = attach_response_packet(old, phase, width)
                                new, packet = prepare_response_topology_packet(
                                    views, cap, phase, width)
                                self.assertTrue(torch.equal(old.device_rows, new.device_rows))
                                self.assertTrue(torch.equal(old.host_rows, new.host_rows))
                                self.assertEqual(old.integer_rows, new.integer_rows)
                                self.assertTrue(torch.equal(old_packet, packet))
                                self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))

    @torch.inference_mode()
    def test_snapshot_and_packet_do_not_alias_executor(self):
        from test_response_topology import views as valid_views
        source = valid_views('cuda')
        snapshot, packet = prepare_response_topology_packet(source, 5, 2, 7)
        original_packet = packet.clone()
        for tensor in source.values():
            tensor.fill_(-1)
        gpu, cpu, parsed = snapshot.select(0, 10)
        self.assertEqual(parsed['tok'], (1, 2, 3, 0, 0))
        self.assertTrue(torch.equal(gpu.cpu(), cpu))
        self.assertTrue(torch.equal(packet, original_packet))

    def test_invalid_packet_shapes_rejected(self):
        source = make_views(1, 3, 1, torch.int64)
        for cap, width in ((0, 3), (3, 2)):
            with self.subTest(cap=cap, width=width), self.assertRaises(ValueError):
                prepare_response_topology_packet(source, cap, 1, width)


if __name__ == '__main__':
    unittest.main()
