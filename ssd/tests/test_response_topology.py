from unittest import TestCase, main, skipUnless
import torch
from ssd.engine.helpers.response_topology import prepare_response_topology
from ssd.engine.helpers.p2_tree import pack_tree_ints


def views(device):
    return {k: torch.tensor(v, device=device) for k, v in dict(
        tok=[[1,2,3],[4,5,6]], valid=[3,2], u_valid=[2,1],
        parent_local=[[-1,-1,0],[-1,-1,-1]], sib_order=[[0,1,0],[0,1,2]],
        parent_q_ref=[[0,0,1],[0,0,0]]).items()}


class ResponseTopologyTests(TestCase):
    def check_equivalence(self, device):
        source=views(device)
        for cap in (3,5):
            snapshot=prepare_response_topology(source,cap)
            for root in range(2):
                gpu,cpu,parsed=snapshot.select(root,10)
                expected=pack_tree_ints(source,root,cap)
                self.assertTrue(torch.equal(gpu,expected))
                self.assertTrue(torch.equal(cpu,expected.cpu()))
                self.assertEqual(parsed['valid'],int(source['valid'][root]))
                self.assertIsInstance(parsed['tok'][0],int)

    def test_exact_padding_and_fields(self):
        self.check_equivalence('cpu')

    @skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_pack_and_readback(self):
        self.check_equivalence('cuda:0')

    def test_snapshot_does_not_alias_executor_buffers(self):
        source=views('cpu');snapshot=prepare_response_topology(source,3)
        for tensor in source.values():tensor.fill_(-1)
        gpu,cpu,parsed=snapshot.select(0,10)
        self.assertEqual(parsed['tok'],(1,2,3))
        self.assertTrue(torch.equal(gpu,cpu))

    def test_malformed_hit_is_still_rejected(self):
        for key in ('valid','u_valid','tok','parent_local','sib_order','parent_q_ref'):
            with self.subTest(key=key):
                source=views('cpu')
                if key in ('valid','u_valid'):source[key][0]=99
                else:source[key][0,0]=99
                snapshot=prepare_response_topology(source,3)
                with self.assertRaises(RuntimeError):snapshot.select(0,10)
        with self.assertRaises(IndexError):prepare_response_topology(views('cpu'),3).select(2,10)


if __name__=='__main__':main()
