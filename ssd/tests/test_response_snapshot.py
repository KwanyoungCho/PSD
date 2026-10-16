import copy
from dataclasses import FrozenInstanceError
import unittest
import torch
from ssd.engine.helpers.response_snapshot import parse_host_response
from ssd.engine.speculator_async import extend_seqs_for_verify
from ssd.engine.sequence import Sequence


class NeverRead:
    def tolist(self):
        raise AssertionError('Unexpected second tensor readback')


class ResponseSnapshot(unittest.TestCase):
    def test_existing_wire_and_mutable_packet_do_not_alias_snapshot(self):
        packet=[1,2,3,11,12,13,14,3,2,1,11,12,13,0]
        snapshot=parse_host_response(packet,1,4,7)
        self.assertEqual(snapshot.cache_hits,(1,))
        self.assertEqual(snapshot.phase_source,(2,))
        self.assertEqual(snapshot.valid_k,(3,))
        self.assertEqual(snapshot.tokens,((11,12,13,14),))
        self.assertEqual(snapshot.tree_ints,((3,2,1,11,12,13,0),))
        packet[:]=[-1]*len(packet)
        self.assertEqual(snapshot.tokens[0][0],11)
        with self.assertRaises(FrozenInstanceError):snapshot.valid_k=(0,)

    def test_ragged_and_uniform_extension_matches_legacy(self):
        for widths in ((3,),(1,),(0,),(3,1),(1,3),(2,2)):
            with self.subTest(widths=widths):
                b=len(widths);w=4
                tokens=torch.arange(b*w,dtype=torch.int64).reshape(b,w)+20
                valid=torch.tensor(widths)
                seqs=[Sequence([1,2,3+i]) for i in range(b)]
                for seq in seqs:seq.num_draft_cached_tokens=2
                legacy=copy.deepcopy(seqs)
                packet=[1]*b+[1]*b+list(widths)+tokens.flatten().tolist()
                snapshot=parse_host_response(packet,b,w)
                old=extend_seqs_for_verify(legacy,tokens,valid,3)
                new=extend_seqs_for_verify(seqs,NeverRead(),NeverRead(),3,snapshot)
                self.assertEqual(old,new)
                for a,z in zip(legacy,seqs):
                    for key in ('token_ids','num_tokens','last_token','num_draft_cached_tokens','verify_valid_k'):
                        self.assertEqual(getattr(a,key),getattr(z,key))

    def test_invalid_dimensions_and_lengths_fail(self):
        for packet,b,w,extra in [([],0,1,0),([],1,0,0),([],2,1,1),
                                  ([1,1,5,0],1,1,0),([1,1,-1,0],1,1,0),([],1,1,0)]:
            with self.subTest(b=b,w=w,extra=extra),self.assertRaises(ValueError):
                parse_host_response(packet,b,w,extra)


if __name__=='__main__':unittest.main()
