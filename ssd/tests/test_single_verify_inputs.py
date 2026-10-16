import unittest
import torch
from ssd.engine.helpers.single_verify_inputs import SingleVerifyInputs
from ssd.engine.helpers.runner_helpers import prepare_decode_tensors_from_seqs
from ssd.engine.sequence import Sequence


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class SingleVerifyInputTests(unittest.TestCase):
    def test_exact_inputs_for_changed_pages_lengths_and_device_tokens(self):
        dev=torch.device('cuda:0')
        fast=SingleVerifyInputs(dev,41,128,16)
        for prefix in (0,5,15,16,111,517):
            for k in (1,3,10,39):
                seq=Sequence(list(range(prefix+k+1)))
                seq.num_cached_tokens=prefix
                # Deliberately non-contiguous physical pages.
                seq.block_table=[3+2*i for i in range((seq.num_tokens+15)//16)]
                expected=prepare_decode_tensors_from_seqs([seq],16,False,True,k)
                for gpu_tokens in (False,True):
                    source=torch.tensor(seq.token_ids[prefix:],dtype=torch.int64,device=dev) if gpu_tokens else None
                    actual=fast.prepare(seq,k,source)
                    for a,z in zip(actual[:4],expected):
                        self.assertEqual(a.dtype,z.dtype)
                        self.assertTrue(torch.equal(a,z),(prefix,k,gpu_tokens))
                    self.assertEqual(actual[5].tolist(),[0,k+1])
                    self.assertEqual(actual[4][0,:len(seq.block_table)].tolist(),seq.block_table)

    def test_reuse_host_staging_without_copy_race(self):
        dev=torch.device('cuda:0');fast=SingleVerifyInputs(dev,41,128,16)
        saved=[]
        for i in range(24):
            seq=Sequence([100+i,200+i,300+i,400+i]);seq.num_cached_tokens=0
            seq.block_table=[i+1]
            result=fast.prepare(seq,3)
            saved.append((result[0].clone(),result[2].clone(),list(seq.token_ids),i+1))
        for ids,slots,want,page in saved:
            self.assertEqual(ids.tolist(),want)
            self.assertEqual(slots.tolist(),[page*16+j for j in range(4)])

    def test_misalignment_and_missing_pages_still_fail(self):
        fast=SingleVerifyInputs(torch.device('cuda:0'),41,128,16)
        seq=Sequence([1,2,3,4]);seq.block_table=[2];seq.num_cached_tokens=1
        with self.assertRaises(AssertionError):fast.prepare(seq,3)
        seq.num_cached_tokens=0;seq.block_table=[]
        with self.assertRaises(ValueError):fast.prepare(seq,3)


if __name__=='__main__':unittest.main()
