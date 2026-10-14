"""Page families must cover actual first-round roots near unaligned caps."""
import os,unittest
from types import SimpleNamespace
from unittest.mock import patch
from ssd.engine.helpers.batch_tree_common import capacity
from ssd.engine.helpers.batch_tree_warmup import page_buckets

class WarmupBounds(unittest.TestCase):
 def test_round_zero_and_clamped_target_families(self):
  with patch.dict(os.environ,{'SSD_TREE_EXEC_WARMUP':'all'}):
   for limit in (2010,2040,2048,4097):
    cfg=SimpleNamespace(max_model_len=limit,kvcache_block_size=256,max_blocks=(limit+255)//256)
    for first_round in (7,14,34,67):
     draft=set(page_buckets(cfg,draft=True,extra=first_round));target=set(page_buckets(cfg))
     for end in range(1,limit+1):
      self.assertIn(capacity((end+first_round+255)//256),draft)
      self.assertIn(min(capacity((end+255)//256),cfg.max_blocks),target)
 def test_existing_debug_disable(self):
  with patch.dict(os.environ,{'SSD_TREE_EXEC_WARMUP':'0'}):
   self.assertEqual(page_buckets(SimpleNamespace()),[])
if __name__=='__main__':unittest.main()
