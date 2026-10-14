"""Review contracts runnable without a Blackwell device or model weights."""
import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
NS=SimpleNamespace

os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from transformers import LlamaConfig, Qwen2Config
from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.llm_engine import LLMEngine
from ssd.utils.debug import debug_tokens_enabled
from ssd.utils.workspace import workspace_bytes, planning_workspace


def make_config(**kwargs):
    def hf(path):
        if path == '/target':
            return LlamaConfig(num_hidden_layers=80, max_position_embeddings=4096,
                               torch_dtype=torch.bfloat16)
        return LlamaConfig(num_hidden_layers=22, max_position_embeddings=2048,
                           torch_dtype=torch.float32)
    with patch('ssd.config.os.path.isdir', return_value=True), \
         patch('ssd.config.AutoConfig.from_pretrained', side_effect=hf), \
         contextlib.redirect_stdout(io.StringIO()):
        values=dict(model='/target', draft='/draft', speculate=True,
            draft_async=True, jit_speculate=True, duet_enabled=True,
            speculate_k=12, duet_phase1_k=8, duet_phase2_k=4,
            duet_exit_layer=56, async_fan_out=3, duet_draft_fan_out=3,
            duet_p2_budget=15, max_model_len=2048)
        values.update(kwargs)
        cfg=Config(**values)
        draft=DraftRunner.create_draft_config(cfg)
    return cfg,draft


class ReviewContracts(unittest.TestCase):
    def test_original_fanout_and_target_exit_survive_draft_reconstruction(self):
        cfg,draft=make_config()
        for c in (cfg,draft):
            self.assertEqual(c.async_fan_out,3)
            self.assertEqual(c.fan_out_list,[3]*13)
            self.assertEqual(c.MQ_LEN,39)
            self.assertEqual(c.duet_proxy_fan_out,0)
            self.assertEqual(c.duet_proxy_total_budget,15)
            self.assertEqual(c.duet_exit_layer,56)
            self.assertEqual(c.target_hf_config.num_hidden_layers,80)
            self.assertEqual(c.max_model_len,2048)
            self.assertEqual(c.hf_config.torch_dtype,torch.bfloat16)
        self.assertEqual(draft.hf_config.num_hidden_layers,22)
        self.assertEqual(draft.runner_role,'draft')

    def test_invalid_contracts_are_errors(self):
        for kwargs in (dict(duet_p2_budget=None),dict(duet_p2_budget=0),
                       dict(duet_exit_layer=80),dict(duet_exit_layer=-1),
                       dict(speculate_k=13),dict(max_num_seqs=0),
                       dict(fan_out_list=[3]*12)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                make_config(**kwargs)
        make_config(duet_exit_layer=0)

    def test_sparse_fanout_is_valid(self):
        cfg,_=make_config(fan_out_list=[3,0]+[3]*11)
        self.assertEqual(cfg.MQ_LEN,36)

    def test_optimized_python_keeps_contracts(self):
        script='from test_blackwell_review import make_config; make_config(duet_exit_layer=80)'
        env=dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(__file__),
            os.path.dirname(os.path.dirname(__file__)),os.environ.get('PYTHONPATH','')]))
        for flags in ([],['-O']):
            run=subprocess.run([sys.executable,*flags,'-c',script],env=env,
                               capture_output=True,text=True,timeout=60)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('duet_exit_layer must be in [0, 80)',run.stderr)

    def test_debug_tokens_default_off_independent_of_python_debug(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertFalse(debug_tokens_enabled())
            os.environ['SSD_DEBUG_TOKENS']='1'
            self.assertTrue(debug_tokens_enabled())

    def test_speculative_debug_switch_preserves_counts_and_rng(self):
        from test_output_accounting import TestOutputAccounting
        states=[]
        for enabled in ('0','1'):
            torch.manual_seed(812)
            with patch.dict(os.environ,SSD_DEBUG_TOKENS=enabled), contextlib.redirect_stdout(io.StringIO()):
                TestOutputAccounting().test_decode_counts_committed_tokens_after_cap_or_eos()
            states.append(torch.get_rng_state().clone())
        self.assertTrue(torch.equal(*states))

    def test_ar_debug_off_never_decodes_and_same_tokens(self):
        from ssd.engine.step import AutoRegressiveStep
        committed=[]
        step=AutoRegressiveStep(NS(postprocess=lambda seqs,tokens,pref:committed.append(tokens)),
            NS(call=lambda *args:[3,4]),object())
        from unittest.mock import Mock
        decoded=Mock(return_value='text')
        for enabled in ('0','1'):
            with patch.dict(os.environ,SSD_DEBUG_TOKENS=enabled), \
                 patch('ssd.engine.step.decode_tokens',decoded), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(step.step([object(),object()],False),2)
        decoded.assert_called_once()
        self.assertEqual(committed,[[3,4],[3,4]])

    def test_pack_rounding_boundary_preserves_ids_and_zero_mask(self):
        from ssd.engine.helpers.p2_tree import pack_piv,unpack_piv
        # Scores on either side of one logarithmic bin boundary are close
        # in FP32 but can legally quantize into adjacent 16-bit codes.
        boundary=torch.tensor(10**(-6+6*(50000.5)/65535),dtype=torch.float32)
        a=torch.nextafter(boundary,torch.tensor(0.))
        z=torch.nextafter(boundary,torch.tensor(float('inf')))
        ids=torch.tensor([128255,9,12]);values=torch.stack([a,torch.tensor(0.),z])
        packed=pack_piv(ids,values);actual,_=unpack_piv(packed)
        self.assertTrue(torch.equal(ids,actual))
        codes=(packed>>32)&65535
        self.assertLessEqual(int((codes[0]-codes[2]).abs()),1)
        self.assertEqual((values>0).tolist(),[True,False,True])
        torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-7)

    def test_workspace_budget_and_original_failure(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(workspace_bytes('TEST_WORKSPACE',256),256*2**20)
        for value in ('0','-1','1.5','bad'):
            with patch.dict(os.environ,{'TEST_WORKSPACE':value}),self.assertRaises(ValueError):
                workspace_bytes('TEST_WORKSPACE',256)
        workspace=torch.empty(128,dtype=torch.uint8)
        with self.assertRaisesRegex(RuntimeError,'197132288') as error:
            with planning_workspace('TEST_WORKSPACE',workspace,batch=2,width=13):
                raise RuntimeError('required bytes: 197132288')
        self.assertIsInstance(error.exception.__cause__,RuntimeError)
        self.assertIn("'batch': 2",str(error.exception))

    def test_dead_target_worker_aborts_draft_as_well(self):
        class Worker:
            def __init__(self,alive,exitcode=None):self.alive=alive;self.exitcode=exitcode
            def is_alive(self):return self.alive
            def terminate(self):self.alive=False
            def join(self,timeout=None):pass
        engine=object.__new__(LLMEngine)
        engine._draft_watch_stop=NS(wait=lambda timeout:False,set=lambda:None)
        engine._exiting=False;engine.ps=[Worker(False,23)]
        engine.draft_ps=Worker(True)
        with patch('ssd.engine.llm_engine.os._exit',side_effect=SystemExit) as leave:
            with self.assertRaises(SystemExit):engine._watch_draft_process()
        self.assertFalse(engine.draft_ps.alive)
        leave.assert_called_once_with(23)

    def test_canonical_mass_does_not_depend_on_padding_strides(self):
        from ssd.engine.helpers.root_policy import vocabulary_mass
        torch.manual_seed(55)
        b,n,v=3,5,4096
        base=torch.rand(b,n,v)
        dense=torch.cat([base.flatten(1),base.new_zeros(b,1)],1)[:,:-1].view(b,n,v)
        compact=torch.cat([base.reshape(b*n,v),base.new_zeros(b*n,1)],1)[:,:v]
        self.assertNotEqual(dense.stride()[0],n*compact.stride()[0])
        a=vocabulary_mass(dense).reshape(b*n,1);z=vocabulary_mass(compact)
        torch.testing.assert_close(a,z,atol=0,rtol=0)

    def test_constructor_failure_terminates_own_workers_and_preserves_error(self):
        # Exercise __init__ itself: failure after workers launch, before atexit.
        cfg=Config.__new__(Config)
        cfg.model='/target';cfg.speculate=False;cfg.draft_async=False
        cfg.num_gpus=2;cfg.kvcache_block_size=256;cfg.speculate_k=1
        cfg.verbose=False;cfg.duet_enabled=False;cfg.max_num_seqs=1
        class Worker:
            def __init__(self): self.alive=False;self.terminated=False
            def start(self): self.alive=True
            def is_alive(self): return self.alive
            def terminate(self): self.terminated=True;self.alive=False
            def join(self,timeout=None): pass
        worker=Worker()
        ctx=SimpleNamespace(Event=lambda:None,Process=lambda **kw:worker)
        with patch('ssd.engine.llm_engine.Config',return_value=cfg), \
             patch('ssd.engine.llm_engine.fields',return_value=[]), \
             patch('ssd.engine.llm_engine.AutoTokenizer.from_pretrained',return_value=object()), \
             patch('ssd.engine.llm_engine.mp.get_context',return_value=ctx), \
             patch('ssd.engine.llm_engine.ModelRunner',side_effect=RuntimeError('first plan failure')):
            with self.assertRaisesRegex(RuntimeError,'first plan failure'):
                LLMEngine('/target')
        self.assertTrue(worker.terminated)


if __name__=='__main__':
    unittest.main()
