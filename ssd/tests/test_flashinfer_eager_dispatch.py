"""CPU-only planner regression: execution mode must not change KV layout."""
import ast
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys
import unittest
from unittest.mock import Mock,patch

SOURCE=Path(__file__).resolve().parents[1]/'ssd/engine/model_runner.py'


class Vector(list):
    def __getitem__(self,key):
        value=super().__getitem__(key)
        return Vector(value) if isinstance(key,slice) else value

    def __sub__(self,other):return Vector(a-b for a,b in zip(self,other))

    def to(self,dtype):return self


def extract_planner(path):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_plan_flashinfer_dense_attention')
    namespace={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_plan_flashinfer_dense_attention'],namespace


class FlashInferEagerDispatch(unittest.TestCase):
    def setUp(self):
        self.planner,self.namespace=extract_planner(SOURCE)
        self.backend=SimpleNamespace(cg_cur='stale graph',plan_paged=Mock(),
            plan_decode=Mock(),plan_prefill_ragged=Mock())
        fake=ModuleType('ssd.layers.fi_attn')
        fake.use_flashinfer_attention=Mock(return_value=True)
        fake.get_fi_backend=Mock(return_value=self.backend)
        self.fake=fake
        self.patch=patch.dict(sys.modules,{'ssd.layers.fi_attn':fake});self.patch.start()
        self.addCleanup(self.patch.stop)
        self.context=SimpleNamespace(block_tables=object(),context_lens=Vector([133]),
            cu_seqlens_q=None,cu_seqlens_k=None)
        self.namespace.update(get_context=lambda:self.context,torch=SimpleNamespace(int32='int32'))
        self.runner=SimpleNamespace(device='cpu-contract',enforce_eager=True,block_size=256,
            hf_config=SimpleNamespace(torch_dtype='bf16'),
            _first_attn_module=lambda:SimpleNamespace(num_heads=32,num_kv_heads=4,head_dim=128,scale=.125))

    def test_eager_decode_without_cumulative_k(self):
        self.planner(self.runner,False)
        self.backend.plan_decode.assert_called_once()
        args=self.backend.plan_decode.call_args.args
        self.assertIs(args[0],self.context.block_tables)
        self.assertIs(args[1],self.context.context_lens)
        self.backend.plan_paged.assert_not_called()
        self.assertIsNone(self.backend.cg_cur)

    def test_eager_verify_uses_context_lengths(self):
        self.context.cu_seqlens_q=Vector([0,6])
        self.planner(self.runner,False)
        self.backend.plan_paged.assert_called_once()
        args=self.backend.plan_paged.call_args.args
        self.assertIs(args[0],self.context.cu_seqlens_q)
        self.assertIs(args[2],self.context.context_lens)
        self.backend.plan_decode.assert_not_called()

    def test_non_eager_decode_layout_unchanged(self):
        self.runner.enforce_eager=False
        self.planner(self.runner,False)
        self.backend.plan_decode.assert_called_once()

    def test_non_eager_verify_layout_unchanged(self):
        self.runner.enforce_eager=False;self.context.cu_seqlens_q=Vector([0,6])
        self.planner(self.runner,False)
        self.assertIs(self.backend.plan_paged.call_args.args[2],self.context.context_lens)

    def test_paged_prefill_uses_cumulative_k(self):
        self.context.cu_seqlens_q=Vector([0,3,7]);self.context.cu_seqlens_k=Vector([0,5,14])
        self.planner(self.runner,True)
        self.assertEqual(self.backend.plan_paged.call_args.args[2],[5,9])
        self.backend.plan_decode.assert_not_called()

    def test_ragged_prefill_unchanged(self):
        self.context.block_tables=None
        self.context.cu_seqlens_q=Vector([0,5]);self.context.cu_seqlens_k=Vector([0,5])
        self.planner(self.runner,True)
        self.backend.plan_prefill_ragged.assert_called_once()
        self.assertIs(self.backend.plan_prefill_ragged.call_args.args[1],self.context.cu_seqlens_k)

    def test_eager_prefill_and_non_eager_prefill_select_same_layout(self):
        self.context.cu_seqlens_q=Vector([0,3]);self.context.cu_seqlens_k=Vector([0,5])
        self.planner(self.runner,True);first=self.backend.plan_paged.call_args
        self.backend.plan_paged.reset_mock();self.runner.enforce_eager=False
        self.planner(self.runner,True)
        self.assertEqual(first,self.backend.plan_paged.call_args)

    def test_other_attention_backend_is_untouched(self):
        self.fake.use_flashinfer_attention.return_value=False
        self.runner._first_attn_module=Mock(side_effect=AssertionError('Should not inspect model'))
        self.planner(self.runner,False)
        self.fake.get_fi_backend.assert_not_called()

    def test_eager_safety_boundary_is_at_run_entry_not_in_layout_selection(self):
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelRunner')
        run=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_model')
        first=run.body[0]
        self.assertIsInstance(first,ast.If)
        self.assertEqual(ast.unparse(first.test),'is_prefill or self.enforce_eager')


if __name__=='__main__':unittest.main()
