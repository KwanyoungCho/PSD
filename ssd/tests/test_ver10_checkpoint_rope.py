"""No model weights or GPU needed for the guarded Llama-2 buffer loader."""
import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn
from ssd.layers.rotary_embedding import RotaryEmbedding
from ssd.utils.checkpoint_rope import validate_frequency,redundant_rope_buffer
from ssd.utils.loader import load_safetensors_model


class CheckpointRope(unittest.TestCase):
    def fixture(self):
        cfg=dict(model_type='llama',rope_scaling=None,num_hidden_layers=1,
                 hidden_size=8,num_attention_heads=1,rope_theta=10000.)
        model=nn.Module();model.model=nn.Module();model.model.layers=nn.ModuleList([nn.Module()])
        attn=model.model.layers[0];attn.self_attn=nn.Module()
        attn.self_attn.rotary_emb=RotaryEmbedding(8,8,16,10000.)
        model.register_parameter('weight',nn.Parameter(torch.zeros(2)))
        expected=1/(10000.**(torch.arange(0,8,2,dtype=torch.float32)/8))
        key='model.layers.0.self_attn.rotary_emb.inv_freq'
        return cfg,model,expected,key

    def test_exact_and_legacy_roundtrip(self):
        _,_,e,_=self.fixture()
        self.assertEqual(validate_frequency(e,e),'exact_float32')
        self.assertEqual(validate_frequency(e.half(),e),'fp16_roundtrip')
        for x in (e+.01,e[:-1],torch.full_like(e,float('nan'))):
            with self.assertRaises(ValueError):validate_frequency(x,e)

    def test_real_loader_keeps_learned_weights_and_runtime_cache(self):
        cfg,model,e,key=self.fixture();before=model.model.layers[0].self_attn.rotary_emb.cos_sin_cache.clone()
        weights={key:e.half(),'weight':torch.tensor([2.,3.])}
        reader=SimpleNamespace(keys=lambda:list(weights),get_tensor=lambda name:weights[name])
        with patch('ssd.utils.loader.glob',return_value=['fixture.safetensors']), \
             patch('ssd.utils.loader.safe_open',return_value=contextlib.nullcontext(reader)), \
             patch('pathlib.Path.read_text',return_value=json.dumps(cfg)), \
             contextlib.redirect_stdout(io.StringIO()):
            load_safetensors_model(model,'/model',{})
        torch.testing.assert_close(model.weight,weights['weight'])
        torch.testing.assert_close(model.model.layers[0].self_attn.rotary_emb.cos_sin_cache,before)

    def test_scaled_wrong_architecture_layer_and_unknown_not_ignored(self):
        cfg,model,e,key=self.fixture();reader=SimpleNamespace(get_tensor=lambda _:e)
        self.assertFalse(redundant_rope_buffer(model,'/model','unknown.weight',reader))
        for change in (dict(rope_scaling={'factor':2}),dict(model_type='qwen2'),dict(num_hidden_layers=0),dict(head_dim=4)):
            with patch('pathlib.Path.read_text',return_value=json.dumps(dict(cfg,**change))),self.assertRaises(ValueError):
                redundant_rope_buffer(model,'/model',key,reader)


if __name__=='__main__':unittest.main()
