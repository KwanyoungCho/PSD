"""Validate redundant, nonlearned Llama-2 RoPE buffers at model loading only.

The runtime regenerates these frequencies from config. Some official older
checkpoints contain FP16-rounded copies. Never ignore arbitrary unknown keys
or silently accept a scaled/different RoPE convention.
"""
import json
from pathlib import Path
import re
import torch

_KEY=re.compile(r'model\.layers\.(\d+)\.self_attn\.rotary_emb\.inv_freq')


def validate_frequency(value, expected):
    value=value.float().cpu()
    if value.shape!=expected.shape or not torch.isfinite(value).all():
        raise ValueError('Malformed checkpoint RoPE frequency buffer')
    if torch.equal(value,expected):return 'exact_float32'
    if torch.equal(value,expected.half().float()):return 'fp16_roundtrip'
    raise ValueError('Checkpoint RoPE frequency does not match the model config')


def redundant_rope_buffer(model, path, name, reader):
    match=_KEY.fullmatch(name)
    if not match:return False
    cfg=json.loads((Path(path)/'config.json').read_text())
    if cfg.get('model_type')!='llama' or cfg.get('rope_scaling') is not None:
        raise ValueError('Redundant RoPE buffer supported only for unscaled Llama checkpoints')
    if not 0<=int(match[1])<cfg['num_hidden_layers']:
        raise ValueError('RoPE buffer layer index outside model')
    dim=cfg.get('head_dim') or cfg['hidden_size']//cfg['num_attention_heads']
    module=model.get_submodule(name.rsplit('.',1)[0])
    from ssd.layers.rotary_embedding import RotaryEmbedding
    if not isinstance(module,RotaryEmbedding) or module.head_size!=dim:
        raise ValueError('Checkpoint RoPE buffer incompatible with runtime module')
    expected=1/(cfg.get('rope_theta',10000.)**(
        torch.arange(0,dim,2,dtype=torch.float32,device='cpu')/dim))
    kind=validate_frequency(reader.get_tensor(name),expected)
    print(f'[checkpoint-rope] validated redundant {name}: {kind}',flush=True)
    return True
