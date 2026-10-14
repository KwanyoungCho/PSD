"""Startup-only resolved settings; no inference-step synchronization."""
import importlib.metadata
import json
import os
import platform
import sys
import torch
from ssd.utils.debug import debug_tokens_enabled


def runtime_manifest(config, hf, rank, is_draft, target_tp, device):
    from ssd.layers.fi_attn import attention_backend
    packages = {}
    for name in ('torch','flashinfer-python','sgl-kernel','transformers','sentencepiece'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tp = 1 if is_draft else max(1, target_tp)
    result = dict(rank=rank, role='draft' if is_draft else 'target',
        python=platform.python_version(), python_abi=sys.implementation.cache_tag,
        optimize=sys.flags.optimize, debug_tokens=debug_tokens_enabled(),
        device=str(device), device_name=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),
        attention_backend=attention_backend(device), packages=packages,
        dtype=str(hf.torch_dtype), vocab_size=hf.vocab_size,
        head_dim=hf.head_dim, local_q_heads=hf.num_attention_heads//tp,
        local_kv_heads=hf.num_key_value_heads//tp, model_tp=tp,
        target_tp=config.num_gpus-1 if config.draft_async else config.num_gpus,
        model=config.model, draft=config.draft, max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs, speculate_k=config.speculate_k,
        async_fan_out=config.async_fan_out, fan_out_list=config.fan_out_list,
        mq_len=getattr(config,'MQ_LEN',None), exit_layer=config.duet_exit_layer,
        k1=config.duet_phase1_k, k2=config.duet_phase2_k,
        p2_budget=config.duet_p2_budget,
        workspace_mib={k:os.getenv(k,str(v)) for k,v in (
            ('SSD_BATCH_TREE_WORKSPACE_MB',256),('SSD_TREE_EXEC_WORKSPACE_MB',128),
            ('SSD_P1_TREE_EXEC_WORKSPACE_MB',128))})
    if config.duet_enabled:
        result['policy'] = dict(root='complement', normalization='full',
            overlap_mix=.25, candidate_temperature='request temperature; T0 soft score',
            requested_expansion=os.getenv('SSD_TREE_EXPANSION_POLICY','q_path'),
            p1_allocation=config.duet_p1_tree_allocation_policy,
            physical_width='reported per executor in [tree-policy-resolved]')
    return result


def log_runtime_manifest(*args, **kwargs):
    print('[runtime-manifest] '+json.dumps(runtime_manifest(*args,**kwargs),
                                         sort_keys=True), flush=True)
