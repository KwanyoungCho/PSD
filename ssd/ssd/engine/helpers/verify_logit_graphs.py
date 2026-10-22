"""Optional exact-row head graphs; do not change proposal or verifier math.

The split model graphs otherwise mix captured all-reduces with eager head
gathers. Full-width requests fuse each head into its model graph. Short
padded trees use exact-real-width auxiliary head graphs, preserving trimming.
Together these permit NCCL_GRAPH_MIXING_SUPPORT=0 without overlapping eager
TP collectives with an outstanding captured TP graph.

Opt-in profile: SSD_VERIFY_GRAPH_LOGITS=1, NCCL_GRAPH_MIXING_SUPPORT=0,
B=1, SSD_DUET_EXIT_REPLICA=0, SSD_PROXY_STREAM=0, full TP head gather.
Do not disable NCCL mixing by itself in an unmodified engine. See:
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-graph-mixing-support
Default behavior is unchanged. Only the target split-verify path is optimized;
eager prefill/diagnostic fallbacks synchronize at safe communicator boundaries.
"""
import os


def enabled(config, *, is_draft=False):
    value=os.environ.get('SSD_VERIFY_GRAPH_LOGITS','0')
    if value not in ('0','1'):
        raise ValueError('SSD_VERIFY_GRAPH_LOGITS must be 0 or 1')
    if value=='0' or is_draft:
        return False
    # Persistent graph outputs must not be overwritten while a separate
    # proxy stream is still reading them. Keep such modes on the native path.
    supported=(config.max_num_seqs==1
            and not getattr(config,'duet_exit_replica',False)
            and not getattr(config,'duet_exit_topm_gather',False)
            and os.environ.get('SSD_PROXY_STREAM','0')!='1')
    if not supported and no_mixing():
        raise ValueError('Non-mixing verify graphs require B1, full TP head, and no side proxy stream')
    return supported


def no_mixing():
    return os.environ.get('NCCL_GRAPH_MIXING_SUPPORT','1')=='0'


def synchronize_eager():
    """Conservative boundary for prefill or a diagnostic eager fallback."""
    if no_mixing():
        import torch
        torch.cuda.synchronize()


def capture(runner, pre_body, post_body, hidden, residual, outputs, rows):
    """Capture each original split body with its original head appended.

    Uses the original norm and compute_logits, including the original TP
    gather and vocabulary ordering. The only new buffers are persistent
    output copies. No probabilities, logits, or model weights are modified.
    """
    import torch
    from ssd.utils.context import get_context,set_context
    old=get_context()
    result={}
    try:
        pool=None
        for kind in ('exit','final'):
            out=(torch.empty((rows,runner.hf_config.vocab_size),
                    dtype=runner.hf_config.torch_dtype,device=runner.device)
                 if runner.rank==0 else None)
            def body():
                set_context(**vars(old))
                (pre_body if kind=='exit' else post_body)()
                set_context(is_prefill=False)
                x=(runner.model.model.norm(hidden[:rows]+residual[:rows],None)
                   if kind=='exit' else outputs[:rows])
                logits=runner.model.compute_logits(x,last_only=False)
                if out is not None:
                    out.copy_(logits)
            # Warm the same collective before capturing it on every rank.
            body()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool):
                body()
            if pool is None:pool=graph.pool()
            result[kind]=(graph,out)
    finally:
        set_context(**vars(old))
    return result


def output(pair, kind, *, chain=False):
    out=pair[kind][1]
    return out.unsqueeze(0) if chain and out is not None else out
