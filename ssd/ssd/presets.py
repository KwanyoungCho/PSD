"""Explicit experiment presets; never installed by monkey-patching Config.

Model locations are supplied by the caller. These are reproducible starting
settings, not claims of optimal parameters on every GPU/model.
"""


def duet_70b_paper_kwargs(draft, *, expansion='last', normalization='topm'):
    """BF16 70B target on TP2 + separate draft GPU; default workload is B=1.

    SamplingParams(temperature=.7, max_new_tokens=1024) is supplied separately.
    BF16 is explicit for both the LayerSkip and original 70B checkpoints.
    C=3 and zero expansion floors are the finalized uncalibrated tree design.
    They intentionally differ from historical C=2/floored speed measurements.
    """
    return dict(draft=draft, dtype='bfloat16', num_gpus=3, max_num_seqs=1,
        max_model_len=2048, max_num_batched_tokens=16384,
        gpu_memory_utilization=.7, speculate=True, draft_async=True,
        jit_speculate=True, duet_enabled=True, speculate_k=12,
        duet_phase1_k=8, duet_phase2_k=4, duet_exit_layer=56,
        async_fan_out=3, duet_draft_fan_out=3, duet_p2_budget=15,
        duet_proxy_top_k=28, duet_tree_root_count=10,
        duet_root_source='complement', duet_root_normalization=normalization,
        duet_root_overlap_mix=.25,
        duet_p1_tree_policy='on', duet_p2_tree_policy='on',
        duet_p1_roots_per_position=3, duet_p1_tree_forward_scale=1.5,
        duet_p1_tree_allocation_policy='dynamic', duet_tree_c_tensor=3,
        duet_tree_expansion=expansion,
        duet_p1_tree_max_nodes=12, duet_p2_tree_max_nodes=8,
        duet_p1_tree_verify_nodes=12, duet_p2_tree_verify_nodes=8,
        duet_tree_proxy_threshold=0., duet_tree_conf_threshold=0.,
        duet_p1_tree_start_threshold=0., duet_p1_tree_conf_threshold=0.)
