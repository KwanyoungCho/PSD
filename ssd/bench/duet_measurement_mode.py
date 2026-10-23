"""Explicit benchmark modes, configured before importing the GPU runtime.

The generic library still honors its caller's environment. The measurement
entry point resets instrumentation so a stale shell cannot enable it silently.
"""
import os

PROFILE_FLAGS = ('SSD_PROFILE_DUET', 'SSD_PROFILE_DUET_DETAIL',
                 'SSD_PROFILE', 'SSD_PROFILE_DRAFT', 'SSD_PROFILE_TARGET')


def configure_measurement(mode='performance'):
    if mode not in ('performance', 'timeline'):
        raise ValueError('measurement mode must be performance|timeline')
    # Only timeline uses CUDA events / per-forward probes. Measurement always
    # uses existing whole-step clocks; selection/filtering happens afterward.
    for name in PROFILE_FLAGS:
        os.environ[name] = '1' if mode == 'timeline' and name in ('SSD_PROFILE_DUET', 'SSD_PROFILE_DUET_DETAIL') else '0'
    os.environ['SSD_DEBUG_TOKENS'] = '0'
    os.environ['SSD_CG_INPUT_CHECK'] = '0'
    os.environ.pop('SSD_CG_MASK_PERTURB', None)
    os.environ['SSD_TREE_EXEC'] = '1'
    os.environ['SSD_TREE_EXEC_WARMUP'] = 'all'
    os.environ['SSD_P1_TREE_EXEC_WARMUP'] = 'all'
    os.environ['SSD_VERIFY_GRAPH_LOGITS'] = '1'
    os.environ['SSD_ASYNC_PROXY_SEND'] = '1'
    # Release defaults, while retaining explicit A/B and opt-in settings.
    os.environ.setdefault('SSD_SCOPED_DEFAULT_DEVICE', '1')
    os.environ.setdefault('SSD_FUSED_RESPONSE_TOPOLOGY', '0')
    os.environ['SSD_DUET_EXIT_REPLICA'] = '0'
    os.environ['SSD_DUET_EXIT_TOPM_GATHER'] = '0'
    os.environ['SSD_PROXY_STREAM'] = '0'
    os.environ['SSD_BATCH_TREE_PROXY_STREAM'] = '0'
    os.environ['SSD_TREE_VERIFY_BUCKETS'] = ''
    os.environ['SSD_TREE_EXPANSION_POLICY'] = 'q_path'
    os.environ['SSD_TREE_LEGACY_CALIBRATION'] = '0'
    os.environ['NCCL_GRAPH_MIXING_SUPPORT'] = '0'
    # Matched experiments used the same NCCL transport setting on both hosts.
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['PAPER_COLD_PREFIX'] = '1'
    for name in tuple(os.environ):
        if name.startswith('DUET_REVIEW_') or name == 'DUET_EQUAL_ROOTS':
            del os.environ[name]
