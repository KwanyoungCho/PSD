"""Validated startup settings for cache-root ranking, not target sampling."""
import math


def options(config):
    return dict(source=getattr(config, 'duet_root_source', 'complement'),
                normalization=getattr(config, 'duet_root_normalization', 'topm'),
                overlap_mix=getattr(config, 'duet_root_overlap_mix', .25),
                sampler_x=getattr(config, 'sampler_x', None),
                fan_out=getattr(config, 'async_fan_out', 1))


def validate_root_policy(config):
    policy = options(config)
    if policy['source'] not in ('complement', 'residual', 'proxy'):
        raise ValueError('duet_root_source must be complement|residual|proxy')
    if policy['normalization'] not in ('full', 'topm'):
        raise ValueError('duet_root_normalization must be full|topm')
    mix = policy['overlap_mix']
    if (not isinstance(mix, (float, int)) or isinstance(mix, bool)
            or not math.isfinite(mix) or not 0 <= mix <= 1):
        raise ValueError('duet_root_overlap_mix must be finite and in [0, 1]')
