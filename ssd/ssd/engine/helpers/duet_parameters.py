"""MLSys v1.1 public parameter contract and shape derivation.

Only canonical fields are serialized. Legacy spelling migration is performed
before Config construction; it cannot enable G>M, thresholds, or manual top-L.
"""
from functools import wraps
from numbers import Integral
import warnings


ALIASES = {
    'duet_p2_budget': 'duet_p2_forward_width',
    'duet_tree_root_count': 'duet_p2_root_count',
    'duet_p1_tree_max_nodes': 'duet_p1_tree_verify_nodes',
    'duet_p2_tree_max_nodes': 'duet_p2_tree_verify_nodes',
    'duet_tree_nv': 'duet_p2_tree_verify_nodes',
}
THRESHOLDS = (
    'duet_tree_proxy_threshold', 'duet_tree_conf_threshold',
    'duet_p1_tree_start_threshold', 'duet_p1_tree_conf_threshold',
)


def normalize_duet_kwargs(kwargs):
    """Accept unambiguous old spellings, reject changes that cannot be mapped."""
    out = dict(kwargs)
    migrated = []
    for old, new in ALIASES.items():
        if old not in out:
            continue
        value = out.pop(old)
        if value is None:
            if old in ('duet_p2_budget', 'duet_tree_root_count'):
                out.setdefault(new, None)
            continue
        if out.get(new) is not None and out[new] != value:
            hint = ('MLSys v1.1 requires G=M. Use MLSys-ver1.0 for G>M reproduction.'
                    if 'nodes' in new else f'Use only {new}.')
            raise ValueError(
                f'{old}={value} conflicts with {new}={out[new]}; '
                + hint)
        out[new] = value
        migrated.append(f'{old}->{new}')
    for old in THRESHOLDS:
        if old in out:
            value = out.pop(old)
            if value != 0:
                raise ValueError(f'{old} is removed in v1.1; expansion thresholds are fixed at zero')
            migrated.append(f'{old}->fixed zero')
    for old, message in (
        ('duet_p1_tree_forward_scale', 'set duet_p1_forward_width to an explicit input count'),
        ('duet_proxy_top_k', 'top-L is derived from the transmitted candidate count'),
    ):
        if old in out:
            value = out.pop(old)
            if value is not None:
                raise ValueError(f'{old} is removed in v1.1; {message}. Use MLSys-ver1.0 for old experiments.')
    if migrated:
        warnings.warn('Deprecated DUET parameters: ' + ', '.join(migrated),
                      FutureWarning, stacklevel=3)
    return out


def canonical_config(cls):
    init = cls.__init__
    @wraps(init)
    def construct(self, *args, **kwargs):
        init(self, *args, **normalize_duet_kwargs(kwargs))
    cls.__init__ = construct
    return cls


def positive_int(name, value, *, optional=False):
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer; got {value!r}')


def p1_continuation_width(config, roots):
    """Root round evaluates all roots; continuation input cap is explicit.

    Smaller context buckets use fewer physical rows, as in v1.0. This does
    not discard cache roots or pad short buckets up to the largest bucket.
    """
    positive_int('DS root count', roots)
    width = getattr(config, 'duet_p1_forward_width', None)
    positive_int('duet_p1_forward_width', width, optional=True)
    if getattr(config, 'duet_p1_tree_allocation_policy', 'dynamic') == 'backbone':
        if width is not None and width < roots:
            raise ValueError('DS backbone requires forward width >= root count')
        return int(roots)
    return int(roots if width is None else min(roots, width))
