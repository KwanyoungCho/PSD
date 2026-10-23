"""Shape-only phase fanout resolution; no tensors, RNG or sampling changes."""


def phase_fanout(config, phase):
    if phase not in ('p1', 'p2'):
        raise ValueError(f'tree phase must be p1|p2; got {phase!r}')
    value = getattr(config, f'duet_{phase}_tree_c_tensor', None)
    return getattr(config, 'duet_tree_c_tensor', 3) if value is None else value


def sibling_capacity(config):
    # Shared verification can receive either phase (or a chain miss).
    return max(phase_fanout(config, 'p1'), phase_fanout(config, 'p2'))


def validate_fanout(config):
    for name in ('duet_tree_c_tensor', 'duet_p1_tree_c_tensor',
                 'duet_p2_tree_c_tensor'):
        value = getattr(config, name, None)
        if value is None and name != 'duet_tree_c_tensor':
            continue
        if type(value) is not int or not 1 <= value <= 8:
            raise ValueError(f'{name} must be an integer in [1,8]; got {value!r}')
