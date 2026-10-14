"""Public, calibration-free expansion choice (no tensor/model imports)."""
import os


def validate_tree_score_mode():
    """Reject stale calibrated launch settings before expensive model loading."""
    mode = os.getenv('SSD_TREE_EXPANSION_POLICY', 'q_path')
    if (mode != 'q_path'
            and os.getenv('SSD_TREE_LEGACY_CALIBRATION', '0') != '1'):
        raise ValueError(
            'Serving uses calibration-free q_path. Unset '
            'SSD_TREE_EXPANSION_POLICY and choose duet_tree_expansion='
            'last|frontier. Historical reproduction only: '
            'SSD_TREE_LEGACY_CALIBRATION=1.')


def resolve_tree_expansion(config, phase):
    """Resolve a phase override, otherwise the shared last/frontier choice."""
    if phase not in ("p1", "p2"):
        raise ValueError(f"tree phase must be p1|p2; got {phase!r}")
    shared = getattr(config, "duet_tree_expansion", "last")
    if shared not in ("last", "frontier"):
        raise ValueError(f"duet_tree_expansion must be last|frontier; got {shared!r}")
    override = getattr(config, f"duet_{phase}_tree_expansion", None)
    if override is not None and override not in ("last", "frontier"):
        raise ValueError(
            f"duet_{phase}_tree_expansion must be last|frontier or None; got {override!r}")
    return shared if override is None else override
