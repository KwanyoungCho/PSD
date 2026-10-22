"""Startup-only target query shapes; never change the generated tree budget.

The wire keeps the phase cap. Extra graph buckets only avoid computing dummy
rows for short trees. Each (width, page count) still owns its FlashInfer plan
storage: sharing that storage across shapes is not safe.
"""


def target_verify_buckets(phase_caps, extra=""):
    caps = {int(cap) for cap in phase_caps}
    if not caps or min(caps) < 1:
        raise ValueError("target verify phase caps must be positive")
    maximum = max(caps)
    if not extra:
        return sorted(caps)
    if extra == "exact":
        return list(range(1, maximum + 1))
    try:
        fields = extra.split(",")
        additional = {int(field) for field in fields}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("SSD_TREE_VERIFY_BUCKETS must be 'exact' or a comma-separated positive node-count list") from exc
    if not additional or min(additional) < 1 or max(additional) > maximum:
        raise ValueError(f"SSD_TREE_VERIFY_BUCKETS entries must be in 1..{maximum}; got {extra!r}")
    return sorted(caps | additional)


def select_target_verify_bucket(buckets, valid):
    if valid < 1:
        raise ValueError("tree verification requires at least one nonroot node")
    for bucket in buckets:
        if bucket >= valid:
            return bucket
    raise ValueError(f"tree valid count {valid} exceeds captured buckets {buckets}")
