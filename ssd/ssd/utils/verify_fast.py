"""All-stochastic JIT verification without data-dependent CPU branching.

Caller must know from CPU request metadata that both temperatures are positive
and every proposal comes from its accompanying q (JIT on cache misses). Mixed
greedy/stochastic and non-JIT requests retain the general verifier.
"""
import torch


def verify_stochastic(logits_p, logits_q, speculations, temperatures_target,
                      temperatures_draft, valid_k=None, sampler_x=None,
                      async_fan_out=None):
    from ssd.utils.verify import _materialize_verified
    from ssd.utils.async_helpers.async_spec_helpers import apply_sampler_x_rescaling
    b, kp1, _ = logits_p.shape
    k = kp1 - 1
    # Match the reference's division/cast order, random draw shapes and order.
    p = torch.softmax((logits_p / temperatures_target[:, None, None].clamp(min=1e-8)).float(), -1)
    q = torch.softmax((logits_q / temperatures_draft[:, None, None].clamp(min=1e-8)).float(), -1)
    if sampler_x is not None:
        if async_fan_out is None:
            raise ValueError("sampler_x requires async_fan_out")
        q = apply_sampler_x_rescaling(q, sampler_x, async_fan_out)
    tokens = speculations[:, 1:, None]
    ratio = (p[:, :k].gather(2, tokens).squeeze(2) /
             (q.gather(2, tokens).squeeze(2) + 1e-10)).clamp(max=1.)
    rejected = torch.rand_like(ratio) > ratio
    counts = torch.where(rejected.any(1), rejected.int().argmax(1), k)
    if valid_k is not None:
        counts = torch.minimum(counts, valid_k)
    rows = torch.arange(b, device=logits_p.device)
    fallback = p[rows, counts]
    fallback = fallback / fallback.sum(1, keepdim=True)
    # Reference subtracts from unnormalized p_fallback (softmax roundoff).
    residual = (p[rows, counts] - q[rows, counts.clamp(max=k-1)]).clamp(min=0.)
    mass = residual.sum(1, keepdim=True)
    adjusted = torch.where(mass > 0, residual / mass, fallback)
    from_residual = torch.multinomial(adjusted, 1).squeeze(1)
    from_target = torch.multinomial(fallback, 1).squeeze(1)
    real_k = k if valid_k is None else valid_k.clamp(max=k)
    recovery = torch.where(counts < real_k, from_residual, from_target)
    return _materialize_verified(speculations, counts, recovery)
