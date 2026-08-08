import torch
from torch import nn
import triton
import triton.language as tl

from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from ssd.layers.fi_attn import get_fi_backend, use_flashinfer_attention
from ssd.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    D_POT: tl.constexpr,
    N_SLOTS: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot < 0:
        # -1 = 정상 padding skip. 그 외 음수(page=-1 유래 -bs..-2)도
        # 음수 주소 store 방지 (리뷰: 잠재 메모리 오염 결함)
        return
    if slot >= N_SLOTS:
        # A stale/corrupt page table must not turn into an arbitrary global
        # memory write.  Debug input checks report the bad lane separately.
        return
    offs = tl.arange(0, D_POT)
    mask = offs < D
    key = tl.load(key_ptr + idx * key_stride + offs, mask=mask, other=0.0)
    value = tl.load(value_ptr + idx * value_stride + offs, mask=mask, other=0.0)
    cache_offsets = slot.to(tl.int64) * D + offs
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask)
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    D_POT = triton.next_power_of_2(D)
    n_slots = k_cache.shape[0] * k_cache.shape[1]
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache,
        slot_mapping, D, D_POT, n_slots)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        # Blackwell sm_120 is not supported by the installed sgl attention
        # kernels.  Keep the existing path everywhere else and switch only
        # the dense attention calls; tree-specific FlashInfer wrappers below
        # are unchanged.
        use_fi = use_flashinfer_attention(q.device)
        fi_backend = get_fi_backend(q.device) if use_fi else None

        if context.is_prefill:
            if use_fi:
                if context.block_tables is not None:
                    o = fi_backend.run_paged(q, k_cache, v_cache)
                else:
                    k = k.view(-1, self.num_kv_heads, self.head_dim)
                    v = v.view(-1, self.num_kv_heads, self.head_dim)
                    o = fi_backend.run_ragged(q, k, v)
            else:
                if context.block_tables is not None:
                    k, v = k_cache, v_cache

                k = k.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                )
        else:
            # P2-tree TREE_VERIFY (T3.1b, docs/duet/internal/20): 명시 mode — 현행
            # 암묵 dispatch(cu_seqlens 유무)로는 트리 mask를 표현할 수
            # 없다 (리뷰4). context.tree_verify_wrapper가 설정된 verify
            # 에서만 진입; 미설정이면 기존 경로 완전 불변.
            _tv_wrapper = getattr(context, "tree_verify_wrapper", None)
            if _tv_wrapper is not None:
                if self.draft:
                    raise RuntimeError(
                        "tree_verify_wrapper set on DRAFT attention — "
                        "mode 오배선 (target verify 전용)")
                o = _tv_wrapper.run(q, (self.k_cache, self.v_cache))
                return o.view(-1, self.num_heads * self.head_dim)
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                if use_fi:
                    if fi_backend.cg_cur is not None:
                        o = fi_backend.cg_cur.run(q, k_cache, v_cache)
                    else:
                        o = fi_backend.run_paged(q, k_cache, v_cache)
                else:
                    o = flash_attn_with_kvcache(
                        q, k_cache, v_cache,
                        cache_seqlens=context.context_lens,
                        page_table=context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                        cu_seqlens_q=context.cu_seqlens_q,
                        max_seqlen_q=context.max_seqlen_q,
                    )

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    # Layout-aware: use context.active_mq_len/active_wrappers if set
                    mq_len = context.active_mq_len if context.active_mq_len is not None else self.F * (self.K+1)
                    wrappers = context.active_wrappers if context.active_wrappers is not None else self.prefill_wrappers
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            else: # single query decode
                if use_fi:
                    if fi_backend.cg_cur is not None:
                        o = fi_backend.cg_cur.run(q, k_cache, v_cache)
                    else:
                        o = fi_backend.run_decode(q, k_cache, v_cache)
                else:
                    q = q.unsqueeze(1)
                    o = flash_attn_with_kvcache(
                        q, k_cache, v_cache,
                        cache_seqlens=context.context_lens,
                        page_table=context.block_tables,
                        softmax_scale=self.scale,
                        causal=True,
                    )

        o = o.view(-1, self.num_heads * self.head_dim)
        return o
