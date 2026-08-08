"""FlashInfer fallback for GPUs unsupported by ``sgl-kernel`` attention.

The normal SSD path remains ``sgl-kernel``.  ``auto`` selects this backend on
SM120+ (currently RTX PRO 6000 Blackwell), where the installed sgl attention
kernels cannot execute.  Set ``SSD_ATTN_BACKEND=sgl`` or ``flashinfer`` to
override the automatic choice.

FlashInfer planning is deliberately outside model layers: eager execution
plans once per model call and CUDA graphs plan once immediately before replay.
All layers then reuse the same planned wrapper.
"""

from __future__ import annotations

import os

import flashinfer
import torch


_WORKSPACE_BYTES = 320 * 1024 * 1024
_VALID_BACKENDS = {"auto", "sgl", "flashinfer"}
_AUTO_SELECTION: dict[int, str] = {}


def attention_backend(device: torch.device | int | None = None) -> str:
    """Resolve the dense-attention backend without allocating GPU memory."""
    requested = os.environ.get("SSD_ATTN_BACKEND", "auto").strip().lower()
    if requested not in _VALID_BACKENDS:
        raise ValueError(
            "SSD_ATTN_BACKEND must be auto|sgl|flashinfer; "
            f"got {requested!r}"
        )
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        try:
            if device is None:
                index = torch.cuda.current_device()
            elif isinstance(device, torch.device):
                index = device.index
                if index is None:
                    index = torch.cuda.current_device()
            else:
                index = int(device)
            cached = _AUTO_SELECTION.get(index)
            if cached is not None:
                return cached
            major, _minor = torch.cuda.get_device_capability(index)
            selected = "flashinfer" if major >= 12 else "sgl"
            _AUTO_SELECTION[index] = selected
            return selected
        except (RuntimeError, AssertionError, ValueError):
            # Import-time/unit-test fallback below.
            pass

    arch = os.environ.get("SSD_CUDA_ARCH", "")
    try:
        major = int(arch.split(".", 1)[0])
    except (TypeError, ValueError):
        major = 0
    return "flashinfer" if major >= 12 else "sgl"


def use_flashinfer_attention(device: torch.device | int | None = None) -> bool:
    return attention_backend(device) == "flashinfer"


def graph_batch_sizes(max_bs: int) -> list[int]:
    """Return graph buckets that never exceed their backing buffer rows."""
    if max_bs <= 0:
        raise ValueError(f"max_bs must be positive; got {max_bs}")
    candidates = [1, 2, 4, 8]
    candidates.extend(range(16, max_bs + 1, 16))
    candidates.append(max_bs)
    return sorted({bs for bs in candidates if bs <= max_bs})


def duet_graph_name(k_plus_1: int) -> str:
    """Stable wrapper key shared by DUET graph capture and replay."""
    if int(k_plus_1) <= 0:
        raise ValueError("DUET FlashInfer query width must be positive")
    return f"duet_verify_kp{int(k_plus_1)}"


class FIAttnBackend:
    """Eager wrappers and named CUDA-graph wrapper families for one GPU."""

    def __init__(self, device: torch.device):
        self.device = device
        # Allocate eager workspaces only if that mode is actually reached.
        # Graph-mode decode/verify does not need the three eager workspaces.
        self._decode = None
        self._paged = None
        self._ragged = None
        self.cg: dict[str, FIGraphWrappers] = {}
        self.cg_cur: FIGraphWrappers | None = None

    def _ensure_decode(self):
        if self._decode is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
            )
            self._decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD"
            )

    def _ensure_paged(self):
        if self._paged is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
            )
            self._paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace, "NHD"
            )

    def _ensure_ragged(self):
        if self._ragged is None:
            workspace = torch.empty(
                _WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
            )
            self._ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace, "NHD"
            )

    def cg_get(
        self,
        name: str,
        bs_list: list[int],
        max_num_blocks: int,
        qlen: int,
    ) -> "FIGraphWrappers":
        wrapper = self.cg.get(name)
        if wrapper is None:
            wrapper = FIGraphWrappers(
                self.device,
                bs_list,
                max_num_blocks,
                qlen,
                backend=self,
            )
            self.cg[name] = wrapper
        elif wrapper.qlen != qlen or sorted(wrapper.wrappers) != sorted(bs_list):
            raise RuntimeError(
                f"FlashInfer graph family {name!r} reused with a different "
                "query width or batch buckets"
            )
        return wrapper

    @staticmethod
    def _paged_index(
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        counts = (context_lens + page_size - 1) // page_size
        zero = torch.zeros(1, dtype=torch.int32, device=counts.device)
        kv_indptr = torch.cat([zero, counts.cumsum(0).to(torch.int32)])
        columns = torch.arange(
            block_tables.size(1), device=block_tables.device
        )[None, :]
        kv_indices = block_tables[columns < counts[:, None]].to(torch.int32)
        last = context_lens % page_size
        last = torch.where(
            last == 0, torch.full_like(last, page_size), last
        ).to(torch.int32)
        return kv_indptr, kv_indices, last

    def plan_prefill_ragged(
        self, cu_q, cu_k, nq, nkv, head_dim, scale, dtype
    ) -> None:
        self.cg_cur = None
        self._ensure_ragged()
        self._ragged.plan(
            cu_q.to(torch.int32),
            cu_k.to(torch.int32),
            nq,
            nkv,
            head_dim,
            causal=True,
            sm_scale=scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )

    def plan_paged(
        self, cu_q, block_tables, context_lens, page_size,
        nq, nkv, head_dim, scale, dtype
    ) -> None:
        self.cg_cur = None
        self._ensure_paged()
        kvp, kvi, last = self._paged_index(
            block_tables, context_lens, page_size
        )
        self._paged.plan(
            cu_q.to(torch.int32), kvp, kvi, last,
            nq, nkv, head_dim, page_size,
            causal=True, sm_scale=scale,
            q_data_type=dtype, kv_data_type=dtype,
        )

    def plan_decode(
        self, block_tables, context_lens, page_size,
        nq, nkv, head_dim, scale, dtype
    ) -> None:
        self.cg_cur = None
        self._ensure_decode()
        kvp, kvi, last = self._paged_index(
            block_tables, context_lens, page_size
        )
        self._decode.plan(
            kvp, kvi, last, nq, nkv, head_dim, page_size,
            sm_scale=scale, q_data_type=dtype, kv_data_type=dtype,
        )

    def run_ragged(self, q, k, v):
        return self._ragged.run(q, k, v)

    def run_paged(self, q, k_cache, v_cache):
        return self._paged.run(q, (k_cache, v_cache))

    def run_decode(self, q, k_cache, v_cache):
        return self._decode.run(q, (k_cache, v_cache))


class FIGraphWrappers:
    """Graph-safe paged-prefill wrappers, one per batch bucket."""

    def __init__(
        self,
        device: torch.device,
        bs_list: list[int],
        max_num_blocks: int,
        qlen: int,
        ws_bytes: int = _WORKSPACE_BYTES,
        backend: FIAttnBackend | None = None,
    ):
        if not bs_list or min(bs_list) <= 0:
            raise ValueError(f"invalid FlashInfer graph buckets: {bs_list}")
        self.device = device
        self.backend = backend
        self.qlen = qlen
        self.max_bs = max(bs_list)
        self.wrappers = {}
        self.buffers = {}
        workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=device)
        for bs in bs_list:
            qo = torch.arange(bs + 1, dtype=torch.int32, device=device) * qlen
            kvp = torch.zeros(bs + 1, dtype=torch.int32, device=device)
            kvi = torch.zeros(
                max(bs * max_num_blocks, 1), dtype=torch.int32, device=device
            )
            kvl = torch.zeros(bs, dtype=torch.int32, device=device)
            self.wrappers[bs] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=qo,
                paged_kv_indptr_buf=kvp,
                paged_kv_indices_buf=kvi,
                paged_kv_last_page_len_buf=kvl,
            )
            self.buffers[bs] = (qo, kvp, kvi, kvl)
        self.active = None

    def bucket(self, bs: int) -> int:
        for candidate in sorted(self.wrappers):
            if candidate >= bs:
                return candidate
        raise ValueError(
            f"batch {bs} exceeds largest FlashInfer graph bucket {self.max_bs}"
        )

    def plan(
        self,
        bs,
        block_tables,
        context_lens,
        page_size,
        nq,
        nkv,
        head_dim,
        scale,
        dtype,
        causal=True,
        qo_src=None,
    ):
        bucket = self.bucket(bs)
        qo, kvp, kvi, kvl = self.buffers[bucket]
        if bs > context_lens.numel() or bs > block_tables.size(0):
            raise RuntimeError(
                "FlashInfer graph plan batch exceeds supplied rows: "
                f"bs={bs}, context_rows={context_lens.numel()}, "
                f"block_rows={block_tables.size(0)}"
            )
        if qo_src is not None:
            copied = min(qo.numel(), qo_src.numel())
            qo[:copied] = qo_src[:copied].to(torch.int32)
            if copied < qo.numel():
                qo[copied:] = qo[copied - 1]

        n = min(bs, context_lens.numel(), block_tables.size(0))
        if n == 1 and bucket == 1:
            cl0 = context_lens[0]
            count = (cl0 + page_size - 1) // page_size
            kvp[1] = count
            copied = min(kvi.numel(), block_tables.size(1))
            kvi[:copied] = block_tables[0, :copied]
            kvl[0] = cl0 - (count - 1) * page_size
        else:
            counts = (context_lens[:n] + page_size - 1) // page_size
            kvp[1:n + 1] = torch.cumsum(counts, 0, dtype=torch.int32)
            if n < bucket:
                kvp[n + 1:] = kvp[n]
            if n == 1:
                copied = min(kvi.numel(), block_tables.size(1))
                kvi[:copied] = block_tables[0, :copied]
            else:
                columns = torch.arange(
                    block_tables.size(1), device=block_tables.device
                )[None, :]
                indices = block_tables[:n][columns < counts[:, None]]
                kvi[:indices.numel()] = indices
            last = context_lens[:n] % page_size
            kvl[:n] = torch.where(
                last == 0, torch.full_like(last, page_size), last
            )
            if n < bucket:
                kvl[n:bucket] = 1

        wrapper = self.wrappers[bucket]
        wrapper.plan(
            qo, kvp, kvi, kvl,
            nq, nkv, head_dim, page_size,
            causal=causal, sm_scale=scale,
            q_data_type=dtype, kv_data_type=dtype,
        )
        self.active = wrapper
        if self.backend is not None:
            self.backend.cg_cur = self
        return wrapper

    def run(self, q, k_cache, v_cache):
        if self.active is None:
            raise RuntimeError("FlashInfer graph wrapper used before plan()")
        return self.active.run(q, (k_cache, v_cache))


_BACKENDS: dict[int, FIAttnBackend] = {}


def get_fi_backend(device: torch.device | int) -> FIAttnBackend:
    if isinstance(device, torch.device):
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
    else:
        index = int(device)
    backend = _BACKENDS.get(index)
    if backend is None:
        backend = FIAttnBackend(torch.device(f"cuda:{index}"))
        _BACKENDS[index] = backend
    return backend
