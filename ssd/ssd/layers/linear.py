import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# AWQ/quant integration: when `ssd.quant.init_context.quant_init_context()`
# is active, TP linear modules allocate `weight` on the meta device so no
# GPU memory is spent on dense weights. The quant loader later attaches an
# `AwqQuantState` via `attach_quant_state()`. Forward dispatches on
# `self.quant_state` being non-None — see plan §6.3.1 option (2).
def _new_weight(shape, *, quant_mode: bool) -> nn.Parameter:
    if quant_mode:
        return nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
    return nn.Parameter(torch.empty(shape))


def _dense_forward(weight: torch.Tensor, x: torch.Tensor, bias):
    return F.linear(x, weight, bias)


def _quant_forward(state, x: torch.Tensor, bias):
    # Import locally to avoid a circular import between layers.linear and
    # quant.marlin (which also imports torch bits).
    from ssd.quant.marlin import awq_matmul
    return awq_matmul(x, state, bias)


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_group = tp_group
        self.tp_size = tp_size

        assert not (tp_group is None and self.tp_size > 1), "ERROR in LinearBase: tp_group is None and tp_size > 1"

        if self.tp_size > 1:
            # target shards [0, N-2] during draft_async get tp_group, self.tp_rank=N-1 then
            self.tp_rank = dist.get_rank(group=self.tp_group)
        else:
            # normal decoding or we are draft
            self.tp_rank = 0

        # AWQ quant state — None in dense mode. Attached post-construction by
        # the quant loader. When non-None, forward uses Marlin W4A16.
        self.quant_state = None

    def attach_quant_state(self, state) -> None:
        """Attach an AwqQuantState. Clears any meta/dense `weight` buffer.

        Shape validation: the state must match the per-partition shape that
        the module was constructed with.
        """
        from ssd.quant.state import AwqQuantState
        assert isinstance(state, AwqQuantState), \
            f"attach_quant_state expects AwqQuantState, got {type(state)}"
        # Sanity: validate shapes against module's per-partition expectations.
        expected_in = getattr(self, "input_size_per_partition", self.input_size)
        expected_out = getattr(self, "output_size_per_partition", self.output_size)
        assert state.in_features == expected_in, \
            f"quant_state.in_features={state.in_features} != module.in={expected_in}"
        assert state.out_features == expected_out, \
            f"quant_state.out_features={state.out_features} != module.out={expected_out}"
        self.quant_state = state
        # Drop the placeholder weight. We keep the attribute as None so that
        # `hasattr(mod, 'weight')` still works but any code that reaches for it
        # will hit a clean AttributeError instead of firing against meta.
        if hasattr(self, "weight") and self.weight is not None:
            # Use nn.Module.__setattr__ machinery to unregister the Parameter.
            del self._parameters["weight"]
            self.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__(input_size, output_size, tp_group=tp_group, tp_size=tp_size)
        from ssd.quant.init_context import is_quant_init_active
        _q = is_quant_init_active()
        self.weight = _new_weight((self.output_size, self.input_size), quant_mode=_q)
        if self.weight is not None:
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_state is not None:
            return _quant_forward(self.quant_state, x, self.bias)
        return _dense_forward(self.weight, x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__(input_size, output_size, 0, tp_group=tp_group, tp_size=tp_size)
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        from ssd.quant.init_context import is_quant_init_active
        _q = is_quant_init_active()
        self.weight = _new_weight(
            (self.output_size_per_partition, self.input_size), quant_mode=_q,
        )
        if self.weight is not None:
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_state is not None:
            return _quant_forward(self.quant_state, x, self.bias)
        return _dense_forward(self.weight, x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        self.output_sizes = output_sizes
        self.tp_group = tp_group
        self.tp_size = tp_size
        super().__init__(input_size, sum(output_sizes), bias=bias, tp_group=tp_group, tp_size=tp_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias, tp_group, tp_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_group: dist.ProcessGroup | None = None,
        tp_size: int = 1,
    ):
        super().__init__(input_size, output_size, 1, tp_group=tp_group, tp_size=tp_size)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size
        self.tp_group = tp_group

        from ssd.quant.init_context import is_quant_init_active
        _q = is_quant_init_active()
        self.weight = _new_weight(
            (self.output_size, self.input_size_per_partition), quant_mode=_q,
        )
        if self.weight is not None:
            self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_state is not None:
            # RowParallel bias is only applied on rank 0 in the dense path;
            # match that convention exactly.
            bias = self.bias if self.tp_rank == 0 else None
            y = _quant_forward(self.quant_state, x, bias)
        else:
            y = _dense_forward(self.weight, x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y, group=self.tp_group)
        return y
