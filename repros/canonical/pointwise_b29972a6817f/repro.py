"""
Standalone repro captured via capture_hook.
Label: torchbench_modded_nanogpt_train
Pattern hash: b29972a6817f
Shape hash: 2f69670d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([6144, 3072], bf16), T([6144, 3072], bf16), S([1, 6144, 3072]), S([1, 6144, 3072]), S([6144, 3072]))"

class Repro(torch.nn.Module):
    def forward(self, mm_59: "bf16[6144, 3072]", mm_55: "bf16[6144, 3072]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:727 in forward, code: x = F.linear(x, self.c_proj.type_as(x))
        reshape_default: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_59, _shape_param_0);  mm_59 = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        reshape_default_1: "bf16[1, 6144, 3072]" = torch.ops.aten.reshape.default(mm_55, _shape_param_1);  mm_55 = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        relu_default: "bf16[1, 6144, 3072]" = torch.ops.aten.relu.default(reshape_default_1);  reshape_default_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:726 in forward, code: ).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        pow_tensor_scalar: "bf16[1, 6144, 3072]" = torch.ops.aten.pow.Tensor_Scalar(relu_default, 1.0)
        mul_scalar: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 2.0);  pow_tensor_scalar = None
        mul_tensor: "bf16[1, 6144, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, mul_scalar);  reshape_default = mul_scalar = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:724 in forward, code: x = F.relu(
        le_scalar: "b8[1, 6144, 3072]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "bf16[1, 6144, 3072]" = torch.ops.aten.where.self(le_scalar, full_default, mul_tensor);  le_scalar = full_default = mul_tensor = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/modded_nanogpt/model.py:723 in forward, code: x = F.linear(x, self.c_fc.T.type_as(x))
        reshape_default_2: "bf16[6144, 3072]" = torch.ops.aten.reshape.default(where_self, _shape_param_2);  where_self = _shape_param_2 = None
        return reshape_default_2



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
