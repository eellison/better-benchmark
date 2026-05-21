"""
Standalone repro captured via capture_hook.
Label: hf_GPTNeoForCausalLM_train_001
Pattern hash: ebebe1ce6cce
Shape hash: 52c397d6
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
_shapes_config = "(T([512, 128, 128], f32), T([32, 16, 128, 128], f32), T([1, 1, 2048, 2048], b8), T([], f32), S([32, 16, 128, 128]), S([512, 128, 128]))"

class Repro(torch.nn.Module):
    def forward(self, bmm_93: "f32[512, 128, 128]", arg225_1: "f32[32, 16, 128, 128]", arg6_1: "b8[1, 1, 2048, 2048]", full_1: "f32[]", _shape_param_0, _shape_param_1):
        # No stacktrace found for following nodes
        view_default: "f32[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_0);  bmm_93 = _shape_param_0 = None
        mul_tensor: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_default, arg225_1);  view_default = None
        sum_dim_int_list: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True)
        neg_default: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg225_1);  arg225_1 = None
        fma_default: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        slice_tensor: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg6_1, 2, 0, 128);  arg6_1 = None
        slice_tensor_1: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128);  slice_tensor = None
        where_self: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_tensor_1, fma_default, full_1);  slice_tensor_1 = fma_default = full_1 = None
        view_default_1: "f32[512, 128, 128]" = torch.ops.aten.view.default(where_self, _shape_param_1);  where_self = _shape_param_1 = None
        return view_default_1



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
