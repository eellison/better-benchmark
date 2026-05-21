"""
Standalone repro captured via capture_hook.
Label: torchbench_demucs_train_004
Pattern hash: b691b8dad90a
Shape hash: cbf90c79
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
_shapes_config = "(T([64, 64, 95696], f32), T([64, 64, 95696], f32), T([64, 128, 95696], f32))"

class Repro(torch.nn.Module):
    def forward(self, arg31_1: "f32[64, 64, 95696]", getitem_27: "f32[64, 64, 95696]", arg14_1: "f32[64, 128, 95696]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[64, 64, 95696]" = torch.ops.aten.add.Tensor(arg31_1, getitem_27);  arg31_1 = getitem_27 = None
        slice_tensor: "f32[64, 64, 95696]" = torch.ops.aten.slice.Tensor(arg14_1, 1, 0, 64)
        slice_tensor_1: "f32[64, 64, 95696]" = torch.ops.aten.slice.Tensor(arg14_1, 1, 64, 128);  arg14_1 = None
        sigmoid_default: "f32[64, 64, 95696]" = torch.ops.aten.sigmoid.default(slice_tensor_1);  slice_tensor_1 = None
        sub_tensor: "f32[64, 64, 95696]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_default)
        mul_tensor: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(sub_tensor, sigmoid_default);  sub_tensor = None
        mul_tensor_1: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(mul_tensor, slice_tensor);  mul_tensor = slice_tensor = None
        mul_tensor_2: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor);  mul_tensor_1 = None
        mul_tensor_3: "f32[64, 64, 95696]" = torch.ops.aten.mul.Tensor(sigmoid_default, add_tensor);  sigmoid_default = add_tensor = None
        cat_default: "f32[64, 128, 95696]" = torch.ops.aten.cat.default([mul_tensor_3, mul_tensor_2], 1);  mul_tensor_3 = mul_tensor_2 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(cat_default, [0, 2]);  cat_default = None
        return sum_dim_int_list



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
