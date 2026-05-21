"""
Standalone repro captured via capture_hook.
Label: torchbench_llama_infer
Pattern hash: b282e7417b3d
Shape hash: 06d2847f
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
_shapes_config = "(T([1024, 512], f32), T([32, 32, 512], f32), T([512], f32), T([32000, 512], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), T([64, 1024, 8, 64], f32), S([32, 32, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_55: "f32[1024, 512]", add_36: "f32[32, 32, 512]", arg91_1: "f32[512]", arg92_1: "f32[32000, 512]", arg7_1: "f32[64, 1024, 8, 64]", slice_scatter_1: "f32[64, 1024, 8, 64]", arg8_1: "f32[64, 1024, 8, 64]", slice_scatter_3: "f32[64, 1024, 8, 64]", arg18_1: "f32[64, 1024, 8, 64]", slice_scatter_5: "f32[64, 1024, 8, 64]", arg19_1: "f32[64, 1024, 8, 64]", slice_scatter_7: "f32[64, 1024, 8, 64]", arg29_1: "f32[64, 1024, 8, 64]", slice_scatter_9: "f32[64, 1024, 8, 64]", arg30_1: "f32[64, 1024, 8, 64]", slice_scatter_11: "f32[64, 1024, 8, 64]", arg40_1: "f32[64, 1024, 8, 64]", slice_scatter_13: "f32[64, 1024, 8, 64]", arg41_1: "f32[64, 1024, 8, 64]", slice_scatter_15: "f32[64, 1024, 8, 64]", arg51_1: "f32[64, 1024, 8, 64]", slice_scatter_17: "f32[64, 1024, 8, 64]", arg52_1: "f32[64, 1024, 8, 64]", slice_scatter_19: "f32[64, 1024, 8, 64]", arg62_1: "f32[64, 1024, 8, 64]", slice_scatter_21: "f32[64, 1024, 8, 64]", arg63_1: "f32[64, 1024, 8, 64]", slice_scatter_23: "f32[64, 1024, 8, 64]", arg73_1: "f32[64, 1024, 8, 64]", slice_scatter_25: "f32[64, 1024, 8, 64]", arg74_1: "f32[64, 1024, 8, 64]", slice_scatter_27: "f32[64, 1024, 8, 64]", arg84_1: "f32[64, 1024, 8, 64]", slice_scatter_29: "f32[64, 1024, 8, 64]", arg85_1: "f32[64, 1024, 8, 64]", slice_scatter_31: "f32[64, 1024, 8, 64]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/utils/_device.py:122 in __torch_function__, code: return func(*args, **kwargs)
        reshape_default: "f32[32, 32, 512]" = torch.ops.aten.reshape.default(mm_55, _shape_param_0);  mm_55 = _shape_param_0 = None
        add_tensor: "f32[32, 32, 512]" = torch.ops.aten.add.Tensor(add_36, reshape_default);  add_36 = reshape_default = None
        pow_tensor_scalar: "f32[32, 32, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_tensor, 2)
        mean_dim: "f32[32, 32, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None
        add_tensor_1: "f32[32, 32, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-05);  mean_dim = None
        rsqrt_default: "f32[32, 32, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  add_tensor = rsqrt_default = None
        mul_tensor_1: "f32[32, 32, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, arg91_1);  mul_tensor = arg91_1 = None
        select_int: "f32[32, 512]" = torch.ops.aten.select.int(mul_tensor_1, 1, -1);  mul_tensor_1 = None
        permute_default: "f32[512, 32000]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
        copy__default: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg7_1, slice_scatter_1);  arg7_1 = slice_scatter_1 = None
        copy__default_1: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg8_1, slice_scatter_3);  arg8_1 = slice_scatter_3 = None
        copy__default_2: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg18_1, slice_scatter_5);  arg18_1 = slice_scatter_5 = None
        copy__default_3: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg19_1, slice_scatter_7);  arg19_1 = slice_scatter_7 = None
        copy__default_4: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg29_1, slice_scatter_9);  arg29_1 = slice_scatter_9 = None
        copy__default_5: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg30_1, slice_scatter_11);  arg30_1 = slice_scatter_11 = None
        copy__default_6: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg40_1, slice_scatter_13);  arg40_1 = slice_scatter_13 = None
        copy__default_7: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg41_1, slice_scatter_15);  arg41_1 = slice_scatter_15 = None
        copy__default_8: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg51_1, slice_scatter_17);  arg51_1 = slice_scatter_17 = None
        copy__default_9: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg52_1, slice_scatter_19);  arg52_1 = slice_scatter_19 = None
        copy__default_10: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg62_1, slice_scatter_21);  arg62_1 = slice_scatter_21 = None
        copy__default_11: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg63_1, slice_scatter_23);  arg63_1 = slice_scatter_23 = None
        copy__default_12: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg73_1, slice_scatter_25);  arg73_1 = slice_scatter_25 = None
        copy__default_13: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg74_1, slice_scatter_27);  arg74_1 = slice_scatter_27 = None
        copy__default_14: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg84_1, slice_scatter_29);  arg84_1 = slice_scatter_29 = None
        copy__default_15: "f32[64, 1024, 8, 64]" = torch.ops.aten.copy_.default(arg85_1, slice_scatter_31);  arg85_1 = slice_scatter_31 = None
        return (select_int, permute_default, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15)



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
