"""
Standalone repro captured via capture_hook.
Label: timm_dm_nfnet_f0_train_001
Pattern hash: 6f056653b0d6
Shape hash: 4d544b47
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
_shapes_config = "(T([128, 128, 48, 48], f32), T([128, 128, 48, 48], f32), T([128, 128, 48, 48], f32))"

class Repro(torch.nn.Module):
    def forward(self, getitem_225: "f32[128, 128, 48, 48]", getitem_228: "f32[128, 128, 48, 48]", arg166_1: "f32[128, 128, 48, 48]"):
        # No stacktrace found for following nodes
        add_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(getitem_225, getitem_228);  getitem_225 = getitem_228 = None
        mul_tensor: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_tensor, 1.0);  add_tensor = None
        mul_tensor_1: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor, 1.7015043497085571);  mul_tensor = None
        mul_tensor_2: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(arg166_1, 0.7071067811865476)
        erf_default: "f32[128, 128, 48, 48]" = torch.ops.aten.erf.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(erf_default, 1);  erf_default = None
        mul_tensor_3: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5);  add_tensor_1 = None
        mul_tensor_4: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(arg166_1, arg166_1)
        mul_tensor_5: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_4, -0.5);  mul_tensor_4 = None
        exp_default: "f32[128, 128, 48, 48]" = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        mul_tensor_6: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_7: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(arg166_1, mul_tensor_6);  arg166_1 = mul_tensor_6 = None
        add_tensor_2: "f32[128, 128, 48, 48]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_7);  mul_tensor_3 = mul_tensor_7 = None
        mul_tensor_8: "f32[128, 128, 48, 48]" = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_2);  mul_tensor_1 = add_tensor_2 = None
        sum_dim_int_list: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 2, 3]);  mul_tensor_8 = None
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
