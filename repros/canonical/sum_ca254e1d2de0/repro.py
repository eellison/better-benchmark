"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g53
Pattern hash: ca254e1d2de0
Shape hash: 8afdb540
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg358_1: "f32[32, 1]", arg217_1: "f16[32, 512]", arg531_1: "f32[32, 512]"):
        # No stacktrace found for following nodes
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        clamp_min_default: "f32[32, 1]" = torch.ops.aten.clamp_min.default(arg358_1, 1e-12)
        expand_default: "f32[32, 512]" = torch.ops.aten.expand.default(clamp_min_default, [32, 512]);  clamp_min_default = None
        div_tensor: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg217_1, expand_default)
        div_tensor_1: "f32[32, 512]" = torch.ops.aten.div.Tensor(div_tensor, expand_default);  div_tensor = None
        neg_default: "f32[32, 512]" = torch.ops.aten.neg.default(arg531_1)
        mul_tensor: "f32[32, 512]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg531_1, expand_default);  arg531_1 = expand_default = None
        convert_element_type_default: "f16[32, 512]" = torch.ops.prims.convert_element_type.default(div_tensor_2, torch.float16);  div_tensor_2 = None
        sum_dim_int_list: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True);  mul_tensor = None
        ge_scalar: "b8[32, 1]" = torch.ops.aten.ge.Scalar(arg358_1, 1e-12)
        where_self: "f32[32, 1]" = torch.ops.aten.where.self(ge_scalar, sum_dim_int_list, full_default);  ge_scalar = sum_dim_int_list = None
        div_tensor_3: "f32[32, 512]" = torch.ops.aten.div.Tensor(arg217_1, arg358_1);  arg217_1 = None
        eq_scalar: "b8[32, 1]" = torch.ops.aten.eq.Scalar(arg358_1, 0);  arg358_1 = None
        where_self_1: "f32[32, 512]" = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor_3);  eq_scalar = full_default = div_tensor_3 = None
        mul_tensor_1: "f32[32, 512]" = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        convert_element_type_default_1: "f16[32, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float16);  mul_tensor_1 = None
        add_tensor: "f16[32, 512]" = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        return add_tensor


def _default_make_inputs():
    return [
    torch.randn([32, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512], dtype=torch.float16, device='cuda'),
    torch.randn([32, 512], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
