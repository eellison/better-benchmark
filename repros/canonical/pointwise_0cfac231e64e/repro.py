"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Reformer_train_014
Pattern hash: 0cfac231e64e
Shape hash: bc1e210c
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
_shapes_config = "(T([8, 4096, 256], b8), T([8, 4096, 256], f32), T([8, 4096], i64, gen=Index(320)))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "b8[8, 4096, 256]", arg3_1: "f32[8, 4096, 256]", arg0_1: "i64[8, 4096]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[8, 4096, 256]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0526315789473684);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(arg3_1, mul_tensor);  arg3_1 = mul_tensor = None
        eq_scalar: "b8[8, 4096]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[8, 4096, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 4096, 256]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_1);  unsqueeze_default = full_default = mul_tensor_1 = None
        full_default_1: "f32[320, 256]" = torch.ops.aten.full.default([320, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[320, 256]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self, True);  full_default_1 = arg0_1 = where_self = None
        return index_put_default



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
