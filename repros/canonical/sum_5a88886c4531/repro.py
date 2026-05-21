"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: 5a88886c4531
Shape hash: 8233313b
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
_shapes_config = "(T([256, 128, 512], f32), T([1, 512], i64, gen=Index(512)), T([], f32))"

class Repro(torch.nn.Module):
    def forward(self, mul_538: "f32[256, 128, 512]", arg1_1: "i64[1, 512]", full_1: "f32[]"):
        # No stacktrace found for following nodes
        sum_dim_int_list: "f32[1, 128, 512]" = torch.ops.aten.sum.dim_IntList(mul_538, [0], True);  mul_538 = None
        slice_tensor: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 1, 0, 128);  arg1_1 = None
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(slice_tensor, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 512]" = torch.ops.aten.where.self(unsqueeze_default, full_1, sum_dim_int_list);  unsqueeze_default = full_1 = sum_dim_int_list = None
        full_default: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 512]" = torch.ops.aten.index_put.default(full_default, [slice_tensor], where_self, True);  full_default = slice_tensor = where_self = None
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
