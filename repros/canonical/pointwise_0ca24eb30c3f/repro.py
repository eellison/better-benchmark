"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train_000
Pattern hash: 0ca24eb30c3f
Shape hash: fc72ed2f
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([4, 512], i64))"

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 512]"):
        # No stacktrace found for following nodes
        full_default: "i64[4, 512]" = torch.ops.aten.full.default([4, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_tensor: "i64[4, 511]" = torch.ops.aten.slice.Tensor(full_default, 1, 1, 9223372036854775807)
        slice_tensor_1: "i64[4, 511]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, -1);  arg0_1 = None
        clone_default: "i64[4, 511]" = torch.ops.aten.clone.default(slice_tensor_1);  slice_tensor_1 = None
        copy_default: "i64[4, 511]" = torch.ops.aten.copy.default(slice_tensor, clone_default);  slice_tensor = clone_default = None
        slice_scatter_default: "i64[4, 512]" = torch.ops.aten.slice_scatter.default(full_default, copy_default, 1, 1, 9223372036854775807);  full_default = copy_default = None
        select_int: "i64[4]" = torch.ops.aten.select.int(slice_scatter_default, 1, 0)
        full_default_1: "i64[]" = torch.ops.aten.full.default([], 2, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_1: "i64[4]" = torch.ops.aten.copy.default(select_int, full_default_1);  select_int = full_default_1 = None
        select_scatter_default: "i64[4, 512]" = torch.ops.aten.select_scatter.default(slice_scatter_default, copy_default_1, 1, 0);  slice_scatter_default = copy_default_1 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(select_scatter_default, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "i64[4, 512]" = torch.ops.aten.where.self(eq_scalar, full_default_2, select_scatter_default);  eq_scalar = full_default_2 = select_scatter_default = None
        return where_self

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
