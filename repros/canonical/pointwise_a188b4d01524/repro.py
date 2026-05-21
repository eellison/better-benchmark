"""
Standalone repro captured via capture_hook.
Label: hf_MobileBertForMaskedLM_train_014
Pattern hash: a188b4d01524
Shape hash: c92ff449
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
_shapes_config = "(T([], f32), T([256, 128, 512], f32), T([256, 128], i64, gen=Index(2)))"

class Repro(torch.nn.Module):
    def forward(self, full_1: "f32[]", mul_538: "f32[256, 128, 512]", arg583_1: "i64[256, 128]"):
        # No stacktrace found for following nodes
        full_default: "b8[256, 128, 1]" = torch.ops.aten.full.default([256, 128, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[256, 128, 512]" = torch.ops.aten.where.self(full_default, full_1, mul_538);  full_default = full_1 = mul_538 = None
        full_default_1: "f32[2, 512]" = torch.ops.aten.full.default([2, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2, 512]" = torch.ops.aten.index_put.default(full_default_1, [arg583_1], where_self, True);  full_default_1 = arg583_1 = where_self = None
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
