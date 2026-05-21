"""
Standalone repro captured via capture_hook.
Label: vllm_facebook_opt-125m_004
Pattern hash: 1d416376ef2f
Shape hash: 49b9db20
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
_shapes_config = "(T([4, 512, 768], f16), T([4, 512], i64, gen=Index(2050)))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[4, 512, 768]", arg0_1: "i64[4, 512]"):
        # No stacktrace found for following nodes
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(arg0_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, convert_element_type_default);  unsqueeze_default = full_default = convert_element_type_default = None
        full_default_1: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2050, 768]" = torch.ops.aten.index_put.default(full_default_1, [arg0_1], where_self, True);  full_default_1 = arg0_1 = where_self = None
        convert_element_type_default_1: "f16[2050, 768]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.float16);  index_put_default = None
        return convert_element_type_default_1



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
