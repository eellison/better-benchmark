"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import glob
import os
import torch
from math import inf
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f16[4, 512, 768]", add: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        convert_element_type_default: "f32[4, 512, 768]" = torch.ops.prims.convert_element_type.default(tangents_1, torch.float32);  tangents_1 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(add, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, convert_element_type_default);  unsqueeze_default = full_default = convert_element_type_default = None
        full_default_1: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2050, 768]" = torch.ops.aten.index_put.default(full_default_1, [add], where_self, True);  full_default_1 = add = where_self = None
        convert_element_type_default_1: "f16[2050, 768]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.float16);  index_put_default = None
        return convert_element_type_default_1


def _default_make_inputs():
    return [
    torch.randn([4, 512, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 100, [4, 512], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
