"""
Standalone repro captured via capture_hook.
Label: hf_OPTForCausalLM_train
Pattern hash: 693e171c3299
Shape hash: d3f3ead3
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4, 2048], i64, max=2050), T([4, 2048, 768], f32))"

class Repro(torch.nn.Module):
    def forward(self, add: "i64[4, 2048]", tangents_1: "f32[4, 2048, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/opt/modeling_opt.py:70 in forward, code: return super().forward(position_ids + self.offset)
        eq_scalar: "b8[4, 2048]" = torch.ops.aten.eq.Scalar(add, -1)
        unsqueeze_default: "b8[4, 2048, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 2048, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, tangents_1);  unsqueeze_default = full_default = tangents_1 = None
        full_default_1: "f32[2050, 768]" = torch.ops.aten.full.default([2050, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[2050, 768]" = torch.ops.aten.index_put.default(full_default_1, [add], where_self, True);  full_default_1 = add = where_self = None
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
