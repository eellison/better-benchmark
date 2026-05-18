"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 1024, 1024]", primals_2: "i64[8, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:116 in forward, code: return super().forward(input_ids) * self.embed_scale
        mul_tensor: "f32[8, 1024, 1024]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        eq_scalar: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_2, 1)
        unsqueeze_default: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 1024, 1024]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor);  unsqueeze_default = full_default = mul_tensor = None
        full_default_1: "f32[50265, 1024]" = torch.ops.aten.full.default([50265, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50265, 1024]" = torch.ops.aten.index_put.default(full_default_1, [primals_2], where_self, True);  full_default_1 = primals_2 = where_self = None
        return index_put_default


def _default_make_inputs():
    return [
    torch.randn([8, 1024, 1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
