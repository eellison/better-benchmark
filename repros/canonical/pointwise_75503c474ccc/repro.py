"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_Bart_train
Pattern hash: 75503c474ccc
Shape hash: 79a6b47b
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
_shapes_config = "(T([4, 512, 768], f32), T([4, 512], i64, gen=Index(50265)))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[4, 512, 768]", primals_2: "i64[4, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/bart/modeling_bart.py:111 in forward, code: return super().forward(input_ids) * self.embed_scale
        mul_tensor: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(tangents_1, 1.0);  tangents_1 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_2, 1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor);  unsqueeze_default = full_default = mul_tensor = None
        full_default_1: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50265, 768]" = torch.ops.aten.index_put.default(full_default_1, [primals_2], where_self, True);  full_default_1 = primals_2 = where_self = None
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
