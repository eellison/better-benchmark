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
    def forward(self, gt: "b8[8, 4096, 256]", tangents_1: "f32[8, 4096, 256]", primals_1: "i64[8, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:365 in forward, code: embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        convert_element_type_default: "f32[8, 4096, 256]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.0526315789473684);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 4096, 256]" = torch.ops.aten.mul.Tensor(tangents_1, mul_tensor);  tangents_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/reformer/modeling_reformer.py:356 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar: "b8[8, 4096]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[8, 4096, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[8, 4096, 256]" = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_1);  unsqueeze_default = full_default = mul_tensor_1 = None
        full_default_1: "f32[320, 256]" = torch.ops.aten.full.default([320, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[320, 256]" = torch.ops.aten.index_put.default(full_default_1, [primals_1], where_self, True);  full_default_1 = primals_1 = where_self = None
        return index_put_default


def _default_make_inputs():
    return [
    torch.randint(0, 2, [8, 4096, 256], dtype=torch.bool, device='cuda'),
    torch.randn([8, 4096, 256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 320, [8, 4096], dtype=torch.int64, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
