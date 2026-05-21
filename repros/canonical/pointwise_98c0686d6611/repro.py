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

_repro_version = 2
_shapes_config = "(T([2048, 3072], bf16), T([2048, 3072], bf16), T([2048, 3072], bf16), T([3072, 1024], bf16), T([3072, 1024], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_578: "bf16[2048, 3072]", mm_4: "bf16[2048, 3072]", mm_5: "bf16[2048, 3072]", primals_13: "bf16[3072, 1024]", primals_12: "bf16[3072, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_578, [4, 512, 3072]);  mm_578 = None
        reshape_default_1: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        neg_default: "f32[4, 512, 3072]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 3072]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor)
        convert_element_type_default_1: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_tensor: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, convert_element_type_default_1);  convert_element_type_default_1 = None
        reshape_default_2: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 3072]);  mm_5 = None
        mul_tensor_1: "bf16[4, 512, 3072]" = torch.ops.aten.mul.Tensor(reshape_default, reshape_default_2);  reshape_default = reshape_default_2 = None
        reshape_default_3: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(mul_tensor, [2048, 3072]);  mul_tensor = None
        permute_default: "bf16[1024, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_1: "bf16[3072, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default_2: "f32[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reciprocal_default: "f32[4, 512, 3072]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_2);  convert_element_type_default_2 = None
        sub_tensor: "f32[4, 512, 3072]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        add_tensor_1: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_1);  mul_tensor_3 = add_tensor_1 = None
        convert_element_type_default_3: "bf16[4, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_4: "bf16[2048, 3072]" = torch.ops.aten.reshape.default(convert_element_type_default_3, [2048, 3072]);  convert_element_type_default_3 = None
        permute_default_2: "bf16[1024, 3072]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_default_3: "bf16[3072, 1024]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_3, permute_default_1, reshape_default_4, permute_default_3)


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
