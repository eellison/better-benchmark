"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:

"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_578: "bf16[2048, 3072]", mm_4: "bf16[2048, 3072]", mm_5: "bf16[2048, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_578, [4, 512, 3072]);  mm_578 = None
        reshape_default_1: "bf16[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 3072]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
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
        permute_default: "bf16[3072, 2048]" = torch.ops.aten.permute.default(reshape_default_3, [1, 0]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:99 in forward, code: return nn.functional.silu(input)
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
        permute_default_1: "bf16[3072, 2048]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        return (permute_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 3072], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 3072], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
