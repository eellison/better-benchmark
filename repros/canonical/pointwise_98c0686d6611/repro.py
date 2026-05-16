"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 98c0686d6611
Shape hash: a596654d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_422: "bf16[2048, 4864]", _shape_param_0, mm_1: "bf16[2048, 4864]", _shape_param_1, mm_2: "bf16[2048, 4864]", _shape_param_2, _shape_param_3, primals_14: "bf16[4864, 896]", _shape_param_4, primals_13: "bf16[4864, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 4864]" = torch.ops.aten.reshape.default(mm_422, _shape_param_0);  mm_422 = _shape_param_0 = None
        reshape_default_1: "bf16[4, 512, 4864]" = torch.ops.aten.reshape.default(mm_1, _shape_param_1);  mm_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default: "f32[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(reshape_default_1, torch.float32);  reshape_default_1 = None
        neg_default: "f32[4, 512, 4864]" = torch.ops.aten.neg.default(convert_element_type_default)
        exp_default: "f32[4, 512, 4864]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 512, 4864]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 512, 4864]" = torch.ops.aten.div.Tensor(convert_element_type_default, add_tensor)
        convert_element_type_default_1: "bf16[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        mul_tensor: "bf16[4, 512, 4864]" = torch.ops.aten.mul.Tensor(reshape_default, convert_element_type_default_1);  convert_element_type_default_1 = None
        reshape_default_2: "bf16[4, 512, 4864]" = torch.ops.aten.reshape.default(mm_2, _shape_param_2);  mm_2 = _shape_param_2 = None
        mul_tensor_1: "bf16[4, 512, 4864]" = torch.ops.aten.mul.Tensor(reshape_default, reshape_default_2);  reshape_default = reshape_default_2 = None
        reshape_default_3: "bf16[2048, 4864]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_3);  mul_tensor = _shape_param_3 = None
        permute_default: "bf16[896, 4864]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_default_1: "bf16[4864, 896]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        convert_element_type_default_2: "f32[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.float32);  mul_tensor_1 = None
        reciprocal_default: "f32[4, 512, 4864]" = torch.ops.aten.reciprocal.default(add_tensor);  add_tensor = None
        mul_tensor_2: "f32[4, 512, 4864]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        mul_tensor_3: "f32[4, 512, 4864]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_2);  convert_element_type_default_2 = None
        sub_tensor: "f32[4, 512, 4864]" = torch.ops.aten.sub.Tensor(1, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_4: "f32[4, 512, 4864]" = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        add_tensor_1: "f32[4, 512, 4864]" = torch.ops.aten.add.Tensor(mul_tensor_4, 1);  mul_tensor_4 = None
        mul_tensor_5: "f32[4, 512, 4864]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_1);  mul_tensor_3 = add_tensor_1 = None
        convert_element_type_default_3: "bf16[4, 512, 4864]" = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default_4: "bf16[2048, 4864]" = torch.ops.aten.reshape.default(convert_element_type_default_3, _shape_param_4);  convert_element_type_default_3 = _shape_param_4 = None
        permute_default_2: "bf16[896, 4864]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_default_3: "bf16[4864, 896]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_3, permute_default_1, reshape_default_4, permute_default_3)


def _default_make_inputs():
    return [
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 4864],  # _shape_param_0
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 4864],  # _shape_param_1
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 4864],  # _shape_param_2
    [2048, 4864],  # _shape_param_3
    torch.randn([4864, 896], dtype=torch.bfloat16, device='cuda'),
    [2048, 4864],  # _shape_param_4
    torch.randn([4864, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
