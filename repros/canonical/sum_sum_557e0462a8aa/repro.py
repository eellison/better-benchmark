"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '512'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['8', '1024', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_106: "f32[8192, 512]", tangents_3: "f32[8, 1024, 512]", mm_108: "f32[8192, 512]", mm_126: "f32[8192, 512]", mm_128: "f32[8192, 512]", mm_146: "f32[8192, 512]", mm_148: "f32[8192, 512]", mm_166: "f32[8192, 512]", mm_168: "f32[8192, 512]", mm_186: "f32[8192, 512]", mm_188: "f32[8192, 512]", mm_206: "f32[8192, 512]", mm_208: "f32[8192, 512]", gt_26: "b8[8, 1024, 512]", primals_52: "f32[512]", add_33: "f32[8, 1024, 512]", rsqrt_12: "f32[8, 1024, 1]", gt_25: "b8[8, 1024, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_106, [8, 1024, 512]);  mm_106 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_3, reshape_default);  tangents_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_108, [8, 1024, 512]);  mm_108 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_126, [8, 1024, 512]);  mm_126 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_128, [8, 1024, 512]);  mm_128 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_3);  add_tensor_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_4: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_146, [8, 1024, 512]);  mm_146 = None
        add_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_4);  add_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_148, [8, 1024, 512]);  mm_148 = None
        add_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_5);  add_tensor_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_6: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_166, [8, 1024, 512]);  mm_166 = None
        add_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_7: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_168, [8, 1024, 512]);  mm_168 = None
        add_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_7);  add_tensor_6 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_8: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_186, [8, 1024, 512]);  mm_186 = None
        add_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_8);  add_tensor_7 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_9: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_188, [8, 1024, 512]);  mm_188 = None
        add_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_9);  add_tensor_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:514 in forward, code: value_states = self.v(current_states)
        reshape_default_10: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_206, [8, 1024, 512]);  mm_206 = None
        add_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_10);  add_tensor_9 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:513 in forward, code: key_states = self.k(current_states)
        reshape_default_11: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_208, [8, 1024, 512]);  mm_208 = None
        add_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_11);  add_tensor_10 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1137 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_11, mul_tensor);  add_tensor_11 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_52);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_33, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = mul_tensor_3 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        reshape_default_12: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, [512]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_33)
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_12);  mul_tensor_2 = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True);  mul_tensor_5 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_7: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:255 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_7, [8, 1024, 512]);  mul_tensor_7 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_33, 1.0);  add_33 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_8);  mul_tensor_6 = mul_tensor_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:344 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_12, mul_tensor_9);  add_tensor_12 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:297 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_13: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_10, [8192, 512]);  mul_tensor_10 = None
        permute_default: "f32[512, 8192]" = torch.ops.aten.permute.default(reshape_default_13, [1, 0]);  reshape_default_13 = None
        return (reshape_default_12, permute_default)


def _default_make_inputs():
    return [
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8192, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 512], dtype=torch.float32, device='cuda'),
    torch.randn([8, 1024, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 1024, 512], dtype=torch.bool, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
