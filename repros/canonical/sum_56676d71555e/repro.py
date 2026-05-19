"""
Standalone repro captured via capture_hook.
Label: hf_T5ForConditionalGeneration_train
Pattern hash: 56676d71555e
Shape hash: 42e72c5c
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_106: "f32[8192, 512]", tangents_3: "f32[8, 1024, 512]", mm_108: "f32[8192, 512]", mm_126: "f32[8192, 512]", mm_128: "f32[8192, 512]", mm_146: "f32[8192, 512]", mm_148: "f32[8192, 512]", mm_166: "f32[8192, 512]", mm_168: "f32[8192, 512]", mm_186: "f32[8192, 512]", mm_188: "f32[8192, 512]", mm_206: "f32[8192, 512]", mm_208: "f32[8192, 512]", gt_26: "b8[8, 1024, 512]", primals_52: "f32[512]", add_36: "f32[8, 1024, 512]", rsqrt_12: "f32[8, 1024, 1]", gt_25: "b8[8, 1024, 512]", primals_51: "f32[512, 2048]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_106, _shape_param_0);  mm_106 = _shape_param_0 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_3, reshape_default);  tangents_3 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_108, _shape_param_1);  mm_108 = _shape_param_1 = None
        add_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_1);  add_tensor = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_126, _shape_param_2);  mm_126 = _shape_param_2 = None
        add_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_2);  add_tensor_1 = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_128, _shape_param_3);  mm_128 = _shape_param_3 = None
        add_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_2, reshape_default_3);  add_tensor_2 = reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_146, _shape_param_4);  mm_146 = _shape_param_4 = None
        add_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_3, reshape_default_4);  add_tensor_3 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_5: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_148, _shape_param_5);  mm_148 = _shape_param_5 = None
        add_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_4, reshape_default_5);  add_tensor_4 = reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_6: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_166, _shape_param_6);  mm_166 = _shape_param_6 = None
        add_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_5, reshape_default_6);  add_tensor_5 = reshape_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_7: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_168, _shape_param_7);  mm_168 = _shape_param_7 = None
        add_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_6, reshape_default_7);  add_tensor_6 = reshape_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_8: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_186, _shape_param_8);  mm_186 = _shape_param_8 = None
        add_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_7, reshape_default_8);  add_tensor_7 = reshape_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_9: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_188, _shape_param_9);  mm_188 = _shape_param_9 = None
        add_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_8, reshape_default_9);  add_tensor_8 = reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_10: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_206, _shape_param_10);  mm_206 = _shape_param_10 = None
        add_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_9, reshape_default_10);  add_tensor_9 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_11: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_208, _shape_param_11);  mm_208 = _shape_param_11 = None
        add_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_tensor_10, reshape_default_11);  add_tensor_10 = reshape_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_11, mul_tensor);  add_tensor_11 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_52);  mul_tensor_1 = primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_36)
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_12);  mul_tensor_2 = None
        sum_dim_int_list: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_12);  mul_tensor_5 = _shape_param_12 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 1.0);  add_36 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_12: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor_12, mul_tensor_7);  add_tensor_12 = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_12: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_13);  mul_tensor_8 = _shape_param_13 = None
        permute_default: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_default_1: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_12, permute_default_1)


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
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    [8, 1024, 512],  # _shape_param_0
    [8, 1024, 512],  # _shape_param_1
    [8, 1024, 512],  # _shape_param_2
    [8, 1024, 512],  # _shape_param_3
    [8, 1024, 512],  # _shape_param_4
    [8, 1024, 512],  # _shape_param_5
    [8, 1024, 512],  # _shape_param_6
    [8, 1024, 512],  # _shape_param_7
    [8, 1024, 512],  # _shape_param_8
    [8, 1024, 512],  # _shape_param_9
    [8, 1024, 512],  # _shape_param_10
    [8, 1024, 512],  # _shape_param_11
    [8, 1024, 512],  # _shape_param_12
    [8192, 512],  # _shape_param_13
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
