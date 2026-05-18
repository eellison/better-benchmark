"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: f90930a6d194
Shape hash: 3e52ff80
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_410: "bf16[2048, 896]", _shape_param_0, mm_412: "bf16[2048, 896]", _shape_param_1, primals_24: "bf16[896]", add_13: "bf16[4, 512, 896]", rsqrt_3: "f32[4, 512, 1]", _shape_param_2, add_503: "bf16[4, 512, 896]", _shape_param_3, primals_23: "bf16[896, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_410, _shape_param_0);  mm_410 = _shape_param_0 = None
        reshape_default_1: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_412, _shape_param_1);  mm_412 = _shape_param_1 = None
        add_tensor: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_tensor, primals_24);  add_tensor = primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_13, torch.float32);  add_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_3);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:261 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 896]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_2);  mul_tensor_3 = _shape_param_2 = None
        div_scalar: "f32[4, 512, 896]" = torch.ops.aten.div.Scalar(expand_default, 896);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 896]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 896]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[4, 512, 896]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_2: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(add_503, convert_element_type_default_2);  add_503 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_2: "bf16[2048, 896]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_3);  add_tensor_2 = _shape_param_3 = None
        permute_default: "bf16[896, 896]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_default_1: "bf16[896, 896]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_2, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_0
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_1
    torch.randn([896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    [4, 512, 896],  # _shape_param_2
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [2048, 896],  # _shape_param_3
    torch.randn([896, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
