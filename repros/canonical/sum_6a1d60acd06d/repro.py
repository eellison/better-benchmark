"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 6a1d60acd06d
Shape hash: 3451cca9
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
    def forward(self, mm_335: "f32[512, 768]", _shape_param_0, mm_337: "f32[512, 768]", _shape_param_1, mm_339: "f32[512, 768]", _shape_param_2, primals_17: "f32[768]", add_16: "f32[4, 128, 768]", rsqrt_3: "f32[4, 128, 1]", add_262: "f32[4, 128, 768]", _shape_param_3, gt_6: "b8[4, 128, 768]", _shape_param_4, primals_16: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_335, _shape_param_0);  mm_335 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_337, _shape_param_1);  mm_337 = _shape_param_1 = None
        add_tensor: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[4, 128, 768]" = torch.ops.aten.reshape.default(mm_339, _shape_param_2);  mm_339 = _shape_param_2 = None
        add_tensor_1: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_2);  add_tensor = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_17);  add_tensor_1 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, add_16)
        mul_tensor_2: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor, rsqrt_3);  mul_tensor = None
        sum_dim_int_list: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        add_tensor_2: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_262, mul_tensor_2);  add_262 = mul_tensor_2 = None
        pow_tensor_scalar: "f32[4, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_scalar: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 128, 768]" = torch.ops.aten.expand.default(mul_tensor_3, _shape_param_3);  mul_tensor_3 = _shape_param_3 = None
        div_scalar: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
        mul_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_3: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4);  add_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_tensor_5: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_5);  add_tensor_3 = mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_3: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_6, _shape_param_4);  mul_tensor_6 = _shape_param_4 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)


def _default_make_inputs():
    return [
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_1
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_2
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_3
    torch.randint(0, 2, [4, 128, 768], dtype=torch.bool, device='cuda'),
    [512, 768],  # _shape_param_4
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
