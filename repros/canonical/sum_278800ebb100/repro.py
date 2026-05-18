"""
Standalone repro captured via capture_hook.
Label: hf_t5_base_train
Pattern hash: 278800ebb100
Shape hash: d9fb0745
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
    def forward(self, gt_73: "b8[4, 128, 768]", tangents_49: "f32[4, 128, 768]", primals_160: "f32[768]", add_104: "f32[4, 128, 768]", rsqrt_36: "f32[4, 128, 1]", _shape_param_0, gt_72: "b8[4, 128, 768]", _shape_param_1, primals_159: "f32[768, 3072]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in torch_dynamo_resume_in_forward_at_681, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_73, torch.float32);  gt_73 = None
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(tangents_49, mul_tensor);  tangents_49 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_2: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_1, primals_160);  mul_tensor_1 = primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, add_104)
        mul_tensor_4: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_tensor_2, rsqrt_36);  mul_tensor_2 = None
        sum_dim_int_list: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_36, 3);  rsqrt_36 = None
        mul_scalar: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_5: "f32[4, 128, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 128, 768]" = torch.ops.aten.expand.default(mul_tensor_5, _shape_param_0);  mul_tensor_5 = _shape_param_0 = None
        div_scalar: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 1.0);  add_104 = None
        mul_scalar_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_tensor_7: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_7);  add_tensor = mul_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_1);  mul_tensor_8 = _shape_param_1 = None
        permute_default: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_default_1: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default, permute_default_1)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [4, 128, 768], dtype=torch.bool, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 128, 1], dtype=torch.float32, device='cuda'),
    [4, 128, 768],  # _shape_param_0
    torch.randint(0, 2, [4, 128, 768], dtype=torch.bool, device='cuda'),
    [512, 768],  # _shape_param_1
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
