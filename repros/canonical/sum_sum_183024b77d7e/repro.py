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
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_98: "f32[8192, 512]", gt_65: "b8[8, 1024, 512]", primals_133: "f32[512]", add_93: "f32[8, 1024, 512]", rsqrt_31: "f32[8, 1024, 1]", gt_64: "b8[8, 1024, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1793 in forward, code: lm_logits = self.lm_head(sequence_output)
        reshape_default: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_98, [8, 1024, 512]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1791 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_tensor: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 0.04419417382415922);  reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1137 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_default: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_tensor_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, primals_133);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_93, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:262 in forward, code: return self.weight * hidden_states
        mul_tensor_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None
        sum_dim_int_list: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_1: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list, [512]);  sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:256 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_3, add_93)
        mul_tensor_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_3, rsqrt_31);  mul_tensor_3 = None
        sum_dim_int_list_1: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True);  mul_tensor_6 = None
        pow_tensor_scalar: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_scalar: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_8: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:255 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_default: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_tensor_8, [8, 1024, 512]);  mul_tensor_8 = None
        div_scalar: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_93, 1.0);  add_93 = None
        mul_scalar_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:344 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_default_1: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_tensor_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_11: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_10);  add_tensor = mul_tensor_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:297 in forward, code: hidden_states = self.wo(hidden_states)
        reshape_default_2: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_tensor_11, [8192, 512]);  mul_tensor_11 = None
        permute_default: "f32[512, 8192]" = torch.ops.aten.permute.default(reshape_default_2, [1, 0]);  reshape_default_2 = None
        return (reshape_default_1, permute_default)


def _default_make_inputs():
    return [
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
