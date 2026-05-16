"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '1', '1024'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
#   type=sum, ranges=['4', '512', '1'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_580: "bf16[2048, 1024]", mm_582: "bf16[2048, 1024]", primals_11: "bf16[1024]", mm_3: "bf16[2048, 1024]", embedding: "bf16[4, 512, 1024]", rsqrt_3: "f32[4, 512, 1]", add_660: "bf16[4, 512, 1024]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_580, [4, 512, 1024]);  mm_580 = None
        reshape_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_582, [4, 512, 1024]);  mm_582 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, primals_11);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:229 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 1024]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:270 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, reshape_default_2);  embedding = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, convert_element_type_default_1);  add_tensor = convert_element_type_default_1 = None
        sum_dim_int_list: "bf16[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        reshape_default_3: "bf16[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list, [1024]);  sum_dim_int_list = None
        convert_element_type_default_2: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_3: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default)
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_3);  convert_element_type_default_2 = None
        sum_dim_int_list_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True);  mul_tensor_3 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_5: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_5, [4, 512, 1024]);  mul_tensor_5 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_6: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_6);  mul_tensor_4 = mul_tensor_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_3: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_660, convert_element_type_default_3);  add_660 = convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:229 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_4: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [2048, 1024]);  add_tensor_3 = None
        permute_default: "bf16[1024, 2048]" = torch.ops.aten.permute.default(reshape_default_4, [1, 0]);  reshape_default_4 = None
        return (reshape_default_3, permute_default)


def _default_make_inputs():
    return [
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1024], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
