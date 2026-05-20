"""
Standalone repro captured via capture_hook.
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([2048, 1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), T([1024, 2048], bf16))"

class Repro(torch.nn.Module):
    def forward(self, mm_580: "bf16[2048, 1024]", mm_582: "bf16[2048, 1024]", primals_11: "bf16[1024]", mm_3: "bf16[2048, 1024]", embedding: "bf16[4, 512, 1024]", rsqrt_3: "f32[4, 512, 1]", add_662: "bf16[4, 512, 1024]", primals_10: "bf16[1024, 2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:82 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        reshape_default: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_580, [4, 512, 1024]);  mm_580 = None
        reshape_default_1: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_582, [4, 512, 1024]);  mm_582 = None
        add_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(reshape_default, reshape_default_1);  reshape_default = reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor: "bf16[4, 512, 1024]" = torch.ops.aten.mul.Tensor(add_tensor, primals_11);  add_tensor = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_2: "bf16[4, 512, 1024]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 1024]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:327 in forward, code: hidden_states = residual + hidden_states
        add_tensor_1: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(embedding, reshape_default_2);  embedding = reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:64 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f32[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_3);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:62 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 1024]" = torch.ops.aten.expand.default(mul_tensor_3, [4, 512, 1024]);  mul_tensor_3 = None
        div_scalar: "f32[4, 512, 1024]" = torch.ops.aten.div.Scalar(expand_default, 1024);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 1024]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 1024]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 1024]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:61 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_2: "bf16[4, 512, 1024]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 1024]" = torch.ops.aten.add.Tensor(add_662, convert_element_type_default_2);  add_662 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:290 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 1024]" = torch.ops.aten.reshape.default(add_tensor_3, [2048, 1024]);  add_tensor_3 = None
        permute_default: "bf16[2048, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_1: "bf16[1024, 2048]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_3, permute_default_1)


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
