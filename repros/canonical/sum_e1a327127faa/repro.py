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

_shapes_config = "(T([16384, 1], b8), T([], bf16), T([16384, 2048], bf16), T([16384], i64, max=100), T([2048, 2048], bf16), T([2048], bf16), T([4, 512, 2048], bf16), T([4, 512, 1], f32), T([4, 512, 2048], bf16), T([2048, 4096], bf16))"

class Repro(torch.nn.Module):
    def forward(self, unsqueeze_36: "b8[16384, 1]", full_default_2: "bf16[]", _grouped_mm_11: "bf16[16384, 2048]", div_14: "i64[16384]", mm_24: "bf16[2048, 2048]", primals_44: "bf16[2048]", add_35: "bf16[4, 512, 2048]", rsqrt_15: "f32[4, 512, 1]", convert_element_type_107: "bf16[4, 512, 2048]", primals_43: "bf16[2048, 4096]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:433 in grouped_mm_experts_forward, code: selected_hidden_states_g.masked_fill_(sentinel_mask, 0.0)
        where_self: "bf16[16384, 2048]" = torch.ops.aten.where.self(unsqueeze_36, full_default_2, _grouped_mm_11);  unsqueeze_36 = full_default_2 = _grouped_mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/moe.py:390 in grouped_mm_experts_forward, code: selected_hidden_states_g = hidden_states[perm // num_top_k]
        full_default: "bf16[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "bf16[2048, 2048]" = torch.ops.aten.index_put.default(full_default, [div_14], where_self, True);  full_default = div_14 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:264 in forward, code: hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        add_tensor: "bf16[2048, 2048]" = torch.ops.aten.add.Tensor(index_put_default, mm_24);  index_put_default = mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:283 in forward, code: hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        reshape_default: "bf16[4, 512, 2048]" = torch.ops.aten.reshape.default(add_tensor, [4, 512, 2048]);  add_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor: "bf16[4, 512, 2048]" = torch.ops.aten.mul.Tensor(reshape_default, primals_44);  reshape_default = primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:304 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "f32[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:303 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, convert_element_type_default)
        mul_tensor_2: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_15);  convert_element_type_default_1 = None
        sum_dim_int_list: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True);  mul_tensor_1 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list, -0.5);  sum_dim_int_list = None
        mul_tensor_3: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:302 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 2048]" = torch.ops.aten.expand.default(mul_tensor_3, [4, 512, 2048]);  mul_tensor_3 = None
        div_scalar: "f32[4, 512, 2048]" = torch.ops.aten.div.Scalar(expand_default, 2048);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 1.0);  convert_element_type_default = None
        mul_scalar_1: "f32[4, 512, 2048]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_4: "f32[4, 512, 2048]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1: "f32[4, 512, 2048]" = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:301 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_2: "bf16[4, 512, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2: "bf16[4, 512, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_107, convert_element_type_default_2);  convert_element_type_107 = convert_element_type_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py:194 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_1: "bf16[2048, 2048]" = torch.ops.aten.reshape.default(add_tensor_2, [2048, 2048]);  add_tensor_2 = None
        permute_default: "bf16[4096, 2048]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_default_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        return (reshape_default_1, permute_default_1)


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
