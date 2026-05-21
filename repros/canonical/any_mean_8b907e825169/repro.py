"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: 8b907e825169
Shape hash: fc7efc69
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([2048, 512], f16), T([1, 2048, 512], f16), T([512], f16), T([512, 512], f16), T([1, 2048, 512], f16), T([512, 512], f16), S([1, 2048, 512]), S([2048, 512]), S([2048, 512]))"

class Repro(torch.nn.Module):
    def forward(self, mm_89: "f16[2048, 512]", convert_element_type_382: "f16[1, 2048, 512]", arg123_1: "f16[512]", arg124_1: "f16[512, 512]", mul_27: "f16[1, 2048, 512]", arg125_1: "f16[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        reshape_default: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_89, _shape_param_0);  mm_89 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        add_tensor: "f16[1, 2048, 512]" = torch.ops.aten.add.Tensor(convert_element_type_382, reshape_default);  convert_element_type_382 = reshape_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        convert_element_type_default: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:439 in forward, code: torch.isinf(hidden_states).any(),
        isinf_default: "b8[1, 2048, 512]" = torch.ops.aten.isinf.default(add_tensor);  add_tensor = None
        any_default: "b8[]" = torch.ops.aten.any.default(isinf_default);  isinf_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:438 in forward, code: clamp_value = torch.where(
        full_default: "f32[]" = torch.ops.aten.full.default([], 64504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 65504.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[]" = torch.ops.aten.where.self(any_default, full_default, full_default_1);  any_default = full_default = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:443 in forward, code: hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        neg_default: "f32[]" = torch.ops.aten.neg.default(where_self)
        clamp_min_tensor: "f32[1, 2048, 512]" = torch.ops.aten.clamp_min.Tensor(convert_element_type_default, neg_default);  convert_element_type_default = neg_default = None
        clamp_max_tensor: "f32[1, 2048, 512]" = torch.ops.aten.clamp_max.Tensor(clamp_min_tensor, where_self);  clamp_min_tensor = where_self = None
        convert_element_type_default_1: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(clamp_max_tensor, torch.float16);  clamp_max_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_default_2: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32)
        pow_tensor_scalar: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor_1: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default_3: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg123_1, convert_element_type_default_3);  arg123_1 = convert_element_type_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default_1: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_2: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_27, _shape_param_2);  mul_27 = _shape_param_2 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        return (reshape_default_1, permute_default, reshape_default_2, permute_default_1)



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
