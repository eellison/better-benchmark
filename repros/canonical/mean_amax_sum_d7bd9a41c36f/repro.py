"""
Standalone repro captured via capture_hook.
Label: torchbench_hf_T5_infer
Pattern hash: d7bd9a41c36f
Shape hash: 5f7b5576
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
_shapes_config = "(T([32128, 512], f16), T([1, 2048], i64, gen=Index(32128)), T([512], f16), T([512, 512], f16), T([512, 512], f16), T([8, 2048, 2048], f16), T([1, 8, 2048, 2048], f16, stride=(8, 1, 16384, 8)), T([2048, 512], f16), S([2048, 512]), S([2048, 512]), S([1, 8, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]), S([1, 2048, 512]), S([1, 2048, -1, 64]), S([1, 8, 2048, 64]), S([8, 2048, 64]))"

class Repro(torch.nn.Module):
    def forward(self, arg1_1: "f16[32128, 512]", arg0_1: "i64[1, 2048]", arg52_1: "f16[512]", arg53_1: "f16[512, 512]", arg54_1: "f16[512, 512]", bmm_10: "f16[8, 2048, 2048]", add_7: "f16[1, 8, 2048, 2048]", mm_32: "f16[2048, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f16[1, 2048, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_default: "f32[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32)
        pow_tensor_scalar: "f32[1, 2048, 512]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        mean_dim: "f32[1, 2048, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[1, 2048, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[1, 2048, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[1, 2048, 512]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default_1: "f16[1, 2048, 512]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[1, 2048, 512]" = torch.ops.aten.mul.Tensor(arg52_1, convert_element_type_default_1);  arg52_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f16[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f16[2048, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default_1: "f16[512, 512]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_2: "f16[1, 8, 2048, 2048]" = torch.ops.aten.reshape.default(bmm_10, _shape_param_2);  bmm_10 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_1: "f16[1, 8, 2048, 2048]" = torch.ops.aten.add.Tensor(reshape_default_2, add_7);  reshape_default_2 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_default_2: "f32[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        amax_default: "f32[1, 8, 2048, 1]" = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor: "f32[1, 8, 2048, 2048]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default);  convert_element_type_default_2 = amax_default = None
        exp_default: "f32[1, 8, 2048, 2048]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[1, 8, 2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[1, 8, 2048, 2048]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_3: "f16[1, 8, 2048, 2048]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f16[1, 8, 2048, 2048]" = torch.ops.aten.expand.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
        reshape_default_3: "f16[8, 2048, 2048]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f16[1, 2048, 512]" = torch.ops.aten.reshape.default(mm_32, _shape_param_5);  mm_32 = _shape_param_5 = None
        reshape_default_5: "f16[1, 2048, 8, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None
        permute_default_2: "f16[1, 8, 2048, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f16[1, 8, 2048, 64]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_7);  permute_default_2 = _shape_param_7 = None
        reshape_default_6: "f16[8, 2048, 64]" = torch.ops.aten.reshape.default(expand_default_1, _shape_param_8);  expand_default_1 = _shape_param_8 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_3, reshape_default_6)



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
