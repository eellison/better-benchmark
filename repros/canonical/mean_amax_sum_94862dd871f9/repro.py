"""
Standalone repro captured via capture_hook.
Label: t5_base
Pattern hash: 94862dd871f9
Shape hash: b11ee5a0
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
    def forward(self, arg1_1: "f16[32128, 768]", arg101_1: "i64[4, 128]", arg102_1: "f16[768]", _shape_param_0, arg103_1: "f16[768, 768]", _shape_param_1, arg104_1: "f16[768, 768]", bmm_22: "f16[48, 512, 512]", _shape_param_2, add_7: "f16[4, 12, 512, 512]", _shape_param_3, _shape_param_4, mm_68: "f16[2048, 768]", _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f16[4, 128, 768]" = torch.ops.aten.embedding.default(arg1_1, arg101_1);  arg1_1 = arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        convert_element_type_default: "f32[4, 128, 768]" = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32)
        pow_tensor_scalar: "f32[4, 128, 768]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        mean_dim: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[4, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:66 in forward, code: hidden_states = hidden_states.to(self.weight.dtype)
        convert_element_type_default_1: "f16[4, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float16);  mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f16[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg102_1, convert_element_type_default_1);  arg102_1 = convert_element_type_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f16[768, 768]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f16[512, 768]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default_1: "f16[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_2: "f16[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_22, _shape_param_2);  bmm_22 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_tensor_1: "f16[4, 12, 512, 512]" = torch.ops.aten.add.Tensor(reshape_default_2, add_7);  reshape_default_2 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_default_2: "f32[4, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        amax_default: "f32[4, 12, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default);  convert_element_type_default_2 = amax_default = None
        exp_default: "f32[4, 12, 512, 512]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_3: "f16[4, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(div_tensor, torch.float16);  div_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f16[4, 12, 512, 512]" = torch.ops.aten.expand.default(convert_element_type_default_3, _shape_param_3);  convert_element_type_default_3 = _shape_param_3 = None
        reshape_default_3: "f16[48, 512, 512]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f16[4, 512, 768]" = torch.ops.aten.reshape.default(mm_68, _shape_param_5);  mm_68 = _shape_param_5 = None
        reshape_default_5: "f16[4, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None
        permute_default_2: "f16[4, 12, 512, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f16[4, 12, 512, 64]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_7);  permute_default_2 = _shape_param_7 = None
        clone_default: "f16[4, 12, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f16[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_8);  clone_default = _shape_param_8 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_3, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([32128, 768], dtype=torch.float16, device='cuda'),
    torch.randint(0, 32128, [4, 128], dtype=torch.int64, device='cuda'),
    torch.randn([768], dtype=torch.float16, device='cuda'),
    [512, 768],  # _shape_param_0
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    [512, 768],  # _shape_param_1
    torch.randn([768, 768], dtype=torch.float16, device='cuda'),
    torch.randn([48, 512, 512], dtype=torch.float16, device='cuda'),
    [4, 12, 512, 512],  # _shape_param_2
    torch.randn(12582912, dtype=torch.float16, device='cuda').as_strided([4, 12, 512, 512], [3145728, 1, 6144, 12]),  # add_7
    [4, 12, 512, 512],  # _shape_param_3
    [48, 512, 512],  # _shape_param_4
    torch.randn([2048, 768], dtype=torch.float16, device='cuda'),
    [4, 512, 768],  # _shape_param_5
    [4, 512, -1, 64],  # _shape_param_6
    [4, 12, 512, 64],  # _shape_param_7
    [48, 512, 64],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
