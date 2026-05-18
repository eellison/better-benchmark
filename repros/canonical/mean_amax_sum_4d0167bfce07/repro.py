"""
Standalone repro captured via capture_hook.
Label: hf_MT5ForConditionalGeneration_inference
Pattern hash: 4d0167bfce07
Shape hash: 639e2304
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
    def forward(self, arg1_1: "f32[250112, 512]", arg0_1: "i64[8, 128]", arg76_1: "f32[512]", _shape_param_0, arg77_1: "f32[384, 512]", _shape_param_1, arg78_1: "f32[384, 512]", bmm_14: "f32[48, 128, 128]", _shape_param_2, add_7: "f32[8, 6, 128, 128]", _shape_param_3, _shape_param_4, mm_51: "f32[1024, 384]", _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding_default: "f32[8, 128, 512]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        pow_tensor_scalar: "f32[8, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(embedding_default, 2)
        mean_dim: "f32[8, 128, 1]" = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True);  pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        add_tensor: "f32[8, 128, 1]" = torch.ops.aten.add.Tensor(mean_dim, 1e-06);  mean_dim = None
        rsqrt_default: "f32[8, 128, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  embedding_default = rsqrt_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_tensor_1: "f32[8, 128, 512]" = torch.ops.aten.mul.Tensor(arg76_1, mul_tensor);  arg76_1 = mul_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        reshape_default: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_0);  _shape_param_0 = None
        permute_default: "f32[512, 384]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_1: "f32[1024, 512]" = torch.ops.aten.reshape.default(mul_tensor_1, _shape_param_1);  mul_tensor_1 = _shape_param_1 = None
        permute_default_1: "f32[512, 384]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        reshape_default_2: "f32[8, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, _shape_param_2);  bmm_14 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_tensor_1: "f32[8, 6, 128, 128]" = torch.ops.aten.add.Tensor(reshape_default_2, add_7);  reshape_default_2 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        amax_default: "f32[8, 6, 128, 1]" = torch.ops.aten.amax.default(add_tensor_1, [-1], True)
        sub_tensor: "f32[8, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_tensor_1, amax_default);  add_tensor_1 = amax_default = None
        exp_default: "f32[8, 6, 128, 128]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[8, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[8, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default: "f32[8, 6, 128, 128]" = torch.ops.aten.expand.default(div_tensor, _shape_param_3);  div_tensor = _shape_param_3 = None
        reshape_default_3: "f32[48, 128, 128]" = torch.ops.aten.reshape.default(expand_default, _shape_param_4);  expand_default = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        reshape_default_4: "f32[8, 128, 384]" = torch.ops.aten.reshape.default(mm_51, _shape_param_5);  mm_51 = _shape_param_5 = None
        reshape_default_5: "f32[8, 128, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None
        permute_default_2: "f32[8, 6, 128, 64]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1, 3]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_default_1: "f32[8, 6, 128, 64]" = torch.ops.aten.expand.default(permute_default_2, _shape_param_7);  permute_default_2 = _shape_param_7 = None
        clone_default: "f32[8, 6, 128, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_6: "f32[48, 128, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_8);  clone_default = _shape_param_8 = None
        return (reshape_default, permute_default, reshape_default_1, permute_default_1, reshape_default_3, reshape_default_6)


def _default_make_inputs():
    return [
    torch.randn([250112, 512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [8, 128], dtype=torch.int64, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_0
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    [1024, 512],  # _shape_param_1
    torch.randn([384, 512], dtype=torch.float32, device='cuda'),
    torch.randn([48, 128, 128], dtype=torch.float32, device='cuda'),
    [8, 6, 128, 128],  # _shape_param_2
    torch.randn(786432, dtype=torch.float32, device='cuda').as_strided([8, 6, 128, 128], [98304, 1, 768, 6]),  # add_7
    [8, 6, 128, 128],  # _shape_param_3
    [48, 128, 128],  # _shape_param_4
    torch.randn([1024, 384], dtype=torch.float32, device='cuda'),
    [8, 128, 384],  # _shape_param_5
    [8, 128, -1, 64],  # _shape_param_6
    [8, 6, 128, 64],  # _shape_param_7
    [48, 128, 64],  # _shape_param_8
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
