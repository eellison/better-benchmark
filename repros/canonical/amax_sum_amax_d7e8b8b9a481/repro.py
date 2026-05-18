"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_inference
Pattern hash: d7e8b8b9a481
Shape hash: eb1453ba
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
    def forward(self, bmm_34: "f32[48, 512, 512]", _shape_param_0, _shape_param_1, _shape_param_2, addmm_78: "f32[4096, 384]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, addmm_80: "f32[4096, 384]", _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, mm_11: "f32[4096, 54]", _shape_param_12, arg271_1: "f32[54]", _shape_param_13, _shape_param_14):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:227 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        reshape_default: "f32[8, 6, 512, 512]" = torch.ops.aten.reshape.default(bmm_34, _shape_param_0);  bmm_34 = _shape_param_0 = None

        # No stacktrace found for following nodes
        mul_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.mul.Tensor(reshape_default, 1);  reshape_default = None
        amax_default: "f32[8, 6, 512, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:234 in forward, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        div_tensor: "f32[8, 6, 512, 512]" = torch.ops.aten.div.Tensor(sub_tensor, 8.0);  sub_tensor = None
        exp_default: "f32[8, 6, 512, 512]" = torch.ops.aten.exp.default(div_tensor);  div_tensor = None
        sum_dim_int_list: "f32[8, 6, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor_1: "f32[8, 6, 512, 512]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_default: "f32[8, 6, 512, 512]" = torch.ops.aten.expand.default(div_tensor_1, _shape_param_1);  div_tensor_1 = _shape_param_1 = None
        reshape_default_1: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(expand_default, _shape_param_2);  expand_default = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_2: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(addmm_78, _shape_param_3);  addmm_78 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[8, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_4);  reshape_default_2 = _shape_param_4 = None
        permute_default: "f32[8, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:240 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        expand_default_1: "f32[8, 6, 512, 64]" = torch.ops.aten.expand.default(permute_default, _shape_param_5);  permute_default = _shape_param_5 = None
        clone_default: "f32[8, 6, 512, 64]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_4: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_default, _shape_param_6);  clone_default = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default_5: "f32[8, 512, 384]" = torch.ops.aten.reshape.default(addmm_80, _shape_param_7);  addmm_80 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_default_1: "f32[8, 384, 512]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1]);  reshape_default_5 = None
        clone_default_1: "f32[8, 384, 512]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        unsqueeze_default: "f32[8, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_default_1, -1);  clone_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        constant_pad_nd_default: "f32[8, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 0, 4, 4], 0.0);  unsqueeze_default = None
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        iota_default_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        add_tensor: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        unsqueeze_default_3: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_tensor, -1);  add_tensor = None
        unsqueeze_default_4: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        iota_default_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_5: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 0);  iota_default_2 = None
        iota_default_3: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_6: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_3, -1);  iota_default_3 = None
        add_tensor_1: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_5, unsqueeze_default_6);  unsqueeze_default_5 = unsqueeze_default_6 = None
        index_tensor: "f32[8, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_default, [None, None, unsqueeze_default_4, add_tensor_1]);  constant_pad_nd_default = unsqueeze_default_4 = add_tensor_1 = None
        permute_default_2: "f32[8, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_tensor, [0, 1, 2, 4, 3, 5]);  index_tensor = None
        reshape_default_6: "f32[8, 3456, 512]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_8);  permute_default_2 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_default_3: "f32[8, 512, 3456]" = torch.ops.aten.permute.default(reshape_default_6, [0, 2, 1]);  reshape_default_6 = None
        reshape_default_7: "f32[8, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_default_3, _shape_param_9);  permute_default_3 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_default_2: "f32[8, 512, 384, 9]" = torch.ops.aten.clone.default(reshape_default_7, memory_format = torch.contiguous_format);  reshape_default_7 = None
        reshape_default_8: "f32[24576, 64, 9]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_10);  clone_default_2 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default_2: "f32[24576, 64, 9]" = torch.ops.aten.expand.default(reshape_default_8, _shape_param_11);  reshape_default_8 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_9: "f32[8, 512, 54]" = torch.ops.aten.reshape.default(mm_11, _shape_param_12);  mm_11 = _shape_param_12 = None
        add_tensor_2: "f32[8, 512, 54]" = torch.ops.aten.add.Tensor(reshape_default_9, arg271_1);  reshape_default_9 = arg271_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default_10: "f32[24576, 9, 1]" = torch.ops.aten.reshape.default(add_tensor_2, _shape_param_13);  add_tensor_2 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        amax_default_1: "f32[24576, 1, 1]" = torch.ops.aten.amax.default(reshape_default_10, [1], True)
        sub_tensor_1: "f32[24576, 9, 1]" = torch.ops.aten.sub.Tensor(reshape_default_10, amax_default_1);  reshape_default_10 = amax_default_1 = None
        exp_default_1: "f32[24576, 9, 1]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1: "f32[24576, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default_1, [1], True)
        div_tensor_2: "f32[24576, 9, 1]" = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default_3: "f32[24576, 9, 1]" = torch.ops.aten.expand.default(div_tensor_2, _shape_param_14);  div_tensor_2 = _shape_param_14 = None
        return (reshape_default_1, reshape_default_4, expand_default_2, expand_default_3)


def _default_make_inputs():
    return [
    torch.randn([48, 512, 512], dtype=torch.float32, device='cuda'),
    [8, 6, 512, 512],  # _shape_param_0
    [8, 6, 512, 512],  # _shape_param_1
    [48, 512, 512],  # _shape_param_2
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_3
    [8, 512, -1, 64],  # _shape_param_4
    [8, 6, 512, 64],  # _shape_param_5
    [48, 512, 64],  # _shape_param_6
    torch.randn([4096, 384], dtype=torch.float32, device='cuda'),
    [8, 512, 384],  # _shape_param_7
    [8, 3456, 512],  # _shape_param_8
    [8, 512, 384, 9],  # _shape_param_9
    [24576, 64, 9],  # _shape_param_10
    [24576, 64, 9],  # _shape_param_11
    torch.randn([4096, 54], dtype=torch.float32, device='cuda'),
    [8, 512, 54],  # _shape_param_12
    torch.randn([54], dtype=torch.float32, device='cuda'),
    [-1, 9, 1],  # _shape_param_13
    [24576, 9, 1],  # _shape_param_14
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
