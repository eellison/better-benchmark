"""
Standalone reduction kernel repro.
Extracted from inductor compilation.

Reduction info:
#   type=sum, ranges=['1', '384'], reduction_ranges=[]
#   origins: ['aten.sum.dim_IntList']
"""
import sys
from pathlib import Path

import torch
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, mm_20: "f32[16384, 768]", div_33: "f32[98304, 9, 1]", unsqueeze_8: "i64[9, 512, 1, 1]", add_7: "i64[1, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:424 in forward, code: hidden_states = self.dense(hidden_states)
        reshape_default: "f32[32, 512, 768]" = torch.ops.aten.reshape.default(mm_20, _shape_param_0);  mm_20 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:410 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        reshape_default_1: "f32[32, 512, 12, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:404 in forward, code: context_layer = torch.cat([context_layer, conv_out], 2)
        slice_tensor: "f32[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(reshape_default_1, 2, 0, 6)
        slice_tensor_1: "f32[32, 512, 6, 64]" = torch.ops.aten.slice.Tensor(reshape_default_1, 2, 6, 12);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:403 in forward, code: conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])
        reshape_default_2: "f32[16384, 384]" = torch.ops.aten.reshape.default(slice_tensor_1, _shape_param_2);  slice_tensor_1 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:401 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_default: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:400 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_default: "f32[32, 6, 512, 64]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:380 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])
        clone_default_1: "f32[16384, 384]" = torch.ops.aten.clone.default(reshape_default_2, memory_format = torch.contiguous_format);  reshape_default_2 = None
        reshape_default_3: "f32[98304, 64, 1]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:379 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[98304, 9, 1]" = torch.ops.aten.expand.default(div_33, _shape_param_4);  div_33 = _shape_param_4 = None
        permute_default_1: "f32[98304, 1, 9]" = torch.ops.aten.permute.default(expand_default, [0, 2, 1]);  expand_default = None
        mul_tensor: "f32[98304, 64, 9]" = torch.ops.aten.mul.Tensor(reshape_default_3, permute_default_1);  reshape_default_3 = permute_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:378 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        reshape_default_4: "f32[32, 512, 384, 9]" = torch.ops.aten.reshape.default(mul_tensor, _shape_param_5);  mul_tensor = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:375 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        reshape_default_5: "f32[32, 512, 3456]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_6);  reshape_default_4 = _shape_param_6 = None
        permute_default_2: "f32[32, 3456, 512]" = torch.ops.aten.permute.default(reshape_default_5, [0, 2, 1]);  reshape_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:368 in forward, code: conv_out_layer = nn.functional.unfold(
        reshape_default_6: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_7);  permute_default_2 = _shape_param_7 = None
        permute_default_3: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.permute.default(reshape_default_6, [0, 1, 2, 4, 3, 5]);  reshape_default_6 = None
        full_default: "f32[32, 384, 520, 1]" = torch.ops.aten.full.default([32, 384, 520, 1], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[32, 384, 520, 1]" = torch.ops.aten.index_put.default(full_default, [None, None, unsqueeze_8, add_7], permute_default_3, True);  full_default = unsqueeze_8 = add_7 = permute_default_3 = None
        constant_pad_nd_default: "f32[32, 384, 512, 1]" = torch.ops.aten.constant_pad_nd.default(index_put_default, [0, 0, -4, -4], 0.0);  index_put_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:367 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        squeeze_dim: "f32[32, 384, 512]" = torch.ops.aten.squeeze.dim(constant_pad_nd_default, -1);  constant_pad_nd_default = None
        permute_default_4: "f32[32, 512, 384]" = torch.ops.aten.permute.default(squeeze_dim, [0, 2, 1]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:365 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        clone_default_2: "f32[32, 512, 384]" = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        reshape_default_7: "f32[16384, 384]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_8);  clone_default_2 = _shape_param_8 = None
        permute_default_5: "f32[384, 16384]" = torch.ops.aten.permute.default(reshape_default_7, [1, 0])
        sum_dim_int_list: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(reshape_default_7, [0], True);  reshape_default_7 = None
        reshape_default_8: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_9);  sum_dim_int_list = _shape_param_9 = None
        return (clone_default, permute_default_5, reshape_default_8)


def _default_make_inputs():
    return [
    torch.randn([16384, 768], dtype=torch.float32, device='cuda'),
    torch.randn([98304, 9, 1], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [9, 512, 1, 1], dtype=torch.int64, device='cuda'),
    torch.randint(0, 2, [1, 1], dtype=torch.int64, device='cuda'),
    [32, 512, 768],  # _shape_param_0
    [32, 512, 12, 64],  # _shape_param_1
    [16384, 384],  # _shape_param_2
    [98304, 64, 1],  # _shape_param_3
    [98304, 9, 1],  # _shape_param_4
    [32, 512, 384, 9],  # _shape_param_5
    [32, 512, 3456],  # _shape_param_6
    [32, 384, 9, 1, 512, 1],  # _shape_param_7
    [16384, 384],  # _shape_param_8
    [384],  # _shape_param_9
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
