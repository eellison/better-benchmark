"""
Standalone repro captured via capture_hook.
Label: hf_YituTechConvBert_train
Pattern hash: 81b8e6ab04ec
Shape hash: c04d65a6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, addmm: "f32[16384, 384]", addmm_1: "f32[16384, 384]", view_5: "f32[32, 512, 384]", mm: "f32[16384, 54]", primals_19: "f32[54]", addmm_3: "f32[16384, 384]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:191 in forward, code: mixed_key_layer = self.key(hidden_states)
        reshape_default: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm, _shape_param_0);  addmm = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:192 in forward, code: mixed_value_layer = self.value(hidden_states)
        reshape_default_1: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_1, _shape_param_1);  addmm_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:198 in forward, code: query_layer = mixed_query_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_2: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(view_5, _shape_param_2);  view_5 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:200 in forward, code: key_layer = mixed_key_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_3: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_3);  reshape_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:201 in forward, code: value_layer = mixed_value_layer.view(hidden_shape).transpose(1, 2)
        reshape_default_4: "f32[32, 512, 6, 64]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_4);  reshape_default_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:205 in forward, code: conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        reshape_default_5: "f32[32, 512, 54]" = torch.ops.aten.reshape.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        add_tensor: "f32[32, 512, 54]" = torch.ops.aten.add.Tensor(reshape_default_5, primals_19);  reshape_default_5 = primals_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:206 in forward, code: conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])
        reshape_default_6: "f32[98304, 9, 1]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_6);  add_tensor = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:207 in forward, code: conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)
        amax_default: "f32[98304, 1, 1]" = torch.ops.aten.amax.default(reshape_default_6, [1], True)
        sub_tensor: "f32[98304, 9, 1]" = torch.ops.aten.sub.Tensor(reshape_default_6, amax_default);  reshape_default_6 = amax_default = None
        exp_default: "f32[98304, 9, 1]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[98304, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [1], True)
        div_tensor: "f32[98304, 9, 1]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:209 in forward, code: conv_out_layer = self.conv_out_layer(hidden_states)
        reshape_default_7: "f32[32, 512, 384]" = torch.ops.aten.reshape.default(addmm_3, _shape_param_7);  addmm_3 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:211 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        permute_default: "f32[32, 384, 512]" = torch.ops.aten.permute.default(reshape_default_7, [0, 2, 1]);  reshape_default_7 = None
        clone_default: "f32[32, 384, 512]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[32, 384, 512, 1]" = torch.ops.aten.unsqueeze.default(clone_default, -1);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:212 in forward, code: conv_out_layer = nn.functional.unfold(
        iota_default: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_1: "i64[1, 512]" = torch.ops.aten.unsqueeze.default(iota_default, 0);  iota_default = None
        iota_default_1: "i64[9]" = torch.ops.prims.iota.default(9, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_2: "i64[9, 1]" = torch.ops.aten.unsqueeze.default(iota_default_1, -1);  iota_default_1 = None
        add_tensor_1: "i64[9, 512]" = torch.ops.aten.add.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None
        iota_default_2: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_default_3: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, 0)
        unsqueeze_default_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_default_2, -1);  iota_default_2 = None
        add_tensor_2: "i64[1, 1]" = torch.ops.aten.add.Tensor(unsqueeze_default_3, unsqueeze_default_4);  unsqueeze_default_3 = unsqueeze_default_4 = None
        constant_pad_nd_default: "f32[32, 384, 520, 1]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 0, 4, 4], 0.0);  unsqueeze_default = None
        unsqueeze_default_5: "i64[9, 512, 1]" = torch.ops.aten.unsqueeze.default(add_tensor_1, -1);  add_tensor_1 = None
        unsqueeze_default_6: "i64[9, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        index_tensor: "f32[32, 384, 9, 512, 1, 1]" = torch.ops.aten.index.Tensor(constant_pad_nd_default, [None, None, unsqueeze_default_6, add_tensor_2]);  constant_pad_nd_default = unsqueeze_default_6 = add_tensor_2 = None
        permute_default_1: "f32[32, 384, 9, 1, 512, 1]" = torch.ops.aten.permute.default(index_tensor, [0, 1, 2, 4, 3, 5]);  index_tensor = None
        reshape_default_8: "f32[32, 3456, 512]" = torch.ops.aten.reshape.default(permute_default_1, _shape_param_8);  permute_default_1 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:219 in forward, code: conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
        permute_default_2: "f32[32, 512, 3456]" = torch.ops.aten.permute.default(reshape_default_8, [0, 2, 1]);  reshape_default_8 = None
        reshape_default_9: "f32[32, 512, 384, 9]" = torch.ops.aten.reshape.default(permute_default_2, _shape_param_9);  permute_default_2 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:222 in forward, code: conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])
        clone_default_1: "f32[32, 512, 384, 9]" = torch.ops.aten.clone.default(reshape_default_9, memory_format = torch.contiguous_format);  reshape_default_9 = None
        reshape_default_10: "f32[98304, 64, 9]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_10);  clone_default_1 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py:223 in forward, code: conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        expand_default: "f32[98304, 64, 9]" = torch.ops.aten.expand.default(reshape_default_10, _shape_param_11);  reshape_default_10 = _shape_param_11 = None
        expand_default_1: "f32[98304, 9, 1]" = torch.ops.aten.expand.default(div_tensor, _shape_param_12);  div_tensor = _shape_param_12 = None

        # No stacktrace found for following nodes
        permute_default_3: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        permute_default_4: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        permute_default_5: "f32[32, 6, 512, 64]" = torch.ops.aten.permute.default(reshape_default_4, [0, 2, 1, 3]);  reshape_default_4 = None
        return (expand_default, expand_default_1, permute_default_3, permute_default_4, permute_default_5)


def _default_make_inputs():
    return [
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    torch.randn([32, 512, 384], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 54], dtype=torch.float32, device='cuda'),
    torch.randn([54], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 384], dtype=torch.float32, device='cuda'),
    [32, 512, 384],  # _shape_param_0
    [32, 512, 384],  # _shape_param_1
    [32, 512, -1, 64],  # _shape_param_2
    [32, 512, -1, 64],  # _shape_param_3
    [32, 512, -1, 64],  # _shape_param_4
    [32, 512, 54],  # _shape_param_5
    [-1, 9, 1],  # _shape_param_6
    [32, 512, 384],  # _shape_param_7
    [32, 3456, 512],  # _shape_param_8
    [32, 512, 384, 9],  # _shape_param_9
    [98304, 64, 9],  # _shape_param_10
    [98304, 64, 9],  # _shape_param_11
    [98304, 9, 1],  # _shape_param_12
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
