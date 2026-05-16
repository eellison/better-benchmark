"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: d707422832b4
Shape hash: 648da861
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_84: "f32[169, 12]", primals_85: "i64[2401]", _shape_param_0, permute_60: "f32[3, 16, 12, 49, 32]", bmm_10: "f32[192, 49, 49]", _shape_param_1, full_default: "f32[]", full_default_1: "f32[]", full_default_2: "f32[]", full_default_3: "f32[]", full_default_4: "f32[]", full_default_5: "f32[]", full_default_6: "f32[]", full_default_7: "f32[]", full_default_8: "f32[]", _shape_param_2, _shape_param_3, full_default_9: "f32[]", full_default_10: "f32[]", _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_tensor: "f32[2401, 12]" = torch.ops.aten.index.Tensor(primals_84, [primals_85]);  primals_84 = primals_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default: "f32[49, 49, 12]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        permute_default: "f32[12, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [2, 0, 1]);  reshape_default = None
        clone_default: "f32[12, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 12, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int: "f32[16, 12, 49, 32]" = torch.ops.aten.select.int(permute_60, 0, 2);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default_1: "f32[16, 12, 49, 49]" = torch.ops.aten.reshape.default(bmm_10, _shape_param_1);  bmm_10 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        add_tensor: "f32[16, 12, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_1, unsqueeze_default);  reshape_default_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:195 in shifted_window_attention, code: attn_mask = x.new_zeros((pad_H, pad_W))
        full_default: "f32[14, 14]" = torch.ops.aten.full.default([14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:201 in shifted_window_attention, code: attn_mask[h[0] : h[1], w[0] : w[1]] = count
        slice_tensor: "f32[7, 14]" = torch.ops.aten.slice.Tensor(full_default, 0, 0, -7)
        slice_tensor_1: "f32[7, 7]" = torch.ops.aten.slice.Tensor(slice_tensor, 1, 0, -7)
        full_default_1 = full_default
        copy_default: "f32[7, 7]" = torch.ops.aten.copy.default(slice_tensor_1, full_default_1);  slice_tensor_1 = full_default_1 = None
        slice_scatter_default: "f32[7, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default, 1, 0, -7);  slice_tensor = copy_default = None
        slice_scatter_default_1: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_default, 0, 0, -7);  full_default = slice_scatter_default = None
        slice_tensor_2: "f32[7, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 0, -7)
        slice_tensor_3: "f32[7, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_2, 1, -7, -3)
        full_default_2 = full_default_1
        copy_default_1: "f32[7, 4]" = torch.ops.aten.copy.default(slice_tensor_3, full_default_2);  slice_tensor_3 = full_default_2 = None
        slice_scatter_default_2: "f32[7, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_2, copy_default_1, 1, -7, -3);  slice_tensor_2 = copy_default_1 = None
        slice_scatter_default_3: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 0, 0, -7);  slice_scatter_default_1 = slice_scatter_default_2 = None
        slice_tensor_4: "f32[7, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_3, 0, 0, -7)
        slice_tensor_5: "f32[7, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 1, -3, 9223372036854775807)
        full_default_3 = full_default_2
        copy_default_2: "f32[7, 3]" = torch.ops.aten.copy.default(slice_tensor_5, full_default_3);  slice_tensor_5 = full_default_3 = None
        slice_scatter_default_4: "f32[7, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_4, copy_default_2, 1, -3, 9223372036854775807);  slice_tensor_4 = copy_default_2 = None
        slice_scatter_default_5: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_3, slice_scatter_default_4, 0, 0, -7);  slice_scatter_default_3 = slice_scatter_default_4 = None
        slice_tensor_6: "f32[4, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, -7, -3)
        slice_tensor_7: "f32[4, 7]" = torch.ops.aten.slice.Tensor(slice_tensor_6, 1, 0, -7)
        full_default_4 = full_default_3
        copy_default_3: "f32[4, 7]" = torch.ops.aten.copy.default(slice_tensor_7, full_default_4);  slice_tensor_7 = full_default_4 = None
        slice_scatter_default_6: "f32[4, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_default_3, 1, 0, -7);  slice_tensor_6 = copy_default_3 = None
        slice_scatter_default_7: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_5, slice_scatter_default_6, 0, -7, -3);  slice_scatter_default_5 = slice_scatter_default_6 = None
        slice_tensor_8: "f32[4, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, -7, -3)
        slice_tensor_9: "f32[4, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_8, 1, -7, -3)
        full_default_5 = full_default_4
        copy_default_4: "f32[4, 4]" = torch.ops.aten.copy.default(slice_tensor_9, full_default_5);  slice_tensor_9 = full_default_5 = None
        slice_scatter_default_8: "f32[4, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_8, copy_default_4, 1, -7, -3);  slice_tensor_8 = copy_default_4 = None
        slice_scatter_default_9: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_7, slice_scatter_default_8, 0, -7, -3);  slice_scatter_default_7 = slice_scatter_default_8 = None
        slice_tensor_10: "f32[4, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_9, 0, -7, -3)
        slice_tensor_11: "f32[4, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_10, 1, -3, 9223372036854775807)
        full_default_6 = full_default_5
        copy_default_5: "f32[4, 3]" = torch.ops.aten.copy.default(slice_tensor_11, full_default_6);  slice_tensor_11 = full_default_6 = None
        slice_scatter_default_10: "f32[4, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_10, copy_default_5, 1, -3, 9223372036854775807);  slice_tensor_10 = copy_default_5 = None
        slice_scatter_default_11: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_9, slice_scatter_default_10, 0, -7, -3);  slice_scatter_default_9 = slice_scatter_default_10 = None
        slice_tensor_12: "f32[3, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, -3, 9223372036854775807)
        slice_tensor_13: "f32[3, 7]" = torch.ops.aten.slice.Tensor(slice_tensor_12, 1, 0, -7)
        full_default_7 = full_default_6
        copy_default_6: "f32[3, 7]" = torch.ops.aten.copy.default(slice_tensor_13, full_default_7);  slice_tensor_13 = full_default_7 = None
        slice_scatter_default_12: "f32[3, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_12, copy_default_6, 1, 0, -7);  slice_tensor_12 = copy_default_6 = None
        slice_scatter_default_13: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_11, slice_scatter_default_12, 0, -3, 9223372036854775807);  slice_scatter_default_11 = slice_scatter_default_12 = None
        slice_tensor_14: "f32[3, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_13, 0, -3, 9223372036854775807)
        slice_tensor_15: "f32[3, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_14, 1, -7, -3)
        full_default_8 = full_default_7
        copy_default_7: "f32[3, 4]" = torch.ops.aten.copy.default(slice_tensor_15, full_default_8);  slice_tensor_15 = full_default_8 = None
        slice_scatter_default_14: "f32[3, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_14, copy_default_7, 1, -7, -3);  slice_tensor_14 = copy_default_7 = None
        slice_scatter_default_15: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_13, slice_scatter_default_14, 0, -3, 9223372036854775807);  slice_scatter_default_13 = slice_scatter_default_14 = None
        slice_tensor_16: "f32[3, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_default_15, 0, -3, 9223372036854775807)
        slice_tensor_17: "f32[3, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_16, 1, -3, 9223372036854775807)
        full_default_9 = full_default_8
        copy_default_8: "f32[3, 3]" = torch.ops.aten.copy.default(slice_tensor_17, full_default_9);  slice_tensor_17 = full_default_9 = None
        slice_scatter_default_16: "f32[3, 14]" = torch.ops.aten.slice_scatter.default(slice_tensor_16, copy_default_8, 1, -3, 9223372036854775807);  slice_tensor_16 = copy_default_8 = None
        slice_scatter_default_17: "f32[14, 14]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_15, slice_scatter_default_16, 0, -3, 9223372036854775807);  slice_scatter_default_15 = slice_scatter_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:204 in shifted_window_attention, code: attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        reshape_default_2: "f32[2, 7, 2, 7]" = torch.ops.aten.reshape.default(slice_scatter_default_17, _shape_param_2);  slice_scatter_default_17 = _shape_param_2 = None
        permute_default_1: "f32[2, 2, 7, 7]" = torch.ops.aten.permute.default(reshape_default_2, [0, 2, 1, 3]);  reshape_default_2 = None
        clone_default_1: "f32[2, 2, 7, 7]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[4, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_3);  clone_default_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:205 in shifted_window_attention, code: attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        unsqueeze_default_1: "f32[4, 1, 49]" = torch.ops.aten.unsqueeze.default(reshape_default_3, 1)
        unsqueeze_default_2: "f32[4, 49, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_3, 2);  reshape_default_3 = None
        sub_tensor: "f32[4, 49, 49]" = torch.ops.aten.sub.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:206 in shifted_window_attention, code: attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        ne_scalar: "b8[4, 49, 49]" = torch.ops.aten.ne.Scalar(sub_tensor, 0)
        full_default_10 = full_default_9
        where_self: "f32[4, 49, 49]" = torch.ops.aten.where.self(ne_scalar, full_default_10, sub_tensor);  ne_scalar = full_default_10 = None
        eq_scalar: "b8[4, 49, 49]" = torch.ops.aten.eq.Scalar(sub_tensor, 0);  sub_tensor = None
        full_default_11 = full_default_10
        where_self_1: "f32[4, 49, 49]" = torch.ops.aten.where.self(eq_scalar, full_default_11, where_self);  eq_scalar = full_default_11 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:207 in shifted_window_attention, code: attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        reshape_default_4: "f32[4, 4, 12, 49, 49]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_4);  add_tensor = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:208 in shifted_window_attention, code: attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        unsqueeze_default_3: "f32[4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(where_self_1, 1);  where_self_1 = None
        unsqueeze_default_4: "f32[1, 4, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 0);  unsqueeze_default_3 = None
        add_tensor_1: "f32[4, 4, 12, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_4, unsqueeze_default_4);  reshape_default_4 = unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:209 in shifted_window_attention, code: attn = attn.view(-1, num_heads, x.size(1), x.size(1))
        reshape_default_5: "f32[16, 12, 49, 49]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_5);  add_tensor_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:211 in shifted_window_attention, code: attn = F.softmax(attn, dim=-1)
        amax_default: "f32[16, 12, 49, 1]" = torch.ops.aten.amax.default(reshape_default_5, [-1], True)
        sub_tensor_1: "f32[16, 12, 49, 49]" = torch.ops.aten.sub.Tensor(reshape_default_5, amax_default);  reshape_default_5 = amax_default = None
        exp_default: "f32[16, 12, 49, 49]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[16, 12, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 12, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        expand_default: "f32[16, 12, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_6);  div_tensor = _shape_param_6 = None
        reshape_default_6: "f32[192, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_7);  expand_default = _shape_param_7 = None
        expand_default_1: "f32[16, 12, 49, 32]" = torch.ops.aten.expand.default(select_int, _shape_param_8);  select_int = _shape_param_8 = None
        clone_default_2: "f32[16, 12, 49, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "f32[192, 49, 32]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_9);  clone_default_2 = _shape_param_9 = None
        return (reshape_default_6, reshape_default_7)



def make_inputs():
    return [
    torch.randn([169, 12], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    [49, 49, -1],  # _shape_param_0
    torch.randn(903168, dtype=torch.float32, device='cuda').as_strided([3, 16, 12, 49, 32], [384, 56448, 32, 1152, 1]),  # permute_60
    torch.randn([192, 49, 49], dtype=torch.float32, device='cuda'),
    [16, 12, 49, 49],  # _shape_param_1
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cpu'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    [2, 7, 2, 7],  # _shape_param_2
    [4, 49],  # _shape_param_3
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # full_default_11 (unknown shape)
    [4, 4, 12, 49, 49],  # _shape_param_4
    [-1, 12, 49, 49],  # _shape_param_5
    [16, 12, 49, 49],  # _shape_param_6
    [192, 49, 49],  # _shape_param_7
    [16, 12, 49, 32],  # _shape_param_8
    [192, 49, 32],  # _shape_param_9
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
