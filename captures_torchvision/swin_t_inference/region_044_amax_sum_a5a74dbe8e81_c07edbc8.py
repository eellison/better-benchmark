"""
Standalone repro captured via capture_hook.
Label: swin_t_inference
Pattern hash: a5a74dbe8e81
Shape hash: c07edbc8
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_2: "f32[192, 49, 49]", _shape_param_0, arg21_1: "f32[169, 3]", arg22_1: "i64[2401]", _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, permute_14: "f32[3, 64, 3, 49, 32]", _shape_param_8, _shape_param_9):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default: "f32[64, 3, 49, 49]" = torch.ops.aten.reshape.default(bmm_2, _shape_param_0);  bmm_2 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_tensor: "f32[2401, 3]" = torch.ops.aten.index.Tensor(arg21_1, [arg22_1]);  arg21_1 = arg22_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default_1: "f32[49, 49, 3]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_1);  index_tensor = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        permute_default: "f32[3, 49, 49]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 1]);  reshape_default_1 = None
        clone_default: "f32[3, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 3, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        add_tensor: "f32[64, 3, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default, unsqueeze_default);  reshape_default = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:207 in shifted_window_attention, code: attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        reshape_default_2: "f32[1, 64, 3, 49, 49]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:195 in shifted_window_attention, code: attn_mask = x.new_zeros((pad_H, pad_W))
        full_default: "f32[56, 56]" = torch.ops.aten.full.default([56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:201 in shifted_window_attention, code: attn_mask[h[0] : h[1], w[0] : w[1]] = count
        slice_tensor: "f32[49, 56]" = torch.ops.aten.slice.Tensor(full_default, 0, 0, -7)
        slice_tensor_1: "f32[49, 56]" = torch.ops.aten.slice.Tensor(full_default, 0, 0, -7)
        slice_tensor_2: "f32[49, 49]" = torch.ops.aten.slice.Tensor(slice_tensor_1, 1, 0, -7);  slice_tensor_1 = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default: "f32[49, 49]" = torch.ops.aten.copy.default(slice_tensor_2, full_default_1);  slice_tensor_2 = full_default_1 = None
        slice_scatter_default: "f32[49, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor, copy_default, 1, 0, -7);  slice_tensor = copy_default = None
        slice_scatter_default_1: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_default, 0, 0, -7);  full_default = slice_scatter_default = None
        slice_tensor_3: "f32[49, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 0, -7)
        slice_tensor_4: "f32[49, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_1, 0, 0, -7)
        slice_tensor_5: "f32[49, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_4, 1, -7, -3);  slice_tensor_4 = None
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_1: "f32[49, 4]" = torch.ops.aten.copy.default(slice_tensor_5, full_default_2);  slice_tensor_5 = full_default_2 = None
        slice_scatter_default_2: "f32[49, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_3, copy_default_1, 1, -7, -3);  slice_tensor_3 = copy_default_1 = None
        slice_scatter_default_3: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_1, slice_scatter_default_2, 0, 0, -7);  slice_scatter_default_1 = slice_scatter_default_2 = None
        slice_tensor_6: "f32[49, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_3, 0, 0, -7)
        slice_tensor_7: "f32[49, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_3, 0, 0, -7)
        slice_tensor_8: "f32[49, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_7, 1, -3, 9223372036854775807);  slice_tensor_7 = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 2.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_2: "f32[49, 3]" = torch.ops.aten.copy.default(slice_tensor_8, full_default_3);  slice_tensor_8 = full_default_3 = None
        slice_scatter_default_4: "f32[49, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_6, copy_default_2, 1, -3, 9223372036854775807);  slice_tensor_6 = copy_default_2 = None
        slice_scatter_default_5: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_3, slice_scatter_default_4, 0, 0, -7);  slice_scatter_default_3 = slice_scatter_default_4 = None
        slice_tensor_9: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, -7, -3)
        slice_tensor_10: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_5, 0, -7, -3)
        slice_tensor_11: "f32[4, 49]" = torch.ops.aten.slice.Tensor(slice_tensor_10, 1, 0, -7);  slice_tensor_10 = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 3.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_3: "f32[4, 49]" = torch.ops.aten.copy.default(slice_tensor_11, full_default_4);  slice_tensor_11 = full_default_4 = None
        slice_scatter_default_6: "f32[4, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_9, copy_default_3, 1, 0, -7);  slice_tensor_9 = copy_default_3 = None
        slice_scatter_default_7: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_5, slice_scatter_default_6, 0, -7, -3);  slice_scatter_default_5 = slice_scatter_default_6 = None
        slice_tensor_12: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, -7, -3)
        slice_tensor_13: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_7, 0, -7, -3)
        slice_tensor_14: "f32[4, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_13, 1, -7, -3);  slice_tensor_13 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 4.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_4: "f32[4, 4]" = torch.ops.aten.copy.default(slice_tensor_14, full_default_5);  slice_tensor_14 = full_default_5 = None
        slice_scatter_default_8: "f32[4, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_12, copy_default_4, 1, -7, -3);  slice_tensor_12 = copy_default_4 = None
        slice_scatter_default_9: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_7, slice_scatter_default_8, 0, -7, -3);  slice_scatter_default_7 = slice_scatter_default_8 = None
        slice_tensor_15: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_9, 0, -7, -3)
        slice_tensor_16: "f32[4, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_9, 0, -7, -3)
        slice_tensor_17: "f32[4, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_16, 1, -3, 9223372036854775807);  slice_tensor_16 = None
        full_default_6: "f32[]" = torch.ops.aten.full.default([], 5.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_5: "f32[4, 3]" = torch.ops.aten.copy.default(slice_tensor_17, full_default_6);  slice_tensor_17 = full_default_6 = None
        slice_scatter_default_10: "f32[4, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_15, copy_default_5, 1, -3, 9223372036854775807);  slice_tensor_15 = copy_default_5 = None
        slice_scatter_default_11: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_9, slice_scatter_default_10, 0, -7, -3);  slice_scatter_default_9 = slice_scatter_default_10 = None
        slice_tensor_18: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, -3, 9223372036854775807)
        slice_tensor_19: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_11, 0, -3, 9223372036854775807)
        slice_tensor_20: "f32[3, 49]" = torch.ops.aten.slice.Tensor(slice_tensor_19, 1, 0, -7);  slice_tensor_19 = None
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 6.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_6: "f32[3, 49]" = torch.ops.aten.copy.default(slice_tensor_20, full_default_7);  slice_tensor_20 = full_default_7 = None
        slice_scatter_default_12: "f32[3, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_18, copy_default_6, 1, 0, -7);  slice_tensor_18 = copy_default_6 = None
        slice_scatter_default_13: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_11, slice_scatter_default_12, 0, -3, 9223372036854775807);  slice_scatter_default_11 = slice_scatter_default_12 = None
        slice_tensor_21: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_13, 0, -3, 9223372036854775807)
        slice_tensor_22: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_13, 0, -3, 9223372036854775807)
        slice_tensor_23: "f32[3, 4]" = torch.ops.aten.slice.Tensor(slice_tensor_22, 1, -7, -3);  slice_tensor_22 = None
        full_default_8: "f32[]" = torch.ops.aten.full.default([], 7.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_7: "f32[3, 4]" = torch.ops.aten.copy.default(slice_tensor_23, full_default_8);  slice_tensor_23 = full_default_8 = None
        slice_scatter_default_14: "f32[3, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_21, copy_default_7, 1, -7, -3);  slice_tensor_21 = copy_default_7 = None
        slice_scatter_default_15: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_13, slice_scatter_default_14, 0, -3, 9223372036854775807);  slice_scatter_default_13 = slice_scatter_default_14 = None
        slice_tensor_24: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_15, 0, -3, 9223372036854775807)
        slice_tensor_25: "f32[3, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_default_15, 0, -3, 9223372036854775807)
        slice_tensor_26: "f32[3, 3]" = torch.ops.aten.slice.Tensor(slice_tensor_25, 1, -3, 9223372036854775807);  slice_tensor_25 = None
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        copy_default_8: "f32[3, 3]" = torch.ops.aten.copy.default(slice_tensor_26, full_default_9);  slice_tensor_26 = full_default_9 = None
        slice_scatter_default_16: "f32[3, 56]" = torch.ops.aten.slice_scatter.default(slice_tensor_24, copy_default_8, 1, -3, 9223372036854775807);  slice_tensor_24 = copy_default_8 = None
        slice_scatter_default_17: "f32[56, 56]" = torch.ops.aten.slice_scatter.default(slice_scatter_default_15, slice_scatter_default_16, 0, -3, 9223372036854775807);  slice_scatter_default_15 = slice_scatter_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:204 in shifted_window_attention, code: attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        reshape_default_3: "f32[8, 7, 8, 7]" = torch.ops.aten.reshape.default(slice_scatter_default_17, _shape_param_3);  slice_scatter_default_17 = _shape_param_3 = None
        permute_default_1: "f32[8, 8, 7, 7]" = torch.ops.aten.permute.default(reshape_default_3, [0, 2, 1, 3]);  reshape_default_3 = None
        clone_default_1: "f32[8, 8, 7, 7]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[64, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_4);  clone_default_1 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:205 in shifted_window_attention, code: attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        unsqueeze_default_1: "f32[64, 1, 49]" = torch.ops.aten.unsqueeze.default(reshape_default_4, 1)
        unsqueeze_default_2: "f32[64, 49, 1]" = torch.ops.aten.unsqueeze.default(reshape_default_4, 2);  reshape_default_4 = None
        sub_tensor: "f32[64, 49, 49]" = torch.ops.aten.sub.Tensor(unsqueeze_default_1, unsqueeze_default_2);  unsqueeze_default_1 = unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:206 in shifted_window_attention, code: attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        eq_scalar: "b8[64, 49, 49]" = torch.ops.aten.eq.Scalar(sub_tensor, 0)
        full_default_10: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ne_scalar: "b8[64, 49, 49]" = torch.ops.aten.ne.Scalar(sub_tensor, 0)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -100.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self: "f32[64, 49, 49]" = torch.ops.aten.where.self(ne_scalar, full_default_11, sub_tensor);  ne_scalar = full_default_11 = sub_tensor = None
        where_self_1: "f32[64, 49, 49]" = torch.ops.aten.where.self(eq_scalar, full_default_10, where_self);  eq_scalar = full_default_10 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:208 in shifted_window_attention, code: attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        unsqueeze_default_3: "f32[64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(where_self_1, 1);  where_self_1 = None
        unsqueeze_default_4: "f32[1, 64, 1, 49, 49]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 0);  unsqueeze_default_3 = None
        add_tensor_1: "f32[1, 64, 3, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_2, unsqueeze_default_4);  reshape_default_2 = unsqueeze_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:209 in shifted_window_attention, code: attn = attn.view(-1, num_heads, x.size(1), x.size(1))
        reshape_default_5: "f32[64, 3, 49, 49]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_5);  add_tensor_1 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:211 in shifted_window_attention, code: attn = F.softmax(attn, dim=-1)
        amax_default: "f32[64, 3, 49, 1]" = torch.ops.aten.amax.default(reshape_default_5, [-1], True)
        sub_tensor_1: "f32[64, 3, 49, 49]" = torch.ops.aten.sub.Tensor(reshape_default_5, amax_default);  reshape_default_5 = amax_default = None
        exp_default: "f32[64, 3, 49, 49]" = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list: "f32[64, 3, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[64, 3, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        expand_default: "f32[64, 3, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_6);  div_tensor = _shape_param_6 = None
        reshape_default_6: "f32[192, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_7);  expand_default = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int: "f32[64, 3, 49, 32]" = torch.ops.aten.select.int(permute_14, 0, 2);  permute_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        expand_default_1: "f32[64, 3, 49, 32]" = torch.ops.aten.expand.default(select_int, _shape_param_8);  select_int = _shape_param_8 = None
        clone_default_2: "f32[64, 3, 49, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_7: "f32[192, 49, 32]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_9);  clone_default_2 = _shape_param_9 = None
        return (reshape_default_6, reshape_default_7)



def make_inputs():
    return [
    torch.randn([192, 49, 49], dtype=torch.float32, device='cuda'),
    [64, 3, 49, 49],  # _shape_param_0
    torch.randn([169, 3], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    [49, 49, -1],  # _shape_param_1
    [1, 64, 3, 49, 49],  # _shape_param_2
    [8, 7, 8, 7],  # _shape_param_3
    [64, 49],  # _shape_param_4
    [-1, 3, 49, 49],  # _shape_param_5
    [64, 3, 49, 49],  # _shape_param_6
    [192, 49, 49],  # _shape_param_7
    torch.randn(903168, dtype=torch.float32, device='cuda').as_strided([3, 64, 3, 49, 32], [96, 14112, 32, 288, 1]),  # permute_14
    [64, 3, 49, 32],  # _shape_param_8
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
