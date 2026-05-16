"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: e5e0e37301a9
Shape hash: c275be59
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_140: "f32[169, 12]", primals_141: "i64[2401]", _shape_param_0, permute_104: "f32[3, 16, 12, 49, 32]", bmm_18: "f32[192, 49, 49]", _shape_param_1, _shape_param_2, unsqueeze_17: "f32[1, 4, 1, 49, 49]", _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:53 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
        index_tensor: "f32[2401, 12]" = torch.ops.aten.index.Tensor(primals_140, [primals_141]);  primals_140 = primals_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:54 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.view(N, N, -1)
        reshape_default: "f32[49, 49, 12]" = torch.ops.aten.reshape.default(index_tensor, _shape_param_0);  index_tensor = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:55 in _get_relative_position_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        permute_default: "f32[12, 49, 49]" = torch.ops.aten.permute.default(reshape_default, [2, 0, 1]);  reshape_default = None
        clone_default: "f32[12, 49, 49]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        unsqueeze_default: "f32[1, 12, 49, 49]" = torch.ops.aten.unsqueeze.default(clone_default, 0);  clone_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int: "f32[16, 12, 49, 32]" = torch.ops.aten.select.int(permute_104, 0, 2);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default_1: "f32[16, 12, 49, 49]" = torch.ops.aten.reshape.default(bmm_18, _shape_param_1);  bmm_18 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:191 in shifted_window_attention, code: attn = attn + relative_position_bias
        add_tensor: "f32[16, 12, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_1, unsqueeze_default);  reshape_default_1 = unsqueeze_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:207 in shifted_window_attention, code: attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        reshape_default_2: "f32[4, 4, 12, 49, 49]" = torch.ops.aten.reshape.default(add_tensor, _shape_param_2);  add_tensor = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:208 in shifted_window_attention, code: attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        add_tensor_1: "f32[4, 4, 12, 49, 49]" = torch.ops.aten.add.Tensor(reshape_default_2, unsqueeze_17);  reshape_default_2 = unsqueeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:209 in shifted_window_attention, code: attn = attn.view(-1, num_heads, x.size(1), x.size(1))
        reshape_default_3: "f32[16, 12, 49, 49]" = torch.ops.aten.reshape.default(add_tensor_1, _shape_param_3);  add_tensor_1 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:211 in shifted_window_attention, code: attn = F.softmax(attn, dim=-1)
        amax_default: "f32[16, 12, 49, 1]" = torch.ops.aten.amax.default(reshape_default_3, [-1], True)
        sub_tensor: "f32[16, 12, 49, 49]" = torch.ops.aten.sub.Tensor(reshape_default_3, amax_default);  reshape_default_3 = amax_default = None
        exp_default: "f32[16, 12, 49, 49]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list: "f32[16, 12, 49, 1]" = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
        div_tensor: "f32[16, 12, 49, 49]" = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        expand_default: "f32[16, 12, 49, 49]" = torch.ops.aten.expand.default(div_tensor, _shape_param_4);  div_tensor = _shape_param_4 = None
        reshape_default_4: "f32[192, 49, 49]" = torch.ops.aten.reshape.default(expand_default, _shape_param_5);  expand_default = _shape_param_5 = None
        expand_default_1: "f32[16, 12, 49, 32]" = torch.ops.aten.expand.default(select_int, _shape_param_6);  select_int = _shape_param_6 = None
        clone_default_1: "f32[16, 12, 49, 32]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_5: "f32[192, 49, 32]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_7);  clone_default_1 = _shape_param_7 = None
        return (reshape_default_4, reshape_default_5)



def make_inputs():
    return [
    torch.randn([169, 12], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [2401], dtype=torch.int64, device='cuda'),
    [49, 49, -1],  # _shape_param_0
    torch.randn(903168, dtype=torch.float32, device='cuda').as_strided([3, 16, 12, 49, 32], [384, 56448, 32, 1152, 1]),  # permute_104
    torch.randn([192, 49, 49], dtype=torch.float32, device='cuda'),
    [16, 12, 49, 49],  # _shape_param_1
    [4, 4, 12, 49, 49],  # _shape_param_2
    torch.randn([1, 4, 1, 49, 49], dtype=torch.float32, device='cuda'),
    [-1, 12, 49, 49],  # _shape_param_3
    [16, 12, 49, 49],  # _shape_param_4
    [192, 49, 49],  # _shape_param_5
    [16, 12, 49, 32],  # _shape_param_6
    [192, 49, 32],  # _shape_param_7
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
