"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: fab68b948c58
Shape hash: 8e104bf1
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_68: "f32[768, 49, 32]", _shape_param_0, bmm_70: "f32[768, 32, 49]", _shape_param_1, bmm_71: "f32[768, 49, 32]", _shape_param_2, full_default_119: "f32[3, 256, 3, 49, 32]", _shape_param_3, _shape_param_4, primals_10: "f32[288, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        reshape_default: "f32[256, 3, 49, 32]" = torch.ops.aten.reshape.default(bmm_68, _shape_param_0);  bmm_68 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default_1: "f32[256, 3, 32, 49]" = torch.ops.aten.reshape.default(bmm_70, _shape_param_1);  bmm_70 = _shape_param_1 = None
        reshape_default_2: "f32[256, 3, 49, 32]" = torch.ops.aten.reshape.default(bmm_71, _shape_param_2);  bmm_71 = _shape_param_2 = None
        permute_default: "f32[256, 3, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:188 in shifted_window_attention, code: q = q * (C // num_heads) ** -0.5
        mul_tensor: "f32[256, 3, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_scatter_default: "f32[3, 256, 3, 49, 32]" = torch.ops.aten.select_scatter.default(full_default_119, reshape_default, 0, 2);  reshape_default = None
        select_scatter_default_1: "f32[3, 256, 3, 49, 32]" = torch.ops.aten.select_scatter.default(full_default_119, permute_default, 0, 1);  permute_default = None
        add_tensor: "f32[3, 256, 3, 49, 32]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f32[3, 256, 3, 49, 32]" = torch.ops.aten.select_scatter.default(full_default_119, mul_tensor, 0, 0);  full_default_119 = mul_tensor = None
        add_tensor_1: "f32[3, 256, 3, 49, 32]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:180 in shifted_window_attention, code: qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[256, 49, 3, 3, 32]" = torch.ops.aten.permute.default(add_tensor_1, [1, 3, 0, 2, 4]);  add_tensor_1 = None
        clone_default: "f32[256, 49, 3, 3, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[256, 49, 288]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default_4: "f32[12544, 288]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[96, 288]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_default_3: "f32[288, 96]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_4, permute_default_3)



def make_inputs():
    return [
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    [256, 3, 49, 32],  # _shape_param_0
    torch.randn([768, 32, 49], dtype=torch.float32, device='cuda'),
    [256, 3, 32, 49],  # _shape_param_1
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    [256, 3, 49, 32],  # _shape_param_2
    torch.randn([3, 256, 3, 49, 32], dtype=torch.float32, device='cuda'),
    [256, 49, 288],  # _shape_param_3
    [12544, 288],  # _shape_param_4
    torch.randn([288, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
