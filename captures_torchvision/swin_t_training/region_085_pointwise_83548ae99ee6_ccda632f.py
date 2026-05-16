"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 83548ae99ee6
Shape hash: ccda632f
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_32: "f32[192, 49, 32]", _shape_param_0, bmm_34: "f32[192, 32, 49]", _shape_param_1, bmm_35: "f32[192, 49, 32]", _shape_param_2, _shape_param_3, _shape_param_4, primals_142: "f32[1152, 384]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        reshape_default: "f32[16, 12, 49, 32]" = torch.ops.aten.reshape.default(bmm_32, _shape_param_0);  bmm_32 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        reshape_default_1: "f32[16, 12, 32, 49]" = torch.ops.aten.reshape.default(bmm_34, _shape_param_1);  bmm_34 = _shape_param_1 = None
        reshape_default_2: "f32[16, 12, 49, 32]" = torch.ops.aten.reshape.default(bmm_35, _shape_param_2);  bmm_35 = _shape_param_2 = None
        permute_default: "f32[16, 12, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:188 in shifted_window_attention, code: q = q * (C // num_heads) ** -0.5
        mul_tensor: "f32[16, 12, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        full_default: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.full.default([3, 16, 12, 49, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.select_scatter.default(full_default, reshape_default, 0, 2);  reshape_default = None
        select_scatter_default_1: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.select_scatter.default(full_default, permute_default, 0, 1);  permute_default = None
        add_tensor: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None
        select_scatter_default_2: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.select_scatter.default(full_default, mul_tensor, 0, 0);  full_default = mul_tensor = None
        add_tensor_1: "f32[3, 16, 12, 49, 32]" = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default_2);  add_tensor = select_scatter_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:180 in shifted_window_attention, code: qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[16, 49, 3, 12, 32]" = torch.ops.aten.permute.default(add_tensor_1, [1, 3, 0, 2, 4]);  add_tensor_1 = None
        clone_default: "f32[16, 49, 3, 12, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_3: "f32[16, 49, 1152]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default_4: "f32[784, 1152]" = torch.ops.aten.reshape.default(reshape_default_3, _shape_param_4);  reshape_default_3 = _shape_param_4 = None
        permute_default_2: "f32[384, 1152]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_default_3: "f32[1152, 384]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_4, permute_default_3)



def make_inputs():
    return [
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    [16, 12, 49, 32],  # _shape_param_0
    torch.randn([192, 32, 49], dtype=torch.float32, device='cuda'),
    [16, 12, 32, 49],  # _shape_param_1
    torch.randn([192, 49, 32], dtype=torch.float32, device='cuda'),
    [16, 12, 49, 32],  # _shape_param_2
    [16, 49, 1152],  # _shape_param_3
    [784, 1152],  # _shape_param_4
    torch.randn([1152, 384], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
