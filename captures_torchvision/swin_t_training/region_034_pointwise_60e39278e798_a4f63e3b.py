"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 60e39278e798
Shape hash: a4f63e3b
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, addmm_12: "f32[3136, 576]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:179 in shifted_window_attention, code: qkv = F.linear(x, qkv_weight, qkv_bias)
        reshape_default: "f32[64, 49, 576]" = torch.ops.aten.reshape.default(addmm_12, _shape_param_0);  addmm_12 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:180 in shifted_window_attention, code: qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        reshape_default_1: "f32[64, 49, 3, 6, 32]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[3, 64, 6, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [2, 0, 3, 1, 4]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:181 in shifted_window_attention, code: q, k, v = qkv[0], qkv[1], qkv[2]
        select_int: "f32[64, 6, 49, 32]" = torch.ops.aten.select.int(permute_default, 0, 0)
        select_int_1: "f32[64, 6, 49, 32]" = torch.ops.aten.select.int(permute_default, 0, 1);  permute_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:188 in shifted_window_attention, code: q = q * (C // num_heads) ** -0.5
        mul_tensor: "f32[64, 6, 49, 32]" = torch.ops.aten.mul.Tensor(select_int, 0.1767766952966369);  select_int = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:189 in shifted_window_attention, code: attn = q.matmul(k.transpose(-2, -1))
        permute_default_1: "f32[64, 6, 32, 49]" = torch.ops.aten.permute.default(select_int_1, [0, 1, 3, 2]);  select_int_1 = None
        expand_default: "f32[64, 6, 49, 32]" = torch.ops.aten.expand.default(mul_tensor, _shape_param_2);  mul_tensor = _shape_param_2 = None
        clone_default: "f32[64, 6, 49, 32]" = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        reshape_default_2: "f32[384, 49, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_3);  clone_default = _shape_param_3 = None
        expand_default_1: "f32[64, 6, 32, 49]" = torch.ops.aten.expand.default(permute_default_1, _shape_param_4);  permute_default_1 = _shape_param_4 = None
        clone_default_1: "f32[64, 6, 32, 49]" = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        reshape_default_3: "f32[384, 32, 49]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_5);  clone_default_1 = _shape_param_5 = None
        return (reshape_default_2, reshape_default_3)



def make_inputs():
    return [
    torch.randn([3136, 576], dtype=torch.float32, device='cuda'),
    [64, 49, 576],  # _shape_param_0
    [64, 49, 3, 6, 32],  # _shape_param_1
    [64, 6, 49, 32],  # _shape_param_2
    [384, 49, 32],  # _shape_param_3
    [64, 6, 32, 49],  # _shape_param_4
    [384, 32, 49],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
