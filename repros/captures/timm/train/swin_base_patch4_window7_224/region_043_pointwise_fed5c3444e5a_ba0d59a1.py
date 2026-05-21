"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: fed5c3444e5a
Shape hash: ba0d59a1
"""
_shapes_config = "(T([32768, 49, 32], f32), T([32768, 32, 49], f32), T([32768, 49, 32], f32), T([384, 128], f32), S([8192, 4, 49, 32]), S([8192, 4, 32, 49]), S([8192, 4, 49, 32]), S([3, 8192, 4, 49, 32]), S([8192, 49, 384]), S([401408, 384]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_140: "f32[32768, 49, 32]", bmm_142: "f32[32768, 32, 49]", bmm_143: "f32[32768, 49, 32]", primals_8: "f32[384, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_140, _shape_param_0);  bmm_140 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[8192, 4, 32, 49]" = torch.ops.aten.reshape.default(bmm_142, _shape_param_1);  bmm_142 = _shape_param_1 = None
        reshape_default_2: "f32[8192, 4, 49, 32]" = torch.ops.aten.reshape.default(bmm_143, _shape_param_2);  bmm_143 = _shape_param_2 = None
        permute_default: "f32[8192, 4, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[8192, 4, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[24576, 4, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, reshape_default]);  mul_tensor = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 8192, 4, 49, 32]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.permute.default(reshape_default_3, [1, 3, 0, 2, 4]);  reshape_default_3 = None
        clone_default: "f32[8192, 49, 3, 4, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[8192, 49, 384]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        reshape_default_5: "f32[401408, 384]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_default_3: "f32[384, 128]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_5, permute_default_3)



def make_inputs():
    return [
    torch.randn([32768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([32768, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([384, 128], dtype=torch.float32, device='cuda'),
    [8192, 4, 49, 32],  # _shape_param_0
    [8192, 4, 32, 49],  # _shape_param_1
    [8192, 4, 49, 32],  # _shape_param_2
    [3, 8192, 4, 49, 32],  # _shape_param_3
    [8192, 49, 384],  # _shape_param_4
    [401408, 384],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
