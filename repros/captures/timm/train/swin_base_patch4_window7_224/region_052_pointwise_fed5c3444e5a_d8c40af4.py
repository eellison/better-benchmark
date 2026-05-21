"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: fed5c3444e5a
Shape hash: d8c40af4
"""
_shapes_config = "(T([16384, 49, 32], f32), T([16384, 32, 49], f32), T([16384, 49, 32], f32), T([768, 256], f32), S([2048, 8, 49, 32]), S([2048, 8, 32, 49]), S([2048, 8, 49, 32]), S([3, 2048, 8, 49, 32]), S([2048, 49, 768]), S([100352, 768]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_132: "f32[16384, 49, 32]", bmm_134: "f32[16384, 32, 49]", bmm_135: "f32[16384, 49, 32]", primals_40: "f32[768, 256]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_132, _shape_param_0);  bmm_132 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:235 in forward, code: attn = q @ k.transpose(-2, -1)
        reshape_default_1: "f32[2048, 8, 32, 49]" = torch.ops.aten.reshape.default(bmm_134, _shape_param_1);  bmm_134 = _shape_param_1 = None
        reshape_default_2: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_135, _shape_param_2);  bmm_135 = _shape_param_2 = None
        permute_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 1, 3, 2]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:234 in forward, code: q = q * self.scale
        mul_tensor: "f32[2048, 8, 49, 32]" = torch.ops.aten.mul.Tensor(reshape_default_2, 0.1767766952966369);  reshape_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:220 in forward, code: q, k, v = qkv.unbind(0)
        cat_default: "f32[6144, 8, 49, 32]" = torch.ops.aten.cat.default([mul_tensor, permute_default, reshape_default]);  mul_tensor = permute_default = reshape_default = None
        reshape_default_3: "f32[3, 2048, 8, 49, 32]" = torch.ops.aten.reshape.default(cat_default, _shape_param_3);  cat_default = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_1: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.permute.default(reshape_default_3, [1, 3, 0, 2, 4]);  reshape_default_3 = None
        clone_default: "f32[2048, 49, 3, 8, 32]" = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        reshape_default_4: "f32[2048, 49, 768]" = torch.ops.aten.reshape.default(clone_default, _shape_param_4);  clone_default = _shape_param_4 = None
        reshape_default_5: "f32[100352, 768]" = torch.ops.aten.reshape.default(reshape_default_4, _shape_param_5);  reshape_default_4 = _shape_param_5 = None
        permute_default_2: "f32[256, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_default_3: "f32[768, 256]" = torch.ops.aten.permute.default(permute_default_2, [1, 0]);  permute_default_2 = None
        return (reshape_default_5, permute_default_3)



def make_inputs():
    return [
    torch.randn([16384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 32, 49], dtype=torch.float32, device='cuda'),
    torch.randn([16384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768, 256], dtype=torch.float32, device='cuda'),
    [2048, 8, 49, 32],  # _shape_param_0
    [2048, 8, 32, 49],  # _shape_param_1
    [2048, 8, 49, 32],  # _shape_param_2
    [3, 2048, 8, 49, 32],  # _shape_param_3
    [2048, 49, 768],  # _shape_param_4
    [100352, 768],  # _shape_param_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
