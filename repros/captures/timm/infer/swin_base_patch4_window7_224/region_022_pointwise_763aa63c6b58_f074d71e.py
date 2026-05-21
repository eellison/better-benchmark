"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 763aa63c6b58
Shape hash: f074d71e
"""
_shapes_config = "(T([16384, 49, 32], f32), T([256, 256], f32), S([2048, 8, 49, 32]), S([2048, 49, 256]), S([100352, 256]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_7: "f32[16384, 49, 32]", arg58_1: "f32[256, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.reshape.default(bmm_7, _shape_param_0);  bmm_7 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_default: "f32[2048, 49, 8, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[2048, 49, 8, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_2: "f32[100352, 256]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[256, 256]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([16384, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([256, 256], dtype=torch.float32, device='cuda'),
    [2048, 8, 49, 32],  # _shape_param_0
    [2048, 49, 256],  # _shape_param_1
    [100352, 256],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
