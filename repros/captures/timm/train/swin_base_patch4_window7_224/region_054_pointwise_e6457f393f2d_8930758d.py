"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: e6457f393f2d
Shape hash: 8930758d
"""
_shapes_config = "(T([100352, 256], f32), S([2048, 49, 256]), S([2048, 49, 8, 32]), S([16384, 49, 32]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, mm_181: "f32[100352, 256]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default: "f32[2048, 49, 256]" = torch.ops.aten.reshape.default(mm_181, _shape_param_0);  mm_181 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        reshape_default_1: "f32[2048, 49, 8, 32]" = torch.ops.aten.reshape.default(reshape_default, _shape_param_1);  reshape_default = _shape_param_1 = None
        permute_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.permute.default(reshape_default_1, [0, 2, 1, 3]);  reshape_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        clone_default: "f32[2048, 8, 49, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_2: "f32[16384, 49, 32]" = torch.ops.aten.reshape.default(clone_default, _shape_param_2);  clone_default = _shape_param_2 = None
        return reshape_default_2



def make_inputs():
    return [
    torch.randn([100352, 256], dtype=torch.float32, device='cuda'),
    [2048, 49, 256],  # _shape_param_0
    [2048, 49, 8, 32],  # _shape_param_1
    [16384, 49, 32],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
