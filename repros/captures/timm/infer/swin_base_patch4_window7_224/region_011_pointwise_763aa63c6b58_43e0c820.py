"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_infer
Pattern hash: 763aa63c6b58
Shape hash: 43e0c820
"""
_shapes_config = "(T([8192, 49, 32], f32), T([512, 512], f32), S([512, 16, 49, 32]), S([512, 49, 512]), S([25088, 512]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_43: "f32[8192, 49, 32]", arg322_1: "f32[512, 512]", _shape_param_0, _shape_param_1, _shape_param_2):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:243 in forward, code: x = attn @ v
        reshape_default: "f32[512, 16, 49, 32]" = torch.ops.aten.reshape.default(bmm_43, _shape_param_0);  bmm_43 = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:245 in forward, code: x = x.transpose(1, 2).reshape(B_, N, -1)
        permute_default: "f32[512, 49, 16, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[512, 49, 16, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[512, 49, 512]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        reshape_default_2: "f32[25088, 512]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([8192, 49, 32], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    [512, 16, 49, 32],  # _shape_param_0
    [512, 49, 512],  # _shape_param_1
    [25088, 512],  # _shape_param_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
