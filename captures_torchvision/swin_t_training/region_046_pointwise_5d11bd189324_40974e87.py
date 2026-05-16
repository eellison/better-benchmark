"""
Standalone repro captured via capture_hook.
Label: swin_t_training
Pattern hash: 5d11bd189324
Shape hash: 40974e87
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, bmm_3: "f32[768, 49, 32]", _shape_param_0, _shape_param_1, _shape_param_2, primals_25: "f32[96, 96]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:214 in shifted_window_attention, code: x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        reshape_default: "f32[256, 3, 49, 32]" = torch.ops.aten.reshape.default(bmm_3, _shape_param_0);  bmm_3 = _shape_param_0 = None
        permute_default: "f32[256, 49, 3, 32]" = torch.ops.aten.permute.default(reshape_default, [0, 2, 1, 3]);  reshape_default = None
        clone_default: "f32[256, 49, 3, 32]" = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        reshape_default_1: "f32[256, 49, 96]" = torch.ops.aten.reshape.default(clone_default, _shape_param_1);  clone_default = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/swin_transformer.py:215 in shifted_window_attention, code: x = F.linear(x, proj_weight, proj_bias)
        reshape_default_2: "f32[12544, 96]" = torch.ops.aten.reshape.default(reshape_default_1, _shape_param_2);  reshape_default_1 = _shape_param_2 = None
        permute_default_1: "f32[96, 96]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        return (reshape_default_2, permute_default_1)



def make_inputs():
    return [
    torch.randn([768, 49, 32], dtype=torch.float32, device='cuda'),
    [256, 3, 49, 32],  # _shape_param_0
    [256, 49, 96],  # _shape_param_1
    [12544, 96],  # _shape_param_2
    torch.randn([96, 96], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
