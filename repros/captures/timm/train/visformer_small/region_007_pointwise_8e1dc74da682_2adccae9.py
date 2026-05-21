"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_train
Pattern hash: 8e1dc74da682
Shape hash: 2adccae9
"""
_shapes_config = "(T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)), T([128, 384, 14, 14], f32, stride=(75264, 1, 5376, 384)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_117: "f32[128, 384, 14, 14]", convolution_39: "f32[128, 384, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_117, convolution_39);  add_117 = convolution_39 = None
        return add_tensor



def make_inputs():
    return [
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # add_117
    torch.randn(9633792, dtype=torch.float32, device='cuda').as_strided([128, 384, 14, 14], [75264, 1, 5376, 384]),  # convolution_39
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
