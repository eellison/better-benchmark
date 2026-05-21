"""
Standalone repro captured via capture_hook.
Label: timm_visformer_small_infer
Pattern hash: 8e1dc74da682
Shape hash: 604edb77
"""
_shapes_config = "(T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, add_34: "f32[128, 192, 28, 28]", convolution_22: "f32[128, 192, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        add_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_34, convolution_22);  add_34 = convolution_22 = None
        return add_tensor



def make_inputs():
    return [
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # add_34
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # convolution_22
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
