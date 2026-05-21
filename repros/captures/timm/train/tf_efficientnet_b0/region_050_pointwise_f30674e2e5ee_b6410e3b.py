"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_train
Pattern hash: f30674e2e5ee
Shape hash: b6410e3b
"""
_shapes_config = "(T([128, 3, 224, 224], f32, stride=(150528, 1, 672, 3)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, primals_2: "f32[128, 3, 224, 224]"):
        # File: /tmp/pytorch-work/torch/nn/functional.py:5461 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_default: "f32[128, 3, 225, 225]" = torch.ops.aten.constant_pad_nd.default(primals_2, [0, 1, 0, 1], 0.0);  primals_2 = None
        return constant_pad_nd_default



def make_inputs():
    return [
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 3, 224, 224], [150528, 1, 672, 3]),  # primals_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
