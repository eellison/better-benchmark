"""
Standalone repro captured via capture_hook.
Label: efficientnet_b0_training
Pattern hash: 818690bf8db1
Shape hash: cf79ba7f
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_7: "f32[4, 4, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/ops/misc.py:255 in _scale, code: scale = self.activation(scale)
        neg_default: "f32[4, 4, 1, 1]" = torch.ops.aten.neg.default(convolution_7)
        exp_default: "f32[4, 4, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[4, 4, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[4, 4, 1, 1]" = torch.ops.aten.div.Tensor(convolution_7, add_tensor);  convolution_7 = add_tensor = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([4, 4, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
