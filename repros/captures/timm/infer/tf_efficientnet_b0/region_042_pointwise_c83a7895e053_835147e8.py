"""
Standalone repro captured via capture_hook.
Label: timm_tf_efficientnet_b0_infer
Pattern hash: c83a7895e053
Shape hash: 835147e8
"""
_shapes_config = "(T([128, 4, 1, 1], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_7: "f32[128, 4, 1, 1]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_default: "f32[128, 4, 1, 1]" = torch.ops.aten.neg.default(convolution_7)
        exp_default: "f32[128, 4, 1, 1]" = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor: "f32[128, 4, 1, 1]" = torch.ops.aten.add.Tensor(exp_default, 1);  exp_default = None
        div_tensor: "f32[128, 4, 1, 1]" = torch.ops.aten.div.Tensor(convolution_7, add_tensor);  convolution_7 = add_tensor = None
        return div_tensor



def make_inputs():
    return [
    torch.randn([128, 4, 1, 1], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
