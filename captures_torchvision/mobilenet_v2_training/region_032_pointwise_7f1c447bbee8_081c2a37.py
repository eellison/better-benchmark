"""
Standalone repro captured via capture_hook.
Label: mobilenet_v2_training
Pattern hash: 7f1c447bbee8
Shape hash: 081c2a37
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_16: "f32[1, 24, 1, 1]", convolution_8: "f32[4, 24, 56, 56]", getitem_17: "f32[1, 24, 1, 1]", primals_54: "f32[24]", primals_55: "f32[24]", add_29: "f32[4, 24, 56, 56]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        add_tensor: "f32[1, 24, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_default: "f32[1, 24, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17);  convolution_8 = getitem_17 = None
        mul_tensor: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
        unsqueeze_default_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_default_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        add_tensor_2: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_29, add_tensor_1);  add_29 = add_tensor_1 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 24, 56, 56], dtype=torch.float32, device='cuda'),
    torch.randn([1, 24, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([24], dtype=torch.float32, device='cuda'),
    torch.randn([4, 24, 56, 56], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
