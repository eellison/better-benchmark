"""
Standalone repro captured via capture_hook.
Label: mobilenet_v2_training
Pattern hash: 7f1c447bbee8
Shape hash: a34a8017
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_34: "f32[1, 32, 1, 1]", convolution_17: "f32[4, 32, 28, 28]", getitem_35: "f32[1, 32, 1, 1]", primals_108: "f32[32]", primals_109: "f32[32]", add_76: "f32[4, 32, 28, 28]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        add_tensor: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_default: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_35);  convolution_17 = getitem_35 = None
        mul_tensor: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_default_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_default_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        add_tensor_2: "f32[4, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_76, add_tensor_1);  add_76 = add_tensor_1 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 28, 28], dtype=torch.float32, device='cuda'),
    torch.randn([1, 32, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([32], dtype=torch.float32, device='cuda'),
    torch.randn([4, 32, 28, 28], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
