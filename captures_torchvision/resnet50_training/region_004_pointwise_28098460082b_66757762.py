"""
Standalone repro captured via capture_hook.
Label: resnet50_training
Pattern hash: 28098460082b
Shape hash: 66757762
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_100: "f32[1, 2048, 1, 1]", convolution_49: "f32[4, 2048, 7, 7]", getitem_101: "f32[1, 2048, 1, 1]", primals_300: "f32[2048]", primals_301: "f32[2048]", relu_42: "f32[4, 2048, 7, 7]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_tensor: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_default: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_101);  convolution_49 = getitem_101 = None
        mul_tensor: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_2: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, relu_42);  add_tensor_1 = relu_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
