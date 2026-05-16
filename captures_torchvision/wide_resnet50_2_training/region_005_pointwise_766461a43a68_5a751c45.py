"""
Standalone repro captured via capture_hook.
Label: wide_resnet50_2_training
Pattern hash: 766461a43a68
Shape hash: 5a751c45
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_92: "f32[1, 2048, 1, 1]", convolution_45: "f32[4, 2048, 7, 7]", getitem_93: "f32[1, 2048, 1, 1]", primals_276: "f32[2048]", primals_277: "f32[2048]", getitem_94: "f32[1, 2048, 1, 1]", convolution_46: "f32[4, 2048, 7, 7]", getitem_95: "f32[1, 2048, 1, 1]", primals_282: "f32[2048]", primals_283: "f32[2048]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        add_tensor: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_default: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_93);  convolution_45 = getitem_93 = None
        mul_tensor: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
        unsqueeze_default_1: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_default_3: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        add_tensor_2: "f32[1, 2048, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_default_1: "f32[1, 2048, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[4, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_95);  convolution_46 = getitem_95 = None
        mul_tensor_2: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_282, -1);  primals_282 = None
        unsqueeze_default_5: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[4, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_default_7: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_tensor_4: "f32[4, 2048, 7, 7]" = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor_3);  add_tensor_1 = add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_default: "f32[4, 2048, 7, 7]" = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 2048, 7, 7], dtype=torch.float32, device='cuda'),
    torch.randn([1, 2048, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
