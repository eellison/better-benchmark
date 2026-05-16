"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 729fb4e19187
Shape hash: 4694aee6
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_160: "f32[1, 896, 1, 1]", cat_37: "f32[4, 896, 14, 14]", getitem_161: "f32[1, 896, 1, 1]", primals_479: "f32[896]", primals_480: "f32[896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_tensor: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_default: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 896, 14, 14]" = torch.ops.aten.sub.Tensor(cat_37, getitem_161);  cat_37 = getitem_161 = None
        mul_tensor: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_479, -1);  primals_479 = None
        unsqueeze_default_1: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1);  primals_480 = None
        unsqueeze_default_3: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[4, 896, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 896, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
