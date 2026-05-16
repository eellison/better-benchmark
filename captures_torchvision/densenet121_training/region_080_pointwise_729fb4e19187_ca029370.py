"""
Standalone repro captured via capture_hook.
Label: densenet121_training
Pattern hash: 729fb4e19187
Shape hash: ca029370
"""
import torch
import torch._inductor.config as inductor_config
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, getitem_92: "f32[1, 352, 1, 1]", cat_20: "f32[4, 352, 14, 14]", getitem_93: "f32[1, 352, 1, 1]", primals_275: "f32[352]", primals_276: "f32[352]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_tensor: "f32[1, 352, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_default: "f32[1, 352, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[4, 352, 14, 14]" = torch.ops.aten.sub.Tensor(cat_20, getitem_93);  cat_20 = getitem_93 = None
        mul_tensor: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_275, -1);  primals_275 = None
        unsqueeze_default_1: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[4, 352, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[352, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
        unsqueeze_default_3: "f32[352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[4, 352, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        relu_default: "f32[4, 352, 14, 14]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None
        return relu_default



def make_inputs():
    return [
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 352, 14, 14], dtype=torch.float32, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
