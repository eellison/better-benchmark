"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: a21fbca60398
Shape hash: e76a4b46
"""
_shapes_config = "(T([512, 48, 56, 56], f32, stride=(150528, 1, 2688, 48)), T([48], f32), T([48], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_7: "f32[512, 48, 56, 56]", primals_48: "f32[48]", primals_49: "f32[48]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 48, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 48, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_1);  convolution_7 = getitem_1 = None
        mul_tensor: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_default_1: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 48, 56, 56]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_default_3: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 48, 56, 56]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        return add_tensor_1



def make_inputs():
    return [
    torch.randn(77070336, dtype=torch.float32, device='cuda').as_strided([512, 48, 56, 56], [150528, 1, 2688, 48]),  # convolution_7
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
