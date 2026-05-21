"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: 39d066a8802f
Shape hash: 0e850107
"""
_shapes_config = "(T([512, 56, 14, 14], f32, stride=(10976, 1, 784, 56)), T([56], f32), T([56], f32), T([512, 56, 14, 14], f32, stride=(10976, 1, 784, 56)), T([512, 112, 14, 14], f32, stride=(21952, 1, 1568, 112)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_63: "f32[512, 56, 14, 14]", primals_352: "f32[56]", primals_353: "f32[56]", add_288: "f32[512, 56, 14, 14]", add_272: "f32[512, 112, 14, 14]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 56, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 56, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_1);  convolution_63 = getitem_1 = None
        mul_tensor: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_352, -1);  primals_352 = None
        unsqueeze_default_1: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_353, -1);  primals_353 = None
        unsqueeze_default_3: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 112, 14, 14]" = torch.ops.aten.cat.default([add_288, add_tensor_1], 1);  add_288 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_2: "f32[512, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_default, add_272);  cat_default = add_272 = None
        return add_tensor_2



def make_inputs():
    return [
    torch.randn(5619712, dtype=torch.float32, device='cuda').as_strided([512, 56, 14, 14], [10976, 1, 784, 56]),  # convolution_63
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn(5619712, dtype=torch.float32, device='cuda').as_strided([512, 56, 14, 14], [10976, 1, 784, 56]),  # add_288
    torch.randn(11239424, dtype=torch.float32, device='cuda').as_strided([512, 112, 14, 14], [21952, 1, 1568, 112]),  # add_272
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
