"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: ffa1079cb88e
Shape hash: afc3636c
"""
_shapes_config = "(T([512, 56, 14, 14], f32, stride=(10976, 1, 784, 56)), T([56], f32), T([56], f32), T([512, 56, 14, 14], f32, stride=(10976, 1, 784, 56)), T([512, 112, 14, 14], f32, stride=(21952, 1, 1568, 112)), T([112], f32), T([112], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_55: "f32[512, 56, 14, 14]", primals_312: "f32[56]", primals_313: "f32[56]", add_256: "f32[512, 56, 14, 14]", convolution_57: "f32[512, 112, 14, 14]", primals_324: "f32[112]", primals_325: "f32[112]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 56, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 56, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 56, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 56, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 56, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_1);  convolution_55 = getitem_1 = None
        mul_tensor: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
        unsqueeze_default_1: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 56, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[56, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_default_3: "f32[56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 56, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:70 in forward, code: out = torch.cat([x1, x2], dim=1)
        cat_default: "f32[512, 112, 14, 14]" = torch.ops.aten.cat.default([add_256, add_tensor_1], 1);  add_256 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_57, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 112, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 112, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 112, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 112, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[512, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, getitem_3);  convolution_57 = getitem_3 = None
        mul_tensor_2: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_324, -1);  primals_324 = None
        unsqueeze_default_5: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[512, 112, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[112, 1]" = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_default_7: "f32[112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[512, 112, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        add_tensor_4: "f32[512, 112, 14, 14]" = torch.ops.aten.add.Tensor(cat_default, add_tensor_3);  cat_default = add_tensor_3 = None
        return add_tensor_4



def make_inputs():
    return [
    torch.randn(5619712, dtype=torch.float32, device='cuda').as_strided([512, 56, 14, 14], [10976, 1, 784, 56]),  # convolution_55
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn([56], dtype=torch.float32, device='cuda'),
    torch.randn(5619712, dtype=torch.float32, device='cuda').as_strided([512, 56, 14, 14], [10976, 1, 784, 56]),  # add_256
    torch.randn(11239424, dtype=torch.float32, device='cuda').as_strided([512, 112, 14, 14], [21952, 1, 1568, 112]),  # convolution_57
    torch.randn([112], dtype=torch.float32, device='cuda'),
    torch.randn([112], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
