"""
Standalone repro captured via capture_hook.
Label: timm_ghostnet_100_train
Pattern hash: b26398eed336
Shape hash: 60ffa92d
"""
_shapes_config = "(T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([40], f32), T([40], f32), T([512, 40, 14, 14], f32, stride=(7840, 1, 560, 40)), T([40], f32), T([40], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_34: "f32[512, 40, 14, 14]", primals_194: "f32[40]", primals_195: "f32[40]", convolution_36: "f32[512, 40, 14, 14]", primals_206: "f32[40]", primals_207: "f32[40]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 40, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 40, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_1);  convolution_34 = getitem_1 = None
        mul_tensor: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1);  primals_194 = None
        unsqueeze_default_1: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
        unsqueeze_default_3: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 40, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 40, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 40, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 40, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[512, 40, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_3);  convolution_36 = getitem_3 = None
        mul_tensor_2: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1);  primals_206 = None
        unsqueeze_default_5: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[512, 40, 14, 14]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[40, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
        unsqueeze_default_7: "f32[40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[512, 40, 14, 14]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        return (add_tensor_1, add_tensor_3)



def make_inputs():
    return [
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # convolution_34
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn(4014080, dtype=torch.float32, device='cuda').as_strided([512, 40, 14, 14], [7840, 1, 560, 40]),  # convolution_36
    torch.randn([40], dtype=torch.float32, device='cuda'),
    torch.randn([40], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
