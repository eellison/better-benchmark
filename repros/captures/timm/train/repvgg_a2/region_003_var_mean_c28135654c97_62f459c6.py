"""
Standalone repro captured via capture_hook.
Label: timm_repvgg_a2_train
Pattern hash: c28135654c97
Shape hash: 62f459c6
"""
_shapes_config = "(T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32, stride=(150528, 1, 5376, 192)), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, relu_5: "f32[128, 192, 28, 28]", primals_92: "f32[192]", primals_93: "f32[192]", convolution_12: "f32[128, 192, 28, 28]", primals_98: "f32[192]", primals_99: "f32[192]", convolution_13: "f32[128, 192, 28, 28]", primals_104: "f32[192]", primals_105: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(relu_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 192, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 192, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt_default: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_5, getitem_1);  relu_5 = getitem_1 = None
        mul_tensor: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_92, -1);  primals_92 = None
        unsqueeze_default_1: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_93, -1);  primals_93 = None
        unsqueeze_default_3: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 192, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 192, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_default_1: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_3);  convolution_12 = getitem_3 = None
        mul_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_98, -1);  primals_98 = None
        unsqueeze_default_5: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_99, -1);  primals_99 = None
        unsqueeze_default_7: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 192, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 192, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_4: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_default_2: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_2: "f32[128, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_5);  convolution_13 = getitem_5 = None
        mul_tensor_4: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        unsqueeze_default_8: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_104, -1);  primals_104 = None
        unsqueeze_default_9: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[128, 192, 28, 28]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_9);  mul_tensor_4 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_105, -1);  primals_105 = None
        unsqueeze_default_11: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_5: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_11);  mul_tensor_5 = unsqueeze_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:751 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_tensor_6: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_3, add_tensor_5);  add_tensor_3 = add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:753 in forward, code: x += identity
        add_tensor_7: "f32[128, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_tensor_6, add_tensor_1);  add_tensor_6 = add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_default: "f32[128, 192, 28, 28]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None
        return relu_default



def make_inputs():
    return [
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # relu_5
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # convolution_12
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn(19267584, dtype=torch.float32, device='cuda').as_strided([128, 192, 28, 28], [150528, 1, 5376, 192]),  # convolution_13
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
