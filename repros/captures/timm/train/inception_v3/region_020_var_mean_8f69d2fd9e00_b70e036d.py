"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 8f69d2fd9e00
Shape hash: b70e036d
"""
_shapes_config = "(T([128, 48, 35, 35], f32, stride=(58800, 1, 1680, 48)), T([48], f32), T([48], f32), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([96], f32), T([96], f32), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_20: "f32[128, 48, 35, 35]", primals_126: "f32[48]", primals_127: "f32[48]", convolution_23: "f32[128, 96, 35, 35]", primals_144: "f32[96]", primals_145: "f32[96]", cat_1: "f32[128, 288, 35, 35]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_20, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 48, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 48, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 48, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 48, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 48, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_1);  convolution_20 = getitem_1 = None
        mul_tensor: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_default_1: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 48, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[48, 1]" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_default_3: "f32[48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 48, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 48, 35, 35]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 96, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 96, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001);  getitem_2 = None
        rsqrt_default_1: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 96, 35, 35]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_3);  convolution_23 = getitem_3 = None
        mul_tensor_2: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
        unsqueeze_default_5: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_default_7: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 96, 35, 35]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 96, 35, 35]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_default: "f32[128, 288, 35, 35]" = torch.ops.aten.avg_pool2d.default(cat_1, [3, 3], [1, 1], [1, 1]);  cat_1 = None
        return (relu_default, relu_default_1, avg_pool2d_default)



def make_inputs():
    return [
    torch.randn(7526400, dtype=torch.float32, device='cuda').as_strided([128, 48, 35, 35], [58800, 1, 1680, 48]),  # convolution_20
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn([48], dtype=torch.float32, device='cuda'),
    torch.randn(15052800, dtype=torch.float32, device='cuda').as_strided([128, 96, 35, 35], [117600, 1, 3360, 96]),  # convolution_23
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(45158400, dtype=torch.float32, device='cuda').as_strided([128, 288, 35, 35], [352800, 1, 10080, 288]),  # cat_1
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
