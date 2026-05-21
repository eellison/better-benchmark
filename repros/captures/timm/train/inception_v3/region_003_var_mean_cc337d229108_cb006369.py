"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: cc337d229108
Shape hash: cb006369
"""
_shapes_config = "(T([128, 320, 8, 8], f32, stride=(20480, 1, 2560, 320)), T([320], f32), T([320], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32, stride=(24576, 1, 3072, 384)), T([384], f32), T([384], f32), T([128, 192, 8, 8], f32, stride=(12288, 1, 1536, 192)), T([192], f32), T([192], f32))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_76: "f32[128, 320, 8, 8]", primals_462: "f32[320]", primals_463: "f32[320]", convolution_78: "f32[128, 384, 8, 8]", primals_474: "f32[384]", primals_475: "f32[384]", convolution_79: "f32[128, 384, 8, 8]", primals_480: "f32[384]", primals_481: "f32[384]", convolution_82: "f32[128, 384, 8, 8]", primals_498: "f32[384]", primals_499: "f32[384]", convolution_83: "f32[128, 384, 8, 8]", primals_504: "f32[384]", primals_505: "f32[384]", convolution_84: "f32[128, 192, 8, 8]", primals_510: "f32[192]", primals_511: "f32[192]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_76, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 320, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 320, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 320, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 320, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_1);  convolution_76 = getitem_1 = None
        mul_tensor: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_462, -1);  primals_462 = None
        unsqueeze_default_1: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[320, 1]" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_default_3: "f32[320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 320, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 320, 8, 8]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 384, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 384, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001);  getitem_2 = None
        rsqrt_default_1: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_3);  convolution_78 = getitem_3 = None
        mul_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_474, -1);  primals_474 = None
        unsqueeze_default_5: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
        unsqueeze_default_7: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 384, 1, 1]" = var_mean_correction_2[0]
        getitem_5: "f32[1, 384, 1, 1]" = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_4: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 0.001);  getitem_4 = None
        rsqrt_default_2: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_2: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_5);  convolution_79 = getitem_5 = None
        mul_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        unsqueeze_default_8: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_480, -1);  primals_480 = None
        unsqueeze_default_9: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        mul_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_9);  mul_tensor_4 = unsqueeze_default_9 = None
        unsqueeze_default_10: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_default_11: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        add_tensor_5: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_11);  mul_tensor_5 = unsqueeze_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_2: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_5);  add_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_default: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_1, relu_default_2], 1);  relu_default_1 = relu_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convolution_82, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 384, 1, 1]" = var_mean_correction_3[0]
        getitem_7: "f32[1, 384, 1, 1]" = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_6: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 0.001);  getitem_6 = None
        rsqrt_default_3: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_3: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_82, getitem_7);  convolution_82 = getitem_7 = None
        mul_tensor_6: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = rsqrt_default_3 = None
        unsqueeze_default_12: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_498, -1);  primals_498 = None
        unsqueeze_default_13: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_6, unsqueeze_default_13);  mul_tensor_6 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
        unsqueeze_default_15: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_7: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_7, unsqueeze_default_15);  mul_tensor_7 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_3: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_7);  add_tensor_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 384, 1, 1]" = var_mean_correction_4[0]
        getitem_9: "f32[1, 384, 1, 1]" = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_8: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 0.001);  getitem_8 = None
        rsqrt_default_4: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_4: "f32[128, 384, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_9);  convolution_83 = getitem_9 = None
        mul_tensor_8: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_4);  sub_tensor_4 = rsqrt_default_4 = None
        unsqueeze_default_16: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_504, -1);  primals_504 = None
        unsqueeze_default_17: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, -1);  unsqueeze_default_16 = None
        mul_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_8, unsqueeze_default_17);  mul_tensor_8 = unsqueeze_default_17 = None
        unsqueeze_default_18: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_505, -1);  primals_505 = None
        unsqueeze_default_19: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, -1);  unsqueeze_default_18 = None
        add_tensor_9: "f32[128, 384, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_9, unsqueeze_default_19);  mul_tensor_9 = unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_4: "f32[128, 384, 8, 8]" = torch.ops.aten.relu.default(add_tensor_9);  add_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_default_1: "f32[128, 768, 8, 8]" = torch.ops.aten.cat.default([relu_default_3, relu_default_4], 1);  relu_default_3 = relu_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 192, 1, 1]" = var_mean_correction_5[0]
        getitem_11: "f32[1, 192, 1, 1]" = var_mean_correction_5[1];  var_mean_correction_5 = None
        add_tensor_10: "f32[1, 192, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 0.001);  getitem_10 = None
        rsqrt_default_5: "f32[1, 192, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        sub_tensor_5: "f32[128, 192, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_11);  convolution_84 = getitem_11 = None
        mul_tensor_10: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_5);  sub_tensor_5 = rsqrt_default_5 = None
        unsqueeze_default_20: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_510, -1);  primals_510 = None
        unsqueeze_default_21: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, -1);  unsqueeze_default_20 = None
        mul_tensor_11: "f32[128, 192, 8, 8]" = torch.ops.aten.mul.Tensor(mul_tensor_10, unsqueeze_default_21);  mul_tensor_10 = unsqueeze_default_21 = None
        unsqueeze_default_22: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
        unsqueeze_default_23: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, -1);  unsqueeze_default_22 = None
        add_tensor_11: "f32[128, 192, 8, 8]" = torch.ops.aten.add.Tensor(mul_tensor_11, unsqueeze_default_23);  mul_tensor_11 = unsqueeze_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_5: "f32[128, 192, 8, 8]" = torch.ops.aten.relu.default(add_tensor_11);  add_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        cat_default_2: "f32[128, 2048, 8, 8]" = torch.ops.aten.cat.default([relu_default, cat_default, cat_default_1, relu_default_5], 1);  relu_default = cat_default = cat_default_1 = relu_default_5 = None
        return cat_default_2



def make_inputs():
    return [
    torch.randn(2621440, dtype=torch.float32, device='cuda').as_strided([128, 320, 8, 8], [20480, 1, 2560, 320]),  # convolution_76
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_78
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_79
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_82
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(3145728, dtype=torch.float32, device='cuda').as_strided([128, 384, 8, 8], [24576, 1, 3072, 384]),  # convolution_83
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(1572864, dtype=torch.float32, device='cuda').as_strided([128, 192, 8, 8], [12288, 1, 1536, 192]),  # convolution_84
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
