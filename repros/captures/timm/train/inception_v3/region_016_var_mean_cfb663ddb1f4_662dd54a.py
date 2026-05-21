"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: cfb663ddb1f4
Shape hash: 662dd54a
"""
_shapes_config = "(T([128, 384, 17, 17], f32, stride=(110976, 1, 6528, 384)), T([384], f32), T([384], f32), T([128, 96, 17, 17], f32, stride=(27744, 1, 1632, 96)), T([96], f32), T([96], f32), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_26: "f32[128, 384, 17, 17]", primals_162: "f32[384]", primals_163: "f32[384]", convolution_29: "f32[128, 96, 17, 17]", primals_180: "f32[96]", primals_181: "f32[96]", cat_2: "f32[128, 288, 35, 35]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 384, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 384, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 384, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 384, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 384, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_1);  convolution_26 = getitem_1 = None
        mul_tensor: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
        unsqueeze_default_1: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 384, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[384, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_default_3: "f32[384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 384, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 384, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 96, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 96, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 96, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001);  getitem_2 = None
        rsqrt_default_1: "f32[1, 96, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 96, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_3);  convolution_29 = getitem_3 = None
        mul_tensor_2: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
        unsqueeze_default_5: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 96, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_default_7: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 96, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 96, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:93 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False);  cat_2 = None
        getitem_4: "f32[128, 288, 17, 17]" = _low_memory_max_pool_with_offsets_default[0]
        getitem_5: "i8[128, 288, 17, 17]" = _low_memory_max_pool_with_offsets_default[1];  _low_memory_max_pool_with_offsets_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:100 in forward, code: return torch.cat(outputs, 1)
        cat_default: "f32[128, 768, 17, 17]" = torch.ops.aten.cat.default([relu_default, relu_default_1, getitem_4], 1);  relu_default = relu_default_1 = getitem_4 = None
        return (cat_default, getitem_5)



def make_inputs():
    return [
    torch.randn(14204928, dtype=torch.float32, device='cuda').as_strided([128, 384, 17, 17], [110976, 1, 6528, 384]),  # convolution_26
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn(3551232, dtype=torch.float32, device='cuda').as_strided([128, 96, 17, 17], [27744, 1, 1632, 96]),  # convolution_29
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn(45158400, dtype=torch.float32, device='cuda').as_strided([128, 288, 35, 35], [352800, 1, 10080, 288]),  # cat_2
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
