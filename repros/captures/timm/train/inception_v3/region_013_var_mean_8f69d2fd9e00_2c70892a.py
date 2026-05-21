"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_train
Pattern hash: 8f69d2fd9e00
Shape hash: 2c70892a
"""
_shapes_config = "(T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128], f32), T([128], f32), T([128, 128, 17, 17], f32, stride=(36992, 1, 2176, 128)), T([128], f32), T([128], f32), T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, convolution_32: "f32[128, 128, 17, 17]", primals_198: "f32[128]", primals_199: "f32[128]", convolution_37: "f32[128, 128, 17, 17]", primals_228: "f32[128]", primals_229: "f32[128]", cat_3: "f32[128, 768, 17, 17]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 128, 1, 1]" = var_mean_correction[0]
        getitem_1: "f32[1, 128, 1, 1]" = var_mean_correction[1];  var_mean_correction = None
        add_tensor: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 0.001);  getitem = None
        rsqrt_default: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_1);  convolution_32 = getitem_1 = None
        mul_tensor: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
        unsqueeze_default_1: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_default_3: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_1: "f32[128, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3);  mul_tensor_1 = unsqueeze_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 128, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 128, 1, 1]" = var_mean_correction_1[0]
        getitem_3: "f32[1, 128, 1, 1]" = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2: "f32[1, 128, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 0.001);  getitem_2 = None
        rsqrt_default_1: "f32[1, 128, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1: "f32[128, 128, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_3);  convolution_37 = getitem_3 = None
        mul_tensor_2: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        unsqueeze_default_4: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
        unsqueeze_default_5: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_3: "f32[128, 128, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_2, unsqueeze_default_5);  mul_tensor_2 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_default_7: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_3: "f32[128, 128, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_3, unsqueeze_default_7);  mul_tensor_3 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 128, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_default: "f32[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_3, [3, 3], [1, 1], [1, 1]);  cat_3 = None
        return (relu_default, relu_default_1, avg_pool2d_default)



def make_inputs():
    return [
    torch.randn(4734976, dtype=torch.float32, device='cuda').as_strided([128, 128, 17, 17], [36992, 1, 2176, 128]),  # convolution_32
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn(4734976, dtype=torch.float32, device='cuda').as_strided([128, 128, 17, 17], [36992, 1, 2176, 128]),  # convolution_37
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn(28409856, dtype=torch.float32, device='cuda').as_strided([128, 768, 17, 17], [221952, 1, 13056, 768]),  # cat_3
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
