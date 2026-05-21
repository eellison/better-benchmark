"""
Standalone repro captured via capture_hook.
Label: timm_inception_v3_infer
Pattern hash: a8b0e6ac106c
Shape hash: 316c12ed
"""
_shapes_config = "(T([160], f32), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([160], f32), T([160], f32), T([160], f32), T([160], f32), T([128, 160, 17, 17], f32, stride=(46240, 1, 2720, 160)), T([160], f32), T([160], f32), T([160], f32), T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, arg262_1: "f32[160]", convolution_52: "f32[128, 160, 17, 17]", arg263_1: "f32[160]", arg264_1: "f32[160]", arg265_1: "f32[160]", arg287_1: "f32[160]", convolution_57: "f32[128, 160, 17, 17]", arg288_1: "f32[160]", arg289_1: "f32[160]", arg290_1: "f32[160]", cat_5: "f32[128, 768, 17, 17]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_default_1: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        sub_tensor: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_default_1);  convolution_52 = unsqueeze_default_1 = None
        add_tensor: "f32[160]" = torch.ops.aten.add.Tensor(arg263_1, 0.001);  arg263_1 = None
        sqrt_default: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor);  add_tensor = None
        reciprocal_default: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default);  sqrt_default = None
        mul_tensor: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default, 1);  reciprocal_default = None
        unsqueeze_default_2: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor, -1);  mul_tensor = None
        unsqueeze_default_3: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        mul_tensor_1: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        unsqueeze_default_4: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_default_5: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, -1);  unsqueeze_default_4 = None
        mul_tensor_2: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_1, unsqueeze_default_5);  mul_tensor_1 = unsqueeze_default_5 = None
        unsqueeze_default_6: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_default_7: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, -1);  unsqueeze_default_6 = None
        add_tensor_1: "f32[128, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_2, unsqueeze_default_7);  mul_tensor_2 = unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default: "f32[128, 160, 17, 17]" = torch.ops.aten.relu.default(add_tensor_1);  add_tensor_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_default_8: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_default_9: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, -1);  unsqueeze_default_8 = None
        sub_tensor_1: "f32[128, 160, 17, 17]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_default_9);  convolution_57 = unsqueeze_default_9 = None
        add_tensor_2: "f32[160]" = torch.ops.aten.add.Tensor(arg288_1, 0.001);  arg288_1 = None
        sqrt_default_1: "f32[160]" = torch.ops.aten.sqrt.default(add_tensor_2);  add_tensor_2 = None
        reciprocal_default_1: "f32[160]" = torch.ops.aten.reciprocal.default(sqrt_default_1);  sqrt_default_1 = None
        mul_tensor_3: "f32[160]" = torch.ops.aten.mul.Tensor(reciprocal_default_1, 1);  reciprocal_default_1 = None
        unsqueeze_default_10: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(mul_tensor_3, -1);  mul_tensor_3 = None
        unsqueeze_default_11: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, -1);  unsqueeze_default_10 = None
        mul_tensor_4: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_11);  sub_tensor_1 = unsqueeze_default_11 = None
        unsqueeze_default_12: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_default_13: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, -1);  unsqueeze_default_12 = None
        mul_tensor_5: "f32[128, 160, 17, 17]" = torch.ops.aten.mul.Tensor(mul_tensor_4, unsqueeze_default_13);  mul_tensor_4 = unsqueeze_default_13 = None
        unsqueeze_default_14: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_default_15: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, -1);  unsqueeze_default_14 = None
        add_tensor_3: "f32[128, 160, 17, 17]" = torch.ops.aten.add.Tensor(mul_tensor_5, unsqueeze_default_15);  mul_tensor_5 = unsqueeze_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_default_1: "f32[128, 160, 17, 17]" = torch.ops.aten.relu.default(add_tensor_3);  add_tensor_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_default: "f32[128, 768, 17, 17]" = torch.ops.aten.avg_pool2d.default(cat_5, [3, 3], [1, 1], [1, 1]);  cat_5 = None
        return (relu_default, relu_default_1, avg_pool2d_default)



def make_inputs():
    return [
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # convolution_52
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn(5918720, dtype=torch.float32, device='cuda').as_strided([128, 160, 17, 17], [46240, 1, 2720, 160]),  # convolution_57
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn(28409856, dtype=torch.float32, device='cuda').as_strided([128, 768, 17, 17], [221952, 1, 13056, 768]),  # cat_5
    ]


if __name__ == "__main__":
    mod = Repro()
    inputs = make_inputs()
    compiled = torch.compile(mod)
    with torch.no_grad():
        out = compiled(*inputs)
        torch.cuda.synchronize()
    print("OK")
